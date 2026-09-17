// Built with production device fast-math flags. No inference dispatch calls
// these kernels; original GGUF blocks are unpacked only into registers.
#include "../ds4_rocm.h"
#include "../rocm/ds4_rocm_q4k_types.cuh"
#include "glm5_q4k_fullrow_test.hpp"

__device__ static float dev_f16_to_f32(uint16_t v) {
    return __half2float(*reinterpret_cast<const __half *>(&v));
}
#include "../rocm/ds4_rocm_q4k_dot.cuh"

__global__ void q4k_fullrow_dp4a(const cuda_block_q4_K *w,
        const cuda_block_q8_K *x, float *out, unsigned n, unsigned blocks) {
    const unsigned lane = threadIdx.x & 7u;
    const unsigned row = blockIdx.x*32 + threadIdx.x/8;
    if (row >= n) return;
    float sum = 0;
    for (unsigned b = lane; b < blocks; b += 8)
        sum += dev_dot_q4_K_q8_K_block(w+row*blocks+b, x+blockIdx.y*blocks+b);
    for (unsigned offset = 4; offset; offset /= 2)
        sum += __shfl_down(sum, offset, 8);
    if (!lane) out[blockIdx.y*n+row] = sum;
}

// The stronger cold-prefill control shares unpacked weights across eight
// tokens with the actual production helper and stages its activation tile.
// This is one projection; routing and the fused gate/up pair remain outside
// this leaf experiment and must be measured before integration claims.
__global__ void q4k_fullrow_dp4a8(const cuda_block_q4_K *w,
        const cuda_block_q8_K *x, float *out, unsigned n, unsigned m,
        unsigned blocks) {
    const unsigned lane=threadIdx.x&7u, row=blockIdx.x*32+threadIdx.x/8;
    const unsigned first=blockIdx.y*8, count=min(8u,m-first);
    __shared__ cuda_block_q8_K staged[8][16];
    const cuda_block_q8_K *xp[8]={};
    for (unsigned p=0; p<count; ++p) xp[p]=x+(first+p)*blocks;
    if (blocks<=16) {
        for (unsigned i=threadIdx.x; i<count*blocks; i+=blockDim.x)
            staged[i/blocks][i%blocks]=xp[i/blocks][i%blocks];
        __syncthreads();
        for (unsigned p=0; p<count; ++p) xp[p]=staged[p];
    }
    if (row>=n) return;
    float sum[8]={};
    for (unsigned b=lane; b<blocks; b+=8)
        dev_dot_q4_K_q8_K_block8(w+row*blocks+b,
            xp[0] ? xp[0]+b : nullptr, xp[1] ? xp[1]+b : nullptr,
            xp[2] ? xp[2]+b : nullptr, xp[3] ? xp[3]+b : nullptr,
            xp[4] ? xp[4]+b : nullptr, xp[5] ? xp[5]+b : nullptr,
            xp[6] ? xp[6]+b : nullptr, xp[7] ? xp[7]+b : nullptr,count,sum);
    for (unsigned p=0; p<count; ++p) {
        for (unsigned offset=4; offset; offset/=2)
            sum[p]+=__shfl_down(sum[p],offset,8);
        if (!lane) out[(first+p)*n+row]=sum[p];
    }
}

#include "../rocm/ds4_rocm_glm5_q4k_integer.cuh"

template<bool SplitScale>
__global__ void q4k_fullrow_mma(const cuda_block_q4_K *w,
        const cuda_block_q8_K *x, float *out, unsigned n, unsigned m,
        unsigned blocks) {
    const unsigned col=threadIdx.x%16;
    const unsigned wbase=blockIdx.x*16, tbase=blockIdx.y*16;
    const unsigned tok=tbase+col<m ? tbase+col : 0;
    __shared__ float partial[8][256];
    float sum[8]={};
    glm5_q4k_i8_partials<SplitScale>(w+wbase*blocks,x+tok*blocks,blocks,sum);
    const float value=glm5_q4k_i8_reduce(partial,sum);
    const unsigned at=threadIdx.x;
    if (tbase+at%16<m) out[(tbase+at%16)*n+wbase+at/16]=value;
}

hipError_t glm5_q4k_fullrows(const cuda_block_q4_K *w,
        const cuda_block_q8_K *x, float *out, unsigned n, unsigned m,
        unsigned blocks, unsigned mode) {
    if (!w || !x || !out || !n || n%16 || n>65536 || !m || m>256 ||
        !blocks || blocks>64 || mode>3) return hipErrorInvalidValue;
    if (mode==0) q4k_fullrow_dp4a<<<dim3((n+31)/32,m),256>>>(w,x,out,n,blocks);
    else if (mode==3) q4k_fullrow_dp4a8<<<dim3((n+31)/32,(m+7)/8),256>>>(w,x,out,n,m,blocks);
    else if (mode==1) q4k_fullrow_mma<false><<<dim3(n/16,(m+15)/16),256>>>(w,x,out,n,m,blocks);
    else q4k_fullrow_mma<true><<<dim3(n/16,(m+15)/16),256>>>(w,x,out,n,m,blocks);
    return hipGetLastError();
}
