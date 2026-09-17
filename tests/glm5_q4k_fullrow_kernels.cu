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

// Force the same binary addition tree as the quarter-wave shuffles. Fast
// math must not reassociate the cross-wave sums into a different tree.
__device__ __forceinline__ float q4k_ordered_add(float a, float b) {
    float result;
    asm("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
}

template<bool SplitScale>
__global__ void q4k_fullrow_mma(const cuda_block_q4_K *w,
        const cuda_block_q8_K *x, float *out, unsigned n, unsigned m,
        unsigned blocks) {
    using I4 = int32_t __attribute__((ext_vector_type(4)));
    using I8 = int32_t __attribute__((ext_vector_type(8)));
    const unsigned lane = threadIdx.x%32, wave = threadIdx.x/32, col = lane%16;
    const unsigned wbase = blockIdx.x*16, tbase = blockIdx.y*16;
    const unsigned tok = tbase+col < m ? tbase+col : 0;
    __shared__ float partial[8][256];
    float sum[8] = {};
    #pragma unroll 1
    for (unsigned bidx = wave; bidx < blocks; bidx += 8) {
        const auto &wb = w[(wbase+col)*blocks+bidx];
        const auto &xb = x[tok*blocks+bidx];
        I8 low = {}, high = {}, minimum = {};
        #pragma unroll 1
        for (unsigned g = 0; g < 8; ++g) {
            uint8_t scale, mn;
            dev_q4_K_get_scale_min(g,wb.scales,&scale,&mn);
            I8 dot = {};
            I4 am;
            #pragma unroll
            for (unsigned j = 0; j < 4; ++j) am[j] = uint32_t(mn)*0x01010101u;
            #pragma unroll
            for (unsigned half = 0; half < 2; ++half) {
                I4 al, ah, bv;
                #pragma unroll
                for (unsigned j = 0; j < 4; ++j) {
                    const uint32_t packed =
                        (reinterpret_cast<const uint32_t *>(wb.qs+(g/2)*32)[half*4+j]
                         >> ((g%2)*4)) & 0x0f0f0f0fu;
                    al[j] = SplitScale ? packed*uint32_t(scale&7u) : packed;
                    ah[j] = packed*uint32_t(scale>>3u);
                    bv[j] = reinterpret_cast<const int32_t *>(xb.qs+g*32)[half*4+j];
                }
                if constexpr (SplitScale) {
                    low = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,al,true,bv,low,true);
                    high = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,ah,true,bv,high,true);
                    minimum = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,am,true,bv,minimum,true);
                } else {
                    dot = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,al,true,bv,dot,true);
                }
            }
            if constexpr (!SplitScale) {
                #pragma unroll
                for (unsigned i = 0; i < 8; ++i) {
                    uint8_t sc, minval;
                    dev_q4_K_get_scale_min(g,w[(wbase+2*i+lane/16)*blocks+bidx].scales,&sc,&minval);
                    low[i] += int(sc)*dot[i];
                    minimum[i] += int(minval)*int(xb.bsums[2*g]+xb.bsums[2*g+1]);
                }
            }
        }
        #pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
            const auto &result_w = w[(wbase+2*i+lane/16)*blocks+bidx];
            const int total = SplitScale ? low[i]+8*high[i] : low[i];
            const float xd = dev_f16_to_f32(result_w.d);
            const float xmin = dev_f16_to_f32(result_w.dmin);
            sum[i] += xb.d*xd*float(total) - xb.d*xmin*float(minimum[i]);
        }
    }
    #pragma unroll
    for (unsigned i = 0; i < 8; ++i)
        partial[wave][(2*i+lane/16)*16+col] = sum[i];
    __syncthreads();
    const unsigned at = threadIdx.x;
    float s0 = q4k_ordered_add(partial[0][at], partial[4][at]);
    float s1 = q4k_ordered_add(partial[1][at], partial[5][at]);
    float s2 = q4k_ordered_add(partial[2][at], partial[6][at]);
    float s3 = q4k_ordered_add(partial[3][at], partial[7][at]);
    const float value = q4k_ordered_add(q4k_ordered_add(s0,s2),q4k_ordered_add(s1,s3));
    if (tbase+at%16 < m) out[(tbase+at%16)*n+wbase+at/16] = value;
}

hipError_t glm5_q4k_fullrows(const cuda_block_q4_K *w,
        const cuda_block_q8_K *x, float *out, unsigned n, unsigned m,
        unsigned blocks, unsigned mode) {
    if (!w || !x || !out || !n || n%16 || n>65536 || !m || m>256 ||
        !blocks || blocks>64 || mode>2) return hipErrorInvalidValue;
    if (mode==0) q4k_fullrow_dp4a<<<dim3((n+31)/32,m),256>>>(w,x,out,n,blocks);
    else if (mode==1) q4k_fullrow_mma<false><<<dim3(n/16,(m+15)/16),256>>>(w,x,out,n,m,blocks);
    else q4k_fullrow_mma<true><<<dim3(n/16,(m+15)/16),256>>>(w,x,out,n,m,blocks);
    return hipGetLastError();
}
