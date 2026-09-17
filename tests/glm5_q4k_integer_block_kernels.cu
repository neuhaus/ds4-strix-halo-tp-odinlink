// Match production fast-math flags; host comparisons are compiled precisely.
#include "../ds4_rocm.h"
#include "../rocm/ds4_rocm_q4k_types.cuh"
#include "glm5_q4k_integer_block_test.hpp"

__device__ static float dev_f16_to_f32(uint16_t v) {
    return __half2float(*reinterpret_cast<const __half *>(&v));
}
#include "../rocm/ds4_rocm_q4k_dot.cuh"

template<bool Trace>
__global__ void q4k_block_dp4a(const cuda_block_q4_K *w,
        const cuda_block_q8_K *x, Q4KBlockResult *out, unsigned n, unsigned m) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n*m) return;
    const auto &wb = w[i % n];
    const auto &xb = x[i / n];
    out[i].value = dev_dot_q4_K_q8_K_block(&wb, &xb);
    if constexpr (Trace) {
        int dot = 0, minimum = 0;
        for (unsigned g = 0; g < 8; ++g) {
            uint8_t sc, mn;
            dev_q4_K_get_scale_min(g, wb.scales, &sc, &mn);
            dot += int(sc) * dev_dot_q4_32(wb.qs + (g/2)*32,
                                          xb.qs + g*32, (g%2)*4);
            minimum += int(mn) * int(xb.bsums[2*g] + xb.bsums[2*g+1]);
        }
        out[i].dot = dot;
        out[i].minimum = minimum;
    }
}

// One wave handles sixteen original weight blocks and up to sixteen tokens.
// Unpacking lives only in 1 KiB LDS panels reused for each scale subgroup.
// No persistent weight representation is constructed.
template<bool Trace>
__global__ void q4k_block_mma(const cuda_block_q4_K *w,
        const cuda_block_q8_K *x, Q4KBlockResult *out, unsigned n, unsigned m) {
    using A = rocwmma::fragment<rocwmma::matrix_a,16,16,16,int8_t,rocwmma::row_major>;
    using B = rocwmma::fragment<rocwmma::matrix_b,16,16,16,int8_t,rocwmma::col_major>;
    using C = rocwmma::fragment<rocwmma::accumulator,16,16,16,int32_t>;
    __shared__ int8_t ap[16*32], bp[16*32];
    __shared__ int32_t dots[16*16];
    int total[8] = {}, minimum[8] = {};
    const unsigned lane = threadIdx.x;
    const unsigned wbase = blockIdx.x*16, tbase = blockIdx.y*16;
    #pragma unroll 1
    for (unsigned g = 0; g < 8; ++g) {
        for (unsigned i = lane; i < 16*32; i += 32) {
            const unsigned row = i/32, k = i%32;
            ap[i] = (w[wbase+row].qs[(g/2)*32+k] >> ((g%2)*4)) & 15;
            bp[i] = tbase+row < m ? x[tbase+row].qs[g*32+k] : 0;
        }
        __syncthreads();
        A af; B bf; C cf;
        rocwmma::fill_fragment(cf, 0);
        #pragma unroll
        for (unsigned h = 0; h < 2; ++h) {
            rocwmma::load_matrix_sync(af, ap+h*16, 32);
            rocwmma::load_matrix_sync(bf, bp+h*16, 32);
            rocwmma::mma_sync(cf, af, bf, cf);
        }
        rocwmma::store_matrix_sync(dots, cf, 16, rocwmma::mem_row_major);
        __syncthreads();
        #pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
            const unsigned at = lane+i*32, row = at/16, tok = at%16;
            uint8_t sc, mn;
            dev_q4_K_get_scale_min(g, w[wbase+row].scales, &sc, &mn);
            total[i] += int(sc) * dots[at];
            if (tbase+tok < m) {
                const auto &xb = x[tbase+tok];
                minimum[i] += int(mn) * int(xb.bsums[2*g] + xb.bsums[2*g+1]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
        const unsigned at = lane+i*32, row = at/16, tok = at%16;
        if (tbase+tok >= m) continue;
        const auto &wb = w[wbase+row];
        const auto &xb = x[tbase+tok];
        auto &result = out[(tbase+tok)*n+wbase+row];
        const float xd = dev_f16_to_f32(wb.d), xmin = dev_f16_to_f32(wb.dmin);
        result.value = xb.d * xd * float(total[i]) - xb.d * xmin * float(minimum[i]);
        if constexpr (Trace) {
            result.dot = total[i];
            result.minimum = minimum[i];
        }
    }
}

hipError_t glm5_q4k_integer_blocks(const cuda_block_q4_K *w,
        const cuda_block_q8_K *x, Q4KBlockResult *out, unsigned n, unsigned m,
        bool matrix, bool trace) {
    if (!w || !x || !out || !n || n%16 || !m || m > 256 || n > 65536)
        return hipErrorInvalidValue;
    if (matrix) {
        if (trace) q4k_block_mma<true><<<dim3(n/16,(m+15)/16),32>>>(w,x,out,n,m);
        else q4k_block_mma<false><<<dim3(n/16,(m+15)/16),32>>>(w,x,out,n,m);
    } else {
        if (trace) q4k_block_dp4a<true><<<(n*m+127)/128,128>>>(w,x,out,n,m);
        else q4k_block_dp4a<false><<<(n*m+127)/128,128>>>(w,x,out,n,m);
    }
    return hipGetLastError();
}
