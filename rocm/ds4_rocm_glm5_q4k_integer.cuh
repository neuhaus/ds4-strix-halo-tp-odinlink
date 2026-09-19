#pragma once

// Original packed Q4_K x Q8_K, one 16x16 tile and eight K-partition waves.
// No persistent weight representation. Callers launch exactly 256 threads,
// supply sixteen valid weight rows and a valid activation row per lane.
// Force the same binary addition tree as the quarter-wave shuffles. Fast
// math must not reassociate the cross-wave sums into a different tree.
__device__ __forceinline__ float q4k_ordered_add(float a, float b) {
    float result;
    asm("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
}

template<bool SplitScale>
__device__ __forceinline__ void glm5_q4k_i8_partials(
        const cuda_block_q4_K *w, const cuda_block_q8_K *x,
        unsigned blocks, float sum[8]) {
    using I4 = int32_t __attribute__((ext_vector_type(4)));
    using I8 = int32_t __attribute__((ext_vector_type(8)));
    const unsigned lane = threadIdx.x%32, wave = threadIdx.x/32, col = lane%16;
    #pragma unroll 1
    for (unsigned bidx = wave; bidx < blocks; bidx += 8) {
        const auto &wb = w[col*blocks+bidx];
        const auto &xb = x[bidx];
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
                    dev_q4_K_get_scale_min(g,w[(2*i+lane/16)*blocks+bidx].scales,&sc,&minval);
                    low[i] += int(sc)*dot[i];
                    minimum[i] += int(minval)*int(xb.bsums[2*g]+xb.bsums[2*g+1]);
                }
            }
        }
        #pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
            const auto &result_w = w[(2*i+lane/16)*blocks+bidx];
            const int total = SplitScale ? low[i]+8*high[i] : low[i];
            const float xd = dev_f16_to_f32(result_w.d);
            const float xmin = dev_f16_to_f32(result_w.dmin);
            sum[i] += xb.d*xd*float(total) - xb.d*xmin*float(minimum[i]);
        }
    }
}

__device__ __forceinline__ float glm5_q4k_i8_reduce(float partial[8][256],
                                                    const float sum[8]) {
    const unsigned lane=threadIdx.x%32, wave=threadIdx.x/32, col=lane%16;
    #pragma unroll
    for (unsigned i=0; i<8; ++i)
        partial[wave][(2*i+lane/16)*16+col]=sum[i];
    __syncthreads();
    const unsigned at=threadIdx.x;
    const float s0=q4k_ordered_add(partial[0][at],partial[4][at]);
    const float s1=q4k_ordered_add(partial[1][at],partial[5][at]);
    const float s2=q4k_ordered_add(partial[2][at],partial[6][at]);
    const float s3=q4k_ordered_add(partial[3][at],partial[7][at]);
    return q4k_ordered_add(q4k_ordered_add(s0,s2),q4k_ordered_add(s1,s3));
}
