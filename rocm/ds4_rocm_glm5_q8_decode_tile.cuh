/* Research-only Metal-inspired Q8_0 scheduling, adapted from c285966.
 * The repair keeps shared activation staging and production arithmetic while
 * using a 16-wave CTA for single-pointer rows,
 * while restoring production's per-lane block order and exact ordered scale
 * multiply. Packed GGUF bytes and full source row strides are retained. */
__global__ static void matmul_q8_0_pair_f32_sharedx_warp_rows_w32_pack4_kernel(
        float *, float *, const unsigned char *, const unsigned char *,
        const float *, uint32_t, uint64_t, uint64_t, uint64_t);
template <bool PAIR, unsigned ROWS_PER_BLOCK = 8u>
__global__ static void glm5_q8_decode_tile_kernel(
        float *out0, float *out1, const unsigned char *w0,
        const unsigned char *w1, const float *x, uint32_t n_blocks,
        uint32_t n_rows, uint64_t row_bytes) {
    extern __shared__ float sx[];
    for (unsigned i = threadIdx.x; i < n_blocks*32u; i += blockDim.x)
        sx[i] = x[i];
    __syncthreads();
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x*ROWS_PER_BLOCK + (threadIdx.x >> 5u);
    if (row >= n_rows) return;
    const unsigned pair = blockIdx.y;
    const unsigned char *w = pair ? w1 : w0;
    float *out = pair ? out1 : out0;
    const unsigned char *wr = w + (uint64_t)row*row_bytes;
    float acc = 0.0f;
    if constexpr (PAIR) {
        for (unsigned b = 0u; b < n_blocks; ++b) {
            const unsigned char *blk = wr + (uint64_t)b*34u;
            const float d = q8_0_scale_broadcast_w32(blk);
            const int8_t q = ((const int8_t *)(blk + 2u))[lane];
            acc = fmaf(d * (float)q, sx[(b << 5u) + lane], acc);
        }
        acc = warp_sum_f32(acc);
        if (lane == 0u) out[row] = acc;
        return;
    }
    unsigned b = 0u;
    for (; b + 8u <= n_blocks; b += 8u) {
        float scales[8];
        int8_t weights[8];
        float inputs[8];
#pragma unroll
        for (unsigned u = 0u; u < 8u; ++u) {
            const unsigned char *blk = wr + (uint64_t)(b + u)*34u;
            uint16_t bits = 0u;
            if (lane == 0u)
                bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8u);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
            bits = __shfl(bits, 0, 32);
#else
            bits = __shfl_sync(FULL_WARP_MASK, bits, 0, 32);
#endif
            scales[u] = __half2float(__ushort_as_half(bits));
            weights[u] = ((const int8_t *)(blk + 2u))[lane];
            inputs[u] = sx[((b + u) << 5u) + lane];
        }
#pragma unroll
        for (unsigned u = 0u; u < 8u; ++u) {
            const float scaled = q8_exact_ordered_mul(scales[u],
                                                       (float)weights[u]);
            acc += scaled * inputs[u];
        }
    }
    for (; b < n_blocks; ++b) {
        const unsigned char *blk = wr + (uint64_t)b*34u;
        const float d = q8_0_scale_broadcast_w32(blk);
        const int8_t q = ((const int8_t *)(blk + 2u))[lane];
        acc += d * (float)q * sx[(b << 5u) + lane];
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) out[row] = acc;
}

/* Speed-only attribution lane retained for the old 10.8 t/s measurement.
 * Four lanes cooperate on one Q8 block and each warp emits two adjacent rows.
 * This is intentionally opt-in (mode 3): its reduction order is not the
 * production order and it must never enter a promotion candidate. */
__global__ static void glm5_q8_decode_tile_fast_kernel(
        float *out0, float *out1, const unsigned char *w0,
        const unsigned char *w1, const float *x, uint32_t n_blocks,
        uint32_t n_rows, uint64_t row_bytes) {
    extern __shared__ float sx[];
    for (unsigned i = threadIdx.x; i < n_blocks * 32u; i += blockDim.x)
        sx[i] = x[i];
    __syncthreads();
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = (blockIdx.x * 8u + (threadIdx.x >> 5u)) * 2u;
    if (row >= n_rows) return;
    const unsigned char *w = blockIdx.y ? w1 : w0;
    float *out = blockIdx.y ? out1 : out0;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (unsigned base = 0u; base < n_blocks; base += 4u) {
        const unsigned b = base + (lane >> 3u);
        if (b >= n_blocks) continue;
        const unsigned char *a = w + (uint64_t)row * row_bytes + b * 34u;
        const unsigned char *c = a + row_bytes;
        const float d0 = q8_0_scale_scalar(a), d1 = q8_0_scale_scalar(c);
#pragma unroll
        for (unsigned pair = 0u; pair < 2u; ++pair) {
            const unsigned j = (lane & 7u) * 4u + pair * 2u;
            const float x0 = sx[b * 32u + j], x1 = sx[b * 32u + j + 1u];
            const uint16_t p0 = *reinterpret_cast<const uint16_t *>(a + 2u + j);
            const uint16_t p1 = *reinterpret_cast<const uint16_t *>(c + 2u + j);
            acc0 = fmaf(__fmul_rn(d0, (float)(int8_t)(p0 & 255u)), x0, acc0);
            acc0 = fmaf(__fmul_rn(d0, (float)(int8_t)(p0 >> 8u)), x1, acc0);
            acc1 = fmaf(__fmul_rn(d1, (float)(int8_t)(p1 & 255u)), x0, acc1);
            acc1 = fmaf(__fmul_rn(d1, (float)(int8_t)(p1 >> 8u)), x1, acc1);
        }
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0u) { out[row] = acc0; out[row + 1u] = acc1; }
}

/* Paired variant keeps both accumulators in the same warp, matching the
 * established pair kernel's expression order while using the repaired
 * eight-wave CTA and shared activation tile. */
__global__ static void glm5_q8_decode_tile_pair_kernel(
        float *out0, float *out1, const unsigned char *w0,
        const unsigned char *w1, const float *x, uint32_t n_blocks,
        uint32_t n_rows, uint64_t row_bytes) {
    extern __shared__ float sx[];
    for (unsigned i = threadIdx.x; i < n_blocks*32u; i += blockDim.x)
        sx[i] = x[i];
    __syncthreads();
    const unsigned lane = threadIdx.x & 31u;
    const unsigned wave = threadIdx.x >> 5u;
    const unsigned row = blockIdx.x*8u + wave;
    if (row >= n_rows) return;
    const unsigned char *wr0 = w0 + (uint64_t)row*row_bytes;
    const unsigned char *wr1 = w1 + (uint64_t)row*row_bytes;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (unsigned b = 0u; b < n_blocks; ++b) {
        const float xv = sx[(b << 5u) + lane];
        const unsigned char *blk0 = wr0 + (uint64_t)b*34u;
        const unsigned char *blk1 = wr1 + (uint64_t)b*34u;
        const float d0 = q8_0_scale_broadcast_w32(blk0);
        const float d1 = q8_0_scale_broadcast_w32(blk1);
        const int8_t q0 = ((const int8_t *)(blk0 + 2u))[lane];
        const int8_t q1 = ((const int8_t *)(blk1 + 2u))[lane];
        acc0 += d0 * (float)q0 * xv;
        acc1 += d1 * (float)q1 * xv;
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0u) { out0[row] = acc0; out1[row] = acc1; }
}

static int glm5_q8_decode_tile_mode(void) {
#if defined(DS4_GFX1151_WAVE32)
    const char *value = getenv("DS4_ROCM_GLM5_Q8_DECODE_TILE");
    const char *glm = getenv("DS4_GLM5_NEXT_ENABLE_ORDINARY");
    if (!glm || strcmp(glm, "1") || !value) return 0;
    const int mode = !strcmp(value, "1") ? 1 : !strcmp(value, "2") ? 2 :
                     !strcmp(value, "3") ? 3 : 0;
    if (!mode) return 0;
    static const bool supported = []() {
        int device = 0;
        cudaDeviceProp prop{};
        return cudaGetDevice(&device) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, device) == cudaSuccess &&
            strncmp(prop.gcnArchName, "gfx1151", 7) == 0 && prop.warpSize == 32;
    }();
    return supported ? mode : 0;
#else
    return 0;
#endif
}

/* Exact original-GLM shapes, including distinct full-row strides for K slices.
 * 0 means leave dispatch untouched; launch errors are returned, never retried
 * through a different arithmetic path. Mode 2 measures the existing pack4
 * schedule on the same shapes for attribution. */
static int glm5_q8_decode_tile_shape(uint32_t blocks, uint64_t rows,
                                     uint64_t stride) {
    if (blocks == 48u && rows == 16384u && stride == 48u*34u) return 1;
    if (blocks == 128u && rows == 1024u && stride == 128u*34u) return 2;
    if (blocks == 256u && rows == 4096u && stride == 512u*34u) return 3;
    if (blocks == 32u && rows == 4096u && stride == 64u*34u) return 4;
    return 0;
}

static cudaError_t cuda_launch_glm5_q8_decode_tile(
        int mode, float *out0, float *out1, const unsigned char *w0,
        const unsigned char *w1, const float *x, uint32_t blocks,
        uint32_t rows, uint64_t stride, cudaStream_t stream = 0) {
    const unsigned pairs = w1 ? 2u : 1u;
    // The mode2 attribution kernel reads four complete blocks per iteration.
    // Every production whitelist shape satisfies this; guard the leaf too.
    if (rows % 16u || blocks == 0u || blocks > 256u || blocks % 4u ||
        stride < (uint64_t)blocks*34u || stride % 2u ||
        ((uintptr_t)w0 & 1u) || (w1 && ((uintptr_t)w1 & 1u)))
        return cudaErrorInvalidValue;
    if (mode == 3) {
        glm5_q8_decode_tile_fast_kernel<<<dim3((rows + 15u) / 16u, pairs),
            256u, blocks * 32u * sizeof(float), stream>>>(
                out0, out1, w0, w1, x, blocks, rows, stride);
    } else if (mode == 1) {
        if (w1)
            glm5_q8_decode_tile_pair_kernel<<<
                dim3((rows + 7u)/8u), 256u, blocks*32u*sizeof(float), stream>>>(
                    out0, out1, w0, w1, x, blocks, rows, stride);
        else
            glm5_q8_decode_tile_kernel<false, 16u><<<
                dim3((rows + 15u)/16u, pairs), 512u,
                blocks*32u*sizeof(float), stream>>>(
                    out0, out1, w0, w1, x, blocks, rows, stride);
    } else {
        for (unsigned p = 0; p < pairs; ++p) {
            matmul_q8_0_f32_sharedx_warp_rows_w32_pack4_kernel<<<
                rows/8u, 256u, blocks*32u*sizeof(float), stream>>>(
                    p ? out1 : out0, p ? w1 : w0, x, blocks, rows, stride);
            const cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) return err;
        }
    }
    return cudaGetLastError();
}
