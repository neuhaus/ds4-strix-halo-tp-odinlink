#pragma once
#include <type_traits>

// Experimental layouts only; production kernels remain in rocm/.
/* gfx1151 BF16-WMMA projection candidate.  One 512-thread workgroup covers a
 * 256-row activation panel and two adjacent 16-column output tiles.  The
 * weight tile is loaded once for all 16 M waves.  Splitting each F32
 * activation into BF16 high and residual terms retains substantially more of
 * the incumbent F32-activation accuracy without a persistent conversion
 * buffer.  Production dispatch remains explicit and shape checked. */
template <uint32_t NTilesN, bool CoalescedB = false, bool TransposeB = false,
          uint32_t StageK = 16u>
__global__ __launch_bounds__(16u * 32u, 1)
static void ds4_bf16_panel_probe_kernel(
        float *out,
        const uint16_t *weight,
        const float *x,
        uint32_t in_dim,
        uint32_t out_dim,
        uint32_t tokens) {
    static_assert(NTilesN == 2u,
                  "validated BF16 WMMA candidate uses two N tiles");
    constexpr uint32_t BM = 16u;
    constexpr uint32_t BN = 16u;
    constexpr uint32_t BK = 16u;
    static_assert(StageK == 16u || StageK == 32u,
                  "bounded BF16 panels use K=16 or K=32");
    constexpr uint32_t MTile = 256u;
    constexpr uint32_t MTiles = MTile / BM;
    constexpr uint32_t NThreads = MTiles * 32u;
    __shared__ uint16_t sh_a_hi[MTile * StageK];
    __shared__ uint16_t sh_a_lo[MTile * StageK];
    __shared__ uint16_t sh_b[NTilesN * StageK * BN];
    const uint32_t tid = threadIdx.x;
    const uint32_t mt = tid >> 5u;
    const uint32_t nbase = blockIdx.x * NTilesN * BN;
    const uint32_t mbase = blockIdx.y * MTile;
    if (mbase >= tokens) return;

    using Bf16 = rocwmma::bfloat16_t;
    using FragA = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK,
                                     Bf16, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK,
        Bf16, typename std::conditional<CoalescedB && !TransposeB,
            rocwmma::col_major, rocwmma::row_major>::type>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK,
                                     float>;
    FragA a;
    FragB b;
    FragC acc[NTilesN];
#pragma unroll
    for (uint32_t nt = 0u; nt < NTilesN; ++nt)
        rocwmma::fill_fragment(acc[nt], 0.0f);
    for (uint32_t k0 = 0u; k0 < in_dim; k0 += StageK) {
        for (uint32_t j = tid; j < MTile * StageK; j += NThreads) {
            const uint32_t m = j / StageK;
            const uint32_t kk = j % StageK;
            const uint32_t global_m = mbase + m;
            if (global_m < tokens && k0 + kk < in_dim) {
                const float xv = x[(uint64_t)global_m * in_dim + k0 + kk];
                const uint16_t hi = ds4_bf16_rne_bits(xv);
                const float hi_f = __uint_as_float((uint32_t)hi << 16u);
                sh_a_hi[j] = hi;
                sh_a_lo[j] = ds4_bf16_rne_bits(xv - hi_f);
            } else {
                sh_a_hi[j] = 0u;
                sh_a_lo[j] = 0u;
            }
        }
        for (uint32_t j = tid; j < NTilesN * StageK * BN; j += NThreads) {
            const uint32_t nt = j / (StageK * BN);
            const uint32_t rem = j % (StageK * BN);
            // W is stored [N,K]. Column-major B[K,N] preserves that layout
            // in LDS, so adjacent lanes load adjacent K elements. The false
            // arm retains the incumbent layout and arithmetic for comparison.
            const uint32_t kk = CoalescedB ? rem % StageK : rem / BN;
            const uint32_t nn = CoalescedB ? rem / StageK : rem % BN;
            const uint32_t n = nbase + nt * BN + nn;
            const uint32_t bj = CoalescedB && TransposeB
                ? nt * StageK * BN + kk * BN + nn : j;
            sh_b[bj] = n < out_dim && k0 + kk < in_dim
                ? weight[(uint64_t)n * in_dim + k0 + kk]
                : 0u;
        }
        __syncthreads();
#pragma unroll
        for (uint32_t sk = 0u; sk < StageK; sk += BK) {
            // Retain the incumbent hi/lo order at each K=16 MMA step while
            // amortizing the block barriers over a deeper staging panel.
#pragma unroll
        for (uint32_t nt = 0u; nt < NTilesN; ++nt) {
            const uint32_t b_offset = nt * StageK * BN +
                (CoalescedB && !TransposeB ? sk : sk * BN);
            rocwmma::load_matrix_sync(
                b, reinterpret_cast<const Bf16 *>(
                    sh_b + b_offset), CoalescedB && !TransposeB ? StageK : BN);
            rocwmma::load_matrix_sync(
                a, reinterpret_cast<const Bf16 *>(
                    sh_a_hi + mt * BM * StageK + sk), StageK);
            rocwmma::mma_sync(acc[nt], a, b, acc[nt]);
            rocwmma::load_matrix_sync(
                a, reinterpret_cast<const Bf16 *>(
                    sh_a_lo + mt * BM * StageK + sk), StageK);
            rocwmma::mma_sync(acc[nt], a, b, acc[nt]);
        }
        }
        __syncthreads();
    }
#pragma unroll
    for (uint32_t nt = 0u; nt < NTilesN; ++nt) {
        const uint32_t n0 = nbase + nt * BN;
        if (n0 < out_dim && mbase + mt * BM < tokens)
            rocwmma::store_matrix_sync(
                out + (uint64_t)(mbase + mt * BM) * out_dim + n0,
                acc[nt], out_dim, rocwmma::mem_row_major);
    }
}
