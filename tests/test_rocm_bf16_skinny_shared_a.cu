#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../rocm/ds4_rocm_bf16_toktile.cuh"

namespace {

constexpr uint32_t kThreadsPerProjection = 256u;
constexpr uint32_t kProjections = 3u;
constexpr uint32_t kThreads = kThreadsPerProjection * kProjections;
constexpr uint32_t kKTile = 256u;

[[noreturn]] void fail(const char *what) {
    std::fprintf(stderr, "FAIL %s\n", what);
    std::exit(1);
}

void hip_ok(hipError_t status, const char *what) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "FAIL %s: %s\n", what,
                     hipGetErrorString(status));
        std::exit(1);
    }
}

uint16_t bf16_rne(float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u)
        return (uint16_t)((bits >> 16u) | 0x0040u);
    const uint32_t tie_to_even = (bits >> 16u) & 1u;
    return (uint16_t)((bits + 0x00007fffu + tie_to_even) >> 16u);
}

/*
 * Diagnostic-only shared-A skinny sidecar. Each block computes one f_a and
 * one g_a row, plus the beta row with the same index when it exists. The
 * three projection groups retain their exact 256-lane reduction chains, but
 * all of them read one F32 activation tile staged by the complete block.
 * The reduction LDS is reused between projections, so this does not add a
 * persistent buffer or a three-way reduction allocation.
 */
template <uint32_t TokenTile>
__global__ __launch_bounds__(kThreads, 1)
static void skinny_shared_a_kernel(
        float *out_f, float *out_g, float *out_beta,
        const uint16_t *weight_f, const uint16_t *weight_g,
        const uint16_t *weight_beta, const float *x,
        uint32_t in_dim, uint32_t low_rows, uint32_t beta_rows,
        uint32_t tokens) {
    static_assert(TokenTile == 8u || TokenTile == 16u || TokenTile == 32u,
                  "skinny shared-A probe uses an explicit token tile");
    const uint32_t tid = threadIdx.x;
    const uint32_t projection = tid / kThreadsPerProjection;
    const uint32_t lane = tid % kThreadsPerProjection;
    const uint32_t row = blockIdx.x;
    const uint32_t token_base = blockIdx.y * TokenTile;
    const bool active = projection < 2u ? row < low_rows : row < beta_rows;

    __shared__ float activation[TokenTile * kKTile];
    __shared__ float reduction[TokenTile * kThreadsPerProjection];
    float sums[TokenTile] = {};

    for (uint32_t k0 = 0u; k0 < in_dim; k0 += kKTile) {
        for (uint32_t j = tid; j < TokenTile * kKTile; j += kThreads) {
            const uint32_t token = j / kKTile;
            const uint32_t k = j % kKTile;
            const uint32_t global_token = token_base + token;
            activation[j] = global_token < tokens && k0 + k < in_dim
                ? x[(uint64_t)global_token * in_dim + k0 + k]
                : 0.0f;
        }
        __syncthreads();

        if (active) {
            const uint16_t *weight = projection == 0u ? weight_f :
                                     projection == 1u ? weight_g : weight_beta;
            const uint16_t *weight_row = weight + (uint64_t)row * in_dim;
            for (uint32_t k = lane; k < kKTile && k0 + k < in_dim;
                 k += kThreadsPerProjection) {
                const float w = __uint_as_float(
                    (uint32_t)weight_row[k0 + k] << 16u);
#pragma unroll
                for (uint32_t token = 0u; token < TokenTile; ++token)
                    sums[token] += w * activation[token * kKTile + k];
            }
        }
        __syncthreads();
    }

    for (uint32_t p = 0u; p < kProjections; ++p) {
        if (projection == p && active) {
#pragma unroll
            for (uint32_t token = 0u; token < TokenTile; ++token)
                reduction[token * kThreadsPerProjection + lane] =
                    sums[token];
        }
        __syncthreads();
        for (uint32_t stride = kThreadsPerProjection >> 1u;
             stride > 0u; stride >>= 1u) {
            if (projection == p && active && lane < stride) {
#pragma unroll
                for (uint32_t token = 0u; token < TokenTile; ++token)
                    reduction[token * kThreadsPerProjection + lane] +=
                        reduction[token * kThreadsPerProjection + lane + stride];
            }
            __syncthreads();
        }
        if (projection == p && active && lane == 0u) {
            float *out = p == 0u ? out_f : p == 1u ? out_g : out_beta;
            const uint32_t out_rows = p < 2u ? low_rows : beta_rows;
#pragma unroll
            for (uint32_t token = 0u; token < TokenTile; ++token) {
                const uint32_t global_token = token_base + token;
                if (global_token < tokens)
                    out[(uint64_t)global_token * out_rows + row] =
                        reduction[token * kThreadsPerProjection];
            }
        }
        __syncthreads();
    }
}

/* Single-group repair of the three-group probe above. A block still stages
 * the activation tile once, but keeps one accumulator array per projection in
 * registers and walks the independent rows serially. This is useful when the
 * launch boundary, rather than activation bandwidth, dominates the skinny
 * path: it uses the same 256-thread footprint as skinny_exact and does not
 * force three groups into one occupancy slot. */
template <uint32_t TokenTile, uint32_t KTile>
__global__ __launch_bounds__(kThreadsPerProjection, 1)
static void skinny_shared_a_parallel_reduce_kernel(
        float *out_f, float *out_g, float *out_beta,
        const uint16_t *weight_f, const uint16_t *weight_g,
        const uint16_t *weight_beta, const float *x,
        uint32_t in_dim, uint32_t low_rows, uint32_t beta_rows,
        uint32_t tokens) {
    static_assert(TokenTile == 8u || TokenTile == 16u || TokenTile == 32u,
                  "skinny shared-A probe uses an explicit token tile");
    const uint32_t lane = threadIdx.x;
    const uint32_t row = blockIdx.x;
    const uint32_t token_base = blockIdx.y * TokenTile;
    const bool low_active = row < low_rows;
    const bool beta_active = row < beta_rows;
    static_assert(KTile == 256u || KTile == 512u || KTile == 1024u,
                  "skinny shared-A probe uses an explicit K tile");
    __shared__ float activation[TokenTile * KTile];
    __shared__ float reduction[kProjections][TokenTile]
                                [kThreadsPerProjection];
    float sums[kProjections][TokenTile] = {};

    for (uint32_t k0 = 0u; k0 < in_dim; k0 += KTile) {
        for (uint32_t j = lane; j < TokenTile * KTile;
             j += kThreadsPerProjection) {
            const uint32_t token = j / KTile;
            const uint32_t k = j % KTile;
            const uint32_t global_token = token_base + token;
            activation[j] = global_token < tokens && k0 + k < in_dim
                ? x[(uint64_t)global_token * in_dim + k0 + k]
                : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (uint32_t p = 0u; p < kProjections; ++p) {
            const bool active = p < 2u ? low_active : beta_active;
            if (active) {
                const uint16_t *weight = p == 0u ? weight_f :
                                         p == 1u ? weight_g : weight_beta;
                const uint16_t *weight_row = weight + (uint64_t)row * in_dim;
                for (uint32_t k = lane; k < KTile && k0 + k < in_dim;
                     k += kThreadsPerProjection) {
                    const float w = __uint_as_float(
                        (uint32_t)weight_row[k0 + k] << 16u);
#pragma unroll
                    for (uint32_t token = 0u; token < TokenTile; ++token)
                        sums[p][token] += w * activation[token * KTile + k];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t p = 0u; p < kProjections; ++p) {
        const bool active = p < 2u ? low_active : beta_active;
#pragma unroll
        for (uint32_t token = 0u; token < TokenTile; ++token) {
            if (active)
                reduction[p][token][lane] = sums[p][token];
        }
    }
    __syncthreads();
    for (uint32_t stride = kThreadsPerProjection >> 1u;
         stride > 0u; stride >>= 1u) {
        if (lane < stride) {
#pragma unroll
            for (uint32_t p = 0u; p < kProjections; ++p) {
                const bool active = p < 2u ? low_active : beta_active;
#pragma unroll
                for (uint32_t token = 0u; token < TokenTile; ++token) {
                    if (active)
                        reduction[p][token][lane] +=
                            reduction[p][token][lane + stride];
                }
            }
        }
        __syncthreads();
    }
    if (lane == 0u) {
#pragma unroll
        for (uint32_t p = 0u; p < kProjections; ++p) {
            const bool active = p < 2u ? low_active : beta_active;
            if (!active) continue;
            float *out = p == 0u ? out_f : p == 1u ? out_g : out_beta;
            const uint32_t out_rows = p < 2u ? low_rows : beta_rows;
#pragma unroll
            for (uint32_t token = 0u; token < TokenTile; ++token) {
                const uint32_t global_token = token_base + token;
                if (global_token < tokens)
                    out[(uint64_t)global_token * out_rows + row] =
                        reduction[p][token][0u];
            }
        }
    }
}

/* Exact-reduction row-fused geometry. A wave owns one output row, while each
 * lane carries the eight partial sums that the original 256-thread skinny
 * kernel would have assigned to lanes (lane + 32*q). The first three
 * butterfly stages therefore stay in registers; the remaining stages are
 * the same wave32 shuffles as the scalar reduction. Two rows from each of
 * f_a, g_a, and beta share one staged activation tile. */
template <uint32_t TokenTile, uint32_t RowsPerProjection, uint32_t KTile>
__global__ __launch_bounds__(8u * 32u, 1)
static void skinny_shared_a_wave_exact_kernel(
        float *out_f, float *out_g, float *out_beta,
        const uint16_t *weight_f, const uint16_t *weight_g,
        const uint16_t *weight_beta, const float *x,
        uint32_t in_dim, uint32_t low_rows, uint32_t beta_rows,
        uint32_t tokens) {
    static_assert(TokenTile == 4u || TokenTile == 8u || TokenTile == 16u,
                  "skinny wave probe uses an explicit token tile");
    static_assert(RowsPerProjection == 1u || RowsPerProjection == 2u,
                  "skinny wave probe uses one or two rows per projection");
    static_assert(KTile == 256u || KTile == 512u,
                  "skinny wave probe uses an explicit K tile");
    constexpr uint32_t Waves = 8u;
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t projection = wave / RowsPerProjection;
    const uint32_t row = blockIdx.x * RowsPerProjection +
                         wave % RowsPerProjection;
    const bool active = projection < kProjections &&
        (projection < 2u ? row < low_rows : row < beta_rows);
    const uint32_t token_base = blockIdx.y * TokenTile;

    __shared__ float activation[TokenTile * KTile];
    float partial[TokenTile][8] = {};

    for (uint32_t k0 = 0u; k0 < in_dim; k0 += KTile) {
        for (uint32_t j = tid; j < TokenTile * KTile; j += Waves * 32u) {
            const uint32_t token = j / KTile;
            const uint32_t k = j % KTile;
            const uint32_t global_token = token_base + token;
            activation[j] = global_token < tokens && k0 + k < in_dim
                ? x[(uint64_t)global_token * in_dim + k0 + k]
                : 0.0f;
        }
        __syncthreads();
        if (active) {
            const uint16_t *weight = projection == 0u ? weight_f :
                                     projection == 1u ? weight_g : weight_beta;
            const uint16_t *weight_row = weight + (uint64_t)row * in_dim;
#pragma unroll
            for (uint32_t q = 0u; q < 8u; ++q) {
                for (uint32_t k = lane + q * 32u; k < KTile;
                     k += 8u * 32u) {
                    if (k0 + k < in_dim) {
                        const float w = __uint_as_float(
                            (uint32_t)weight_row[k0 + k] << 16u);
#pragma unroll
                        for (uint32_t token = 0u; token < TokenTile; ++token)
                            partial[token][q] +=
                                w * activation[token * KTile + k];
                    }
                }
            }
        }
        __syncthreads();
    }

    /* Reproduce the 256-lane tree for indices lane+32*q. */
#pragma unroll
    for (uint32_t token = 0u; token < TokenTile; ++token) {
        partial[token][0u] += partial[token][4u];
        partial[token][1u] += partial[token][5u];
        partial[token][2u] += partial[token][6u];
        partial[token][3u] += partial[token][7u];
        partial[token][0u] += partial[token][2u];
        partial[token][1u] += partial[token][3u];
        partial[token][0u] += partial[token][1u];
        float value = partial[token][0u];
#pragma unroll
        for (uint32_t offset = 16u; offset > 0u; offset >>= 1u)
            value += __shfl_down(value, offset, 32u);
        partial[token][0u] = value;
    }
    if (active && lane == 0u) {
        float *out = projection == 0u ? out_f :
                     projection == 1u ? out_g : out_beta;
        const uint32_t out_rows = projection < 2u ? low_rows : beta_rows;
#pragma unroll
        for (uint32_t token = 0u; token < TokenTile; ++token) {
            const uint32_t global_token = token_base + token;
            if (global_token < tokens)
                out[(uint64_t)global_token * out_rows + row] =
                    partial[token][0u];
        }
    }
}

/* f_a/g_a-only sidecar. Both matrices are replicated 4096->128 projections,
 * so every wave in this eight-wave block remains useful when four rows per
 * projection are assigned. beta stays on the incumbent exact launch. */
template <uint32_t TokenTile, uint32_t RowsPerProjection, uint32_t KTile>
__global__ __launch_bounds__(8u * 32u, 1)
static void skinny_shared_a_wave_exact_fg_kernel(
        float *out_f, float *out_g, const uint16_t *weight_f,
        const uint16_t *weight_g, const float *x, uint32_t in_dim,
        uint32_t low_rows, uint32_t tokens) {
    static_assert(TokenTile == 4u || TokenTile == 8u || TokenTile == 16u,
                  "skinny f/g probe uses an explicit token tile");
    static_assert(RowsPerProjection == 2u || RowsPerProjection == 4u,
                  "skinny f/g probe uses two or four rows per projection");
    static_assert(KTile == 256u || KTile == 512u,
                  "skinny f/g probe uses an explicit K tile");
    constexpr uint32_t Waves = 8u;
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t projection = wave / RowsPerProjection;
    const uint32_t row = blockIdx.x * RowsPerProjection +
                         wave % RowsPerProjection;
    const bool active = projection < 2u && row < low_rows;
    const uint32_t token_base = blockIdx.y * TokenTile;
    __shared__ float activation[TokenTile * KTile];
    float partial[TokenTile][8] = {};

    for (uint32_t k0 = 0u; k0 < in_dim; k0 += KTile) {
        for (uint32_t j = tid; j < TokenTile * KTile; j += Waves * 32u) {
            const uint32_t token = j / KTile;
            const uint32_t k = j % KTile;
            const uint32_t global_token = token_base + token;
            activation[j] = global_token < tokens && k0 + k < in_dim
                ? x[(uint64_t)global_token * in_dim + k0 + k]
                : 0.0f;
        }
        __syncthreads();
        if (active) {
            const uint16_t *weight = projection == 0u ? weight_f : weight_g;
            const uint16_t *weight_row = weight + (uint64_t)row * in_dim;
#pragma unroll
            for (uint32_t q = 0u; q < 8u; ++q) {
                for (uint32_t k = lane + q * 32u; k < KTile;
                     k += Waves * 32u) {
                    if (k0 + k < in_dim) {
                        const float w = __uint_as_float(
                            (uint32_t)weight_row[k0 + k] << 16u);
#pragma unroll
                        for (uint32_t token = 0u; token < TokenTile; ++token)
                            partial[token][q] +=
                                w * activation[token * KTile + k];
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t token = 0u; token < TokenTile; ++token) {
        partial[token][0u] += partial[token][4u];
        partial[token][1u] += partial[token][5u];
        partial[token][2u] += partial[token][6u];
        partial[token][3u] += partial[token][7u];
        partial[token][0u] += partial[token][2u];
        partial[token][1u] += partial[token][3u];
        partial[token][0u] += partial[token][1u];
        float value = partial[token][0u];
#pragma unroll
        for (uint32_t offset = 16u; offset > 0u; offset >>= 1u)
            value += __shfl_down(value, offset, 32u);
        partial[token][0u] = value;
    }
    if (active && lane == 0u) {
        float *out = projection == 0u ? out_f : out_g;
#pragma unroll
        for (uint32_t token = 0u; token < TokenTile; ++token) {
            const uint32_t global_token = token_base + token;
            if (global_token < tokens)
                out[(uint64_t)global_token * low_rows + row] =
                    partial[token][0u];
        }
    }
}

/* Control for the f/g sidecar: retain its row-fused exact reduction, but read
 * activations directly. This deliberately does not claim shared-A; it
 * measures the launch/reduction benefit independently from LDS staging. */
template <uint32_t TokenTile, uint32_t RowsPerProjection>
__global__ __launch_bounds__(8u * 32u, 1)
static void skinny_fused_fg_wave_control_kernel(
        float *out_f, float *out_g, const uint16_t *weight_f,
        const uint16_t *weight_g, const float *x, uint32_t in_dim,
        uint32_t low_rows, uint32_t tokens) {
    static_assert(TokenTile == 8u || TokenTile == 16u,
                  "skinny control uses an explicit token tile");
    static_assert(RowsPerProjection == 2u || RowsPerProjection == 4u,
                  "skinny control uses two or four rows per projection");
    constexpr uint32_t Waves = 8u;
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t projection = wave / RowsPerProjection;
    const uint32_t row = blockIdx.x * RowsPerProjection +
                         wave % RowsPerProjection;
    if (projection >= 2u || row >= low_rows) return;
    const uint32_t token_base = blockIdx.y * TokenTile;
    const uint16_t *weight = projection == 0u ? weight_f : weight_g;
    const uint16_t *weight_row = weight + (uint64_t)row * in_dim;
    float partial[TokenTile][8] = {};
#pragma unroll
    for (uint32_t q = 0u; q < 8u; ++q) {
        for (uint32_t k = lane + q * 32u; k < in_dim; k += Waves * 32u) {
            const float w = __uint_as_float(
                (uint32_t)weight_row[k] << 16u);
#pragma unroll
            for (uint32_t token = 0u; token < TokenTile; ++token) {
                const uint32_t global_token = token_base + token;
                if (global_token < tokens)
                    partial[token][q] +=
                        w * x[(uint64_t)global_token * in_dim + k];
            }
        }
    }
#pragma unroll
    for (uint32_t token = 0u; token < TokenTile; ++token) {
        partial[token][0u] += partial[token][4u];
        partial[token][1u] += partial[token][5u];
        partial[token][2u] += partial[token][6u];
        partial[token][3u] += partial[token][7u];
        partial[token][0u] += partial[token][2u];
        partial[token][1u] += partial[token][3u];
        partial[token][0u] += partial[token][1u];
        float value = partial[token][0u];
#pragma unroll
        for (uint32_t offset = 16u; offset > 0u; offset >>= 1u)
            value += __shfl_down(value, offset, 32u);
        partial[token][0u] = value;
    }
    if (lane == 0u) {
        float *out = projection == 0u ? out_f : out_g;
#pragma unroll
        for (uint32_t token = 0u; token < TokenTile; ++token) {
            const uint32_t global_token = token_base + token;
            if (global_token < tokens)
                out[(uint64_t)global_token * low_rows + row] =
                    partial[token][0u];
        }
    }
}

/* Lower-register shared-A comparison arm. It uses the same staged F32 tile
 * and row assignment as the exact f/g sidecar, but each lane owns one K
 * chain and the final reduction is a wave32 tree. This is intentionally a
 * numerical probe: its error is reported, never silently treated as exact. */
template <uint32_t TokenTile, uint32_t RowsPerProjection, uint32_t KTile>
__global__ __launch_bounds__(8u * 32u, 1)
static void skinny_shared_a_wave_fast_fg_kernel(
        float *out_f, float *out_g, const uint16_t *weight_f,
        const uint16_t *weight_g, const float *x, uint32_t in_dim,
        uint32_t low_rows, uint32_t tokens) {
    static_assert(TokenTile == 8u || TokenTile == 16u,
                  "skinny fast probe uses an explicit token tile");
    static_assert(RowsPerProjection == 4u,
                  "skinny fast probe uses four rows per projection");
    static_assert(KTile == 256u || KTile == 512u,
                  "skinny fast probe uses an explicit K tile");
    constexpr uint32_t Waves = 8u;
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t projection = wave / RowsPerProjection;
    const uint32_t row = blockIdx.x * RowsPerProjection +
                         wave % RowsPerProjection;
    /* Tail blocks still have to reach every activation barrier. */
    const bool active = projection < 2u && row < low_rows;
    const uint32_t token_base = blockIdx.y * TokenTile;
    __shared__ float activation[TokenTile * KTile];
    float sums[TokenTile] = {};
    for (uint32_t k0 = 0u; k0 < in_dim; k0 += KTile) {
        for (uint32_t j = tid; j < TokenTile * KTile; j += Waves * 32u) {
            const uint32_t token = j / KTile;
            const uint32_t k = j % KTile;
            const uint32_t global_token = token_base + token;
            activation[j] = global_token < tokens && k0 + k < in_dim
                ? x[(uint64_t)global_token * in_dim + k0 + k]
                : 0.0f;
        }
        __syncthreads();
        if (active) {
            const uint16_t *weight = projection == 0u ? weight_f : weight_g;
            const uint16_t *weight_row = weight + (uint64_t)row * in_dim;
            for (uint32_t k = lane; k < KTile && k0 + k < in_dim;
                 k += 32u) {
                const float w = __uint_as_float(
                    (uint32_t)weight_row[k0 + k] << 16u);
#pragma unroll
                for (uint32_t token = 0u; token < TokenTile; ++token)
                    sums[token] += w * activation[token * KTile + k];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (uint32_t token = 0u; token < TokenTile; ++token) {
        float value = sums[token];
#pragma unroll
        for (uint32_t offset = 16u; offset > 0u; offset >>= 1u)
            value += __shfl_down(value, offset, 32u);
        if (active && lane == 0u) {
            float *out = projection == 0u ? out_f : out_g;
            const uint32_t global_token = token_base + token;
            if (global_token < tokens)
                out[(uint64_t)global_token * low_rows + row] = value;
        }
    }
}

template <uint32_t TokenTile>
void launch_shared_a(float *out_f, float *out_g, float *out_beta,
                     const uint16_t *weight_f, const uint16_t *weight_g,
                     const uint16_t *weight_beta, const float *x,
                     uint32_t in_dim, uint32_t low_rows, uint32_t beta_rows,
                     uint32_t tokens) {
    const uint32_t rows = std::max(low_rows, beta_rows);
    skinny_shared_a_kernel<TokenTile><<<
        dim3(rows, (tokens + TokenTile - 1u) / TokenTile), kThreads>>>(
        out_f, out_g, out_beta, weight_f, weight_g, weight_beta, x,
        in_dim, low_rows, beta_rows, tokens);
    hip_ok(hipGetLastError(), "shared-A skinny launch");
}

template <uint32_t TokenTile, uint32_t KTile>
void launch_shared_a_parallel_reduce(
        float *out_f, float *out_g, float *out_beta,
        const uint16_t *weight_f, const uint16_t *weight_g,
        const uint16_t *weight_beta, const float *x, uint32_t in_dim,
        uint32_t low_rows, uint32_t beta_rows, uint32_t tokens) {
    const uint32_t rows = std::max(low_rows, beta_rows);
    skinny_shared_a_parallel_reduce_kernel<TokenTile, KTile><<<
        dim3(rows, (tokens + TokenTile - 1u) / TokenTile),
        kThreadsPerProjection>>>(
        out_f, out_g, out_beta, weight_f, weight_g, weight_beta, x,
        in_dim, low_rows, beta_rows, tokens);
    hip_ok(hipGetLastError(), "parallel-reduce shared-A skinny launch");
}

template <uint32_t TokenTile, uint32_t RowsPerProjection, uint32_t KTile>
void launch_shared_a_wave_exact(
        float *out_f, float *out_g, float *out_beta,
        const uint16_t *weight_f, const uint16_t *weight_g,
        const uint16_t *weight_beta, const float *x, uint32_t in_dim,
        uint32_t low_rows, uint32_t beta_rows, uint32_t tokens) {
    const uint32_t rows = std::max(low_rows, beta_rows);
    const uint32_t row_blocks =
        (rows + RowsPerProjection - 1u) / RowsPerProjection;
    skinny_shared_a_wave_exact_kernel<TokenTile, RowsPerProjection, KTile>
        <<<dim3(row_blocks, (tokens + TokenTile - 1u) / TokenTile),
           8u * 32u>>>(out_f, out_g, out_beta, weight_f, weight_g,
                       weight_beta, x, in_dim, low_rows, beta_rows, tokens);
    hip_ok(hipGetLastError(), "wave-exact shared-A skinny launch");
}

template <uint32_t TokenTile, uint32_t RowsPerProjection, uint32_t KTile>
void launch_shared_a_wave_exact_fg(float *out_f, float *out_g,
                                   const uint16_t *weight_f,
                                   const uint16_t *weight_g, const float *x,
                                   uint32_t in_dim, uint32_t low_rows,
                                   uint32_t tokens) {
    const uint32_t row_blocks =
        (low_rows + RowsPerProjection - 1u) / RowsPerProjection;
    skinny_shared_a_wave_exact_fg_kernel<TokenTile, RowsPerProjection, KTile>
        <<<dim3(row_blocks, (tokens + TokenTile - 1u) / TokenTile),
           8u * 32u>>>(out_f, out_g, weight_f, weight_g, x, in_dim,
                       low_rows, tokens);
    hip_ok(hipGetLastError(), "wave-exact f/g shared-A launch");
}

template <uint32_t TokenTile, uint32_t RowsPerProjection>
void launch_fused_fg_wave_control(float *out_f, float *out_g,
                                  const uint16_t *weight_f,
                                  const uint16_t *weight_g, const float *x,
                                  uint32_t in_dim, uint32_t low_rows,
                                  uint32_t tokens) {
    const uint32_t row_blocks =
        (low_rows + RowsPerProjection - 1u) / RowsPerProjection;
    skinny_fused_fg_wave_control_kernel<TokenTile, RowsPerProjection>
        <<<dim3(row_blocks, (tokens + TokenTile - 1u) / TokenTile),
           8u * 32u>>>(out_f, out_g, weight_f, weight_g, x, in_dim,
                       low_rows, tokens);
    hip_ok(hipGetLastError(), "fused f/g control launch");
}

template <uint32_t TokenTile, uint32_t RowsPerProjection, uint32_t KTile>
void launch_shared_a_wave_fast_fg(float *out_f, float *out_g,
                                  const uint16_t *weight_f,
                                  const uint16_t *weight_g, const float *x,
                                  uint32_t in_dim, uint32_t low_rows,
                                  uint32_t tokens) {
    const uint32_t row_blocks =
        (low_rows + RowsPerProjection - 1u) / RowsPerProjection;
    skinny_shared_a_wave_fast_fg_kernel<TokenTile, RowsPerProjection, KTile>
        <<<dim3(row_blocks, (tokens + TokenTile - 1u) / TokenTile),
           8u * 32u>>>(out_f, out_g, weight_f, weight_g, x, in_dim,
                       low_rows, tokens);
    hip_ok(hipGetLastError(), "fast f/g shared-A launch");
}

template <uint32_t TokenTile>
void launch_exact_projection(float *out, const uint16_t *weight,
                             const float *x, uint32_t in_dim,
                             uint32_t out_rows, uint32_t tokens) {
    uint32_t first = 0u;
    while (first < tokens) {
        const uint32_t remain = tokens - first;
        const uint32_t tile = remain >= TokenTile ? TokenTile :
                              remain >= 16u ? 16u :
                              remain >= 8u ? 8u :
                              remain >= 4u ? 4u :
                              remain >= 2u ? 2u : 1u;
        if (tile == 32u) {
            matmul_bf16_f32_skinny_exact_toktile_kernel<32u><<<
                dim3(out_rows, 1u), kThreadsPerProjection>>>(
                out + (uint64_t)first * out_rows, weight,
                x + (uint64_t)first * in_dim, in_dim, out_rows);
        } else if (tile == 16u) {
            matmul_bf16_f32_skinny_exact_toktile_kernel<16u><<<
                dim3(out_rows, 1u), kThreadsPerProjection>>>(
                out + (uint64_t)first * out_rows, weight,
                x + (uint64_t)first * in_dim, in_dim, out_rows);
        } else if (tile == 8u) {
            matmul_bf16_f32_skinny_exact_toktile_kernel<8u><<<
                dim3(out_rows, 1u), kThreadsPerProjection>>>(
                out + (uint64_t)first * out_rows, weight,
                x + (uint64_t)first * in_dim, in_dim, out_rows);
        } else if (tile == 4u) {
            matmul_bf16_f32_skinny_exact_toktile_kernel<4u><<<
                dim3(out_rows, 1u), kThreadsPerProjection>>>(
                out + (uint64_t)first * out_rows, weight,
                x + (uint64_t)first * in_dim, in_dim, out_rows);
        } else if (tile == 2u) {
            matmul_bf16_f32_skinny_exact_toktile_kernel<2u><<<
                dim3(out_rows, 1u), kThreadsPerProjection>>>(
                out + (uint64_t)first * out_rows, weight,
                x + (uint64_t)first * in_dim, in_dim, out_rows);
        } else {
            matmul_bf16_f32_skinny_exact_toktile_kernel<1u><<<
                dim3(out_rows, 1u), kThreadsPerProjection>>>(
                out + (uint64_t)first * out_rows, weight,
                x + (uint64_t)first * in_dim, in_dim, out_rows);
        }
        hip_ok(hipGetLastError(), "exact skinny launch");
        first += tile;
    }
}

void launch_exact_three(float *out_f, float *out_g, float *out_beta,
                        const uint16_t *weight_f, const uint16_t *weight_g,
                        const uint16_t *weight_beta, const float *x,
                        uint32_t in_dim, uint32_t low_rows,
                        uint32_t beta_rows, uint32_t tokens) {
    launch_exact_projection<32u>(out_f, weight_f, x, in_dim, low_rows, tokens);
    launch_exact_projection<32u>(out_g, weight_g, x, in_dim, low_rows, tokens);
    launch_exact_projection<32u>(out_beta, weight_beta, x, in_dim,
                                 beta_rows, tokens);
}

template <typename Launch>
float time_ms(Launch launch) {
    hipEvent_t begin = nullptr, end = nullptr;
    hip_ok(hipEventCreate(&begin), "create begin event");
    hip_ok(hipEventCreate(&end), "create end event");
    for (int i = 0; i < 2; ++i) launch();
    hip_ok(hipDeviceSynchronize(), "warm synchronize");
    hip_ok(hipEventRecord(begin), "record begin event");
    for (int i = 0; i < 5; ++i) launch();
    hip_ok(hipEventRecord(end), "record end event");
    hip_ok(hipEventSynchronize(end), "wait end event");
    float elapsed = 0.0f;
    hip_ok(hipEventElapsedTime(&elapsed, begin, end), "elapsed time");
    hip_ok(hipEventDestroy(end), "destroy end event");
    hip_ok(hipEventDestroy(begin), "destroy begin event");
    return elapsed / 5.0f;
}

struct Error {
    double max_abs;
    double nrmse;
};

Error compare(const std::vector<float> &reference,
              const std::vector<float> &candidate) {
    double square_error = 0.0;
    double square_reference = 0.0;
    double max_abs = 0.0;
    for (size_t i = 0u; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(candidate[i]))
            fail("non-finite skinny output");
        const double delta = (double)candidate[i] - reference[i];
        square_error += delta * delta;
        square_reference += (double)reference[i] * reference[i];
        max_abs = std::max(max_abs, std::abs(delta));
    }
    return {max_abs, std::sqrt(square_error / square_reference)};
}

template <uint32_t TokenTile>
void run_case(uint32_t in_dim, uint32_t low_rows, uint32_t beta_rows,
              uint32_t tokens) {
    const uint64_t x_count = (uint64_t)tokens * in_dim;
    const uint64_t f_count = (uint64_t)tokens * low_rows;
    const uint64_t beta_count = (uint64_t)tokens * beta_rows;
    std::vector<float> host_x(x_count);
    std::vector<uint16_t> host_f((uint64_t)low_rows * in_dim);
    std::vector<uint16_t> host_g((uint64_t)low_rows * in_dim);
    std::vector<uint16_t> host_beta((uint64_t)beta_rows * in_dim);
    for (uint64_t i = 0u; i < x_count; ++i)
        host_x[i] = 0.43f * std::cos((double)(i % 4093u) * 0.009) -
                    0.08f * std::sin((double)i * 0.0017);
    for (uint64_t i = 0u; i < host_f.size(); ++i)
        host_f[i] = bf16_rne(0.035f * std::sin((double)(i % 7919u) * 0.017));
    for (uint64_t i = 0u; i < host_g.size(); ++i)
        host_g[i] = bf16_rne(0.031f * std::cos((double)(i % 6151u) * 0.013));
    for (uint64_t i = 0u; i < host_beta.size(); ++i)
        host_beta[i] = bf16_rne(0.029f * std::sin((double)(i % 3571u) * 0.021));

    float *d_x = nullptr, *d_f = nullptr, *d_g = nullptr, *d_beta = nullptr;
    float *d_f_candidate = nullptr, *d_g_candidate = nullptr;
    float *d_beta_candidate = nullptr;
    float *d_f_serial = nullptr, *d_g_serial = nullptr;
    float *d_beta_serial = nullptr;
    float *d_f_wave8 = nullptr, *d_g_wave8 = nullptr;
    float *d_beta_wave8 = nullptr;
    float *d_f_wave4 = nullptr, *d_g_wave4 = nullptr;
    float *d_beta_wave4 = nullptr;
    float *d_f_fg = nullptr, *d_g_fg = nullptr, *d_beta_fg = nullptr;
    float *d_f_control = nullptr, *d_g_control = nullptr;
    float *d_f_fast = nullptr, *d_g_fast = nullptr, *d_beta_fast = nullptr;
    uint16_t *d_weight_f = nullptr, *d_weight_g = nullptr;
    uint16_t *d_weight_beta = nullptr;
    hip_ok(hipMalloc(&d_x, x_count * sizeof(float)), "allocate skinny x");
    hip_ok(hipMalloc(&d_weight_f, host_f.size() * sizeof(uint16_t)),
           "allocate skinny f weight");
    hip_ok(hipMalloc(&d_weight_g, host_g.size() * sizeof(uint16_t)),
           "allocate skinny g weight");
    hip_ok(hipMalloc(&d_weight_beta, host_beta.size() * sizeof(uint16_t)),
           "allocate skinny beta weight");
    hip_ok(hipMalloc(&d_f, f_count * sizeof(float)), "allocate skinny f");
    hip_ok(hipMalloc(&d_g, f_count * sizeof(float)), "allocate skinny g");
    hip_ok(hipMalloc(&d_beta, beta_count * sizeof(float)),
           "allocate skinny beta");
    hip_ok(hipMalloc(&d_f_candidate, f_count * sizeof(float)),
           "allocate candidate f");
    hip_ok(hipMalloc(&d_g_candidate, f_count * sizeof(float)),
           "allocate candidate g");
    hip_ok(hipMalloc(&d_beta_candidate, beta_count * sizeof(float)),
           "allocate candidate beta");
    hip_ok(hipMalloc(&d_f_serial, f_count * sizeof(float)),
           "allocate serial candidate f");
    hip_ok(hipMalloc(&d_g_serial, f_count * sizeof(float)),
           "allocate serial candidate g");
    hip_ok(hipMalloc(&d_beta_serial, beta_count * sizeof(float)),
           "allocate serial candidate beta");
    hip_ok(hipMalloc(&d_f_wave8, f_count * sizeof(float)),
           "allocate wave candidate f");
    hip_ok(hipMalloc(&d_g_wave8, f_count * sizeof(float)),
           "allocate wave candidate g");
    hip_ok(hipMalloc(&d_beta_wave8, beta_count * sizeof(float)),
           "allocate wave candidate beta");
    hip_ok(hipMalloc(&d_f_wave4, f_count * sizeof(float)),
           "allocate wave4 candidate f");
    hip_ok(hipMalloc(&d_g_wave4, f_count * sizeof(float)),
           "allocate wave4 candidate g");
    hip_ok(hipMalloc(&d_beta_wave4, beta_count * sizeof(float)),
           "allocate wave4 candidate beta");
    hip_ok(hipMalloc(&d_f_fg, f_count * sizeof(float)),
           "allocate f/g candidate f");
    hip_ok(hipMalloc(&d_g_fg, f_count * sizeof(float)),
           "allocate f/g candidate g");
    hip_ok(hipMalloc(&d_beta_fg, beta_count * sizeof(float)),
           "allocate f/g candidate beta");
    hip_ok(hipMalloc(&d_f_control, f_count * sizeof(float)),
           "allocate f/g control f");
    hip_ok(hipMalloc(&d_g_control, f_count * sizeof(float)),
           "allocate f/g control g");
    hip_ok(hipMalloc(&d_f_fast, f_count * sizeof(float)),
           "allocate fast f/g candidate f");
    hip_ok(hipMalloc(&d_g_fast, f_count * sizeof(float)),
           "allocate fast f/g candidate g");
    hip_ok(hipMalloc(&d_beta_fast, beta_count * sizeof(float)),
           "allocate fast beta candidate");
    hip_ok(hipMemcpy(d_x, host_x.data(), x_count * sizeof(float),
                     hipMemcpyHostToDevice), "copy skinny x");
    hip_ok(hipMemcpy(d_weight_f, host_f.data(),
                     host_f.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
           "copy skinny f weight");
    hip_ok(hipMemcpy(d_weight_g, host_g.data(),
                     host_g.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
           "copy skinny g weight");
    hip_ok(hipMemcpy(d_weight_beta, host_beta.data(),
                     host_beta.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
           "copy skinny beta weight");

    const auto baseline = [&] {
        launch_exact_three(d_f, d_g, d_beta, d_weight_f, d_weight_g,
                           d_weight_beta, d_x, in_dim, low_rows, beta_rows,
                           tokens);
    };
    const auto candidate = [&] {
        launch_shared_a<TokenTile>(
            d_f_candidate, d_g_candidate, d_beta_candidate, d_weight_f,
            d_weight_g, d_weight_beta, d_x, in_dim, low_rows, beta_rows,
            tokens);
    };
    const auto serial_candidate = [&] {
        launch_shared_a_parallel_reduce<TokenTile, 256u>(
            d_f_serial, d_g_serial, d_beta_serial, d_weight_f, d_weight_g,
            d_weight_beta, d_x, in_dim, low_rows, beta_rows, tokens);
    };
    const auto wave8_candidate = [&] {
        launch_shared_a_wave_exact<8u, 2u, 512u>(
            d_f_wave8, d_g_wave8, d_beta_wave8, d_weight_f, d_weight_g,
            d_weight_beta, d_x, in_dim, low_rows, beta_rows, tokens);
    };
    const auto wave4_candidate = [&] {
        launch_shared_a_wave_exact<4u, 2u, 256u>(
            d_f_wave4, d_g_wave4, d_beta_wave4, d_weight_f, d_weight_g,
            d_weight_beta, d_x, in_dim, low_rows, beta_rows, tokens);
    };
    const auto fg_candidate = [&] {
        launch_shared_a_wave_exact_fg<4u, 4u, 256u>(
            d_f_fg, d_g_fg, d_weight_f, d_weight_g, d_x, in_dim, low_rows,
            tokens);
        launch_exact_projection<32u>(d_beta_fg, d_weight_beta, d_x, in_dim,
                                     beta_rows, tokens);
    };
    const auto control_candidate = [&] {
        launch_fused_fg_wave_control<16u, 4u>(
            d_f_control, d_g_control, d_weight_f, d_weight_g, d_x, in_dim,
            low_rows, tokens);
    };
    const auto fast_candidate = [&] {
        launch_shared_a_wave_fast_fg<16u, 4u, 256u>(
            d_f_fast, d_g_fast, d_weight_f, d_weight_g, d_x, in_dim,
            low_rows, tokens);
        launch_exact_projection<32u>(d_beta_fast, d_weight_beta, d_x, in_dim,
                                     beta_rows, tokens);
    };
    const float baseline_ms = time_ms(baseline);
    const float candidate_ms = time_ms(candidate);
    const float serial_ms = time_ms(serial_candidate);
    const float wave8_ms = time_ms(wave8_candidate);
    const float wave4_ms = time_ms(wave4_candidate);
    const float fg_ms = time_ms(fg_candidate);
    const float control_ms = time_ms(control_candidate);
    const float fast_ms = time_ms(fast_candidate);
    baseline();
    candidate();
    serial_candidate();
    wave8_candidate();
    wave4_candidate();
    fg_candidate();
    control_candidate();
    fast_candidate();
    hip_ok(hipDeviceSynchronize(), "skinny result synchronize");

    std::vector<float> host_f_reference(f_count), host_g_reference(f_count);
    std::vector<float> host_beta_reference(beta_count);
    std::vector<float> host_f_candidate(f_count), host_g_candidate(f_count);
    std::vector<float> host_beta_candidate(beta_count);
    std::vector<float> host_f_serial(f_count), host_g_serial(f_count);
    std::vector<float> host_beta_serial(beta_count);
    std::vector<float> host_f_wave8(f_count), host_g_wave8(f_count);
    std::vector<float> host_beta_wave8(beta_count);
    std::vector<float> host_f_wave4(f_count), host_g_wave4(f_count);
    std::vector<float> host_beta_wave4(beta_count);
    std::vector<float> host_f_fg(f_count), host_g_fg(f_count);
    std::vector<float> host_beta_fg(beta_count);
    std::vector<float> host_f_control(f_count), host_g_control(f_count);
    std::vector<float> host_f_fast(f_count), host_g_fast(f_count);
    std::vector<float> host_beta_fast(beta_count);
    hip_ok(hipMemcpy(host_f_reference.data(), d_f, f_count * sizeof(float),
                     hipMemcpyDeviceToHost), "read skinny f reference");
    hip_ok(hipMemcpy(host_g_reference.data(), d_g, f_count * sizeof(float),
                     hipMemcpyDeviceToHost), "read skinny g reference");
    hip_ok(hipMemcpy(host_beta_reference.data(), d_beta,
                     beta_count * sizeof(float), hipMemcpyDeviceToHost),
           "read skinny beta reference");
    hip_ok(hipMemcpy(host_f_candidate.data(), d_f_candidate,
                     f_count * sizeof(float), hipMemcpyDeviceToHost),
           "read candidate f");
    hip_ok(hipMemcpy(host_g_candidate.data(), d_g_candidate,
                     f_count * sizeof(float), hipMemcpyDeviceToHost),
           "read candidate g");
    hip_ok(hipMemcpy(host_beta_candidate.data(), d_beta_candidate,
                     beta_count * sizeof(float), hipMemcpyDeviceToHost),
           "read candidate beta");
    hip_ok(hipMemcpy(host_f_serial.data(), d_f_serial,
                     f_count * sizeof(float), hipMemcpyDeviceToHost),
           "read serial candidate f");
    hip_ok(hipMemcpy(host_g_serial.data(), d_g_serial,
                     f_count * sizeof(float), hipMemcpyDeviceToHost),
           "read serial candidate g");
    hip_ok(hipMemcpy(host_beta_serial.data(), d_beta_serial,
                     beta_count * sizeof(float), hipMemcpyDeviceToHost),
           "read serial candidate beta");
    hip_ok(hipMemcpy(host_f_wave8.data(), d_f_wave8,
                     f_count * sizeof(float), hipMemcpyDeviceToHost),
           "read wave candidate f");
    hip_ok(hipMemcpy(host_g_wave8.data(), d_g_wave8,
                     f_count * sizeof(float), hipMemcpyDeviceToHost),
           "read wave candidate g");
    hip_ok(hipMemcpy(host_beta_wave8.data(), d_beta_wave8,
                     beta_count * sizeof(float), hipMemcpyDeviceToHost),
           "read wave candidate beta");
    hip_ok(hipMemcpy(host_f_wave4.data(), d_f_wave4,
                     f_count * sizeof(float), hipMemcpyDeviceToHost),
           "read wave4 candidate f");
    hip_ok(hipMemcpy(host_g_wave4.data(), d_g_wave4,
                     f_count * sizeof(float), hipMemcpyDeviceToHost),
           "read wave4 candidate g");
    hip_ok(hipMemcpy(host_beta_wave4.data(), d_beta_wave4,
                     beta_count * sizeof(float), hipMemcpyDeviceToHost),
           "read wave4 candidate beta");
    hip_ok(hipMemcpy(host_f_fg.data(), d_f_fg,
                     f_count * sizeof(float), hipMemcpyDeviceToHost),
           "read f/g candidate f");
    hip_ok(hipMemcpy(host_g_fg.data(), d_g_fg,
                     f_count * sizeof(float), hipMemcpyDeviceToHost),
           "read f/g candidate g");
    hip_ok(hipMemcpy(host_beta_fg.data(), d_beta_fg,
                     beta_count * sizeof(float), hipMemcpyDeviceToHost),
           "read f/g candidate beta");
    hip_ok(hipMemcpy(host_f_control.data(), d_f_control,
                     f_count * sizeof(float), hipMemcpyDeviceToHost),
           "read f/g control f");
    hip_ok(hipMemcpy(host_g_control.data(), d_g_control,
                     f_count * sizeof(float), hipMemcpyDeviceToHost),
           "read f/g control g");
    hip_ok(hipMemcpy(host_f_fast.data(), d_f_fast,
                     f_count * sizeof(float), hipMemcpyDeviceToHost),
           "read fast f/g candidate f");
    hip_ok(hipMemcpy(host_g_fast.data(), d_g_fast,
                     f_count * sizeof(float), hipMemcpyDeviceToHost),
           "read fast f/g candidate g");
    hip_ok(hipMemcpy(host_beta_fast.data(), d_beta_fast,
                     beta_count * sizeof(float), hipMemcpyDeviceToHost),
           "read fast beta candidate");
    const Error f_error = compare(host_f_reference, host_f_candidate);
    const Error g_error = compare(host_g_reference, host_g_candidate);
    const Error beta_error = compare(host_beta_reference, host_beta_candidate);
    const Error f_serial_error = compare(host_f_reference, host_f_serial);
    const Error g_serial_error = compare(host_g_reference, host_g_serial);
    const Error beta_serial_error =
        compare(host_beta_reference, host_beta_serial);
    const Error f_wave8_error = compare(host_f_reference, host_f_wave8);
    const Error g_wave8_error = compare(host_g_reference, host_g_wave8);
    const Error beta_wave8_error =
        compare(host_beta_reference, host_beta_wave8);
    const Error f_wave4_error = compare(host_f_reference, host_f_wave4);
    const Error g_wave4_error = compare(host_g_reference, host_g_wave4);
    const Error beta_wave4_error =
        compare(host_beta_reference, host_beta_wave4);
    const Error f_fg_error = compare(host_f_reference, host_f_fg);
    const Error g_fg_error = compare(host_g_reference, host_g_fg);
    const Error beta_fg_error = compare(host_beta_reference, host_beta_fg);
    const Error f_control_error = compare(host_f_reference, host_f_control);
    const Error g_control_error = compare(host_g_reference, host_g_control);
    const Error f_fast_error = compare(host_f_reference, host_f_fast);
    const Error g_fast_error = compare(host_g_reference, host_g_fast);
    const Error beta_fast_error =
        compare(host_beta_reference, host_beta_fast);
    const bool f_exact = std::memcmp(host_f_reference.data(),
                                     host_f_candidate.data(),
                                     f_count * sizeof(float)) == 0;
    const bool g_exact = std::memcmp(host_g_reference.data(),
                                     host_g_candidate.data(),
                                     f_count * sizeof(float)) == 0;
    const bool beta_exact = std::memcmp(host_beta_reference.data(),
                                        host_beta_candidate.data(),
                                        beta_count * sizeof(float)) == 0;
    const bool f_serial_exact = std::memcmp(host_f_reference.data(),
                                            host_f_serial.data(),
                                            f_count * sizeof(float)) == 0;
    const bool g_serial_exact = std::memcmp(host_g_reference.data(),
                                            host_g_serial.data(),
                                            f_count * sizeof(float)) == 0;
    const bool beta_serial_exact =
        std::memcmp(host_beta_reference.data(), host_beta_serial.data(),
                    beta_count * sizeof(float)) == 0;
    const bool f_wave8_exact = std::memcmp(host_f_reference.data(),
                                           host_f_wave8.data(),
                                           f_count * sizeof(float)) == 0;
    const bool g_wave8_exact = std::memcmp(host_g_reference.data(),
                                           host_g_wave8.data(),
                                           f_count * sizeof(float)) == 0;
    const bool beta_wave8_exact =
        std::memcmp(host_beta_reference.data(), host_beta_wave8.data(),
                    beta_count * sizeof(float)) == 0;
    const bool f_wave4_exact = std::memcmp(host_f_reference.data(),
                                           host_f_wave4.data(),
                                           f_count * sizeof(float)) == 0;
    const bool g_wave4_exact = std::memcmp(host_g_reference.data(),
                                           host_g_wave4.data(),
                                           f_count * sizeof(float)) == 0;
    const bool beta_wave4_exact =
        std::memcmp(host_beta_reference.data(), host_beta_wave4.data(),
                    beta_count * sizeof(float)) == 0;
    const bool f_fg_exact = std::memcmp(host_f_reference.data(),
                                        host_f_fg.data(),
                                        f_count * sizeof(float)) == 0;
    const bool g_fg_exact = std::memcmp(host_g_reference.data(),
                                        host_g_fg.data(),
                                        f_count * sizeof(float)) == 0;
    const bool beta_fg_exact = std::memcmp(host_beta_reference.data(),
                                           host_beta_fg.data(),
                                           beta_count * sizeof(float)) == 0;
    const bool f_control_exact = std::memcmp(host_f_reference.data(),
                                             host_f_control.data(),
                                             f_count * sizeof(float)) == 0;
    const bool g_control_exact = std::memcmp(host_g_reference.data(),
                                             host_g_control.data(),
                                             f_count * sizeof(float)) == 0;
    const bool f_fast_exact = std::memcmp(host_f_reference.data(),
                                          host_f_fast.data(),
                                          f_count * sizeof(float)) == 0;
    const bool g_fast_exact = std::memcmp(host_g_reference.data(),
                                          host_g_fast.data(),
                                          f_count * sizeof(float)) == 0;
    const bool beta_fast_exact =
        std::memcmp(host_beta_reference.data(), host_beta_fast.data(),
                    beta_count * sizeof(float)) == 0;
    std::printf(
        "skinny_shared_a tile=%u shape=%ux%ux%u beta_rows=%u "
        "baseline_3launch_ms=%.4f candidate_3group_ms=%.4f "
        "serial_ms=%.4f candidate_speedup=%.3fx serial_speedup=%.3fx "
        "f_exact=%d g_exact=%d beta_exact=%d "
        "f_serial_exact=%d g_serial_exact=%d beta_serial_exact=%d "
        "f_nrmse=%.9g g_nrmse=%.9g beta_nrmse=%.9g "
        "f_serial_nrmse=%.9g g_serial_nrmse=%.9g "
        "beta_serial_nrmse=%.9g "
        "wave8_ms=%.4f wave8_speedup=%.3fx "
        "wave8_exact=%d/%d/%d wave8_nrmse=%.9g/%.9g/%.9g "
        "wave4_ms=%.4f wave4_speedup=%.3fx "
        "wave4_exact=%d/%d/%d wave4_nrmse=%.9g/%.9g/%.9g "
        "fg_ms=%.4f fg_speedup=%.3fx fg_exact=%d/%d/%d "
        "fg_nrmse=%.9g/%.9g/%.9g "
        "control_ms=%.4f control_speedup=%.3fx control_exact=%d/%d "
        "control_nrmse=%.9g/%.9g "
        "fast_ms=%.4f fast_speedup=%.3fx "
        "fast_exact=%d/%d/%d fast_nrmse=%.9g/%.9g/%.9g "
        "fast_max_abs=%.9g/%.9g/%.9g\n",
        TokenTile, tokens, low_rows, in_dim, beta_rows, baseline_ms,
        candidate_ms, serial_ms, baseline_ms / candidate_ms,
        baseline_ms / serial_ms, f_exact ? 1 : 0, g_exact ? 1 : 0,
        beta_exact ? 1 : 0, f_serial_exact ? 1 : 0, g_serial_exact ? 1 : 0,
        beta_serial_exact ? 1 : 0, f_error.nrmse, g_error.nrmse,
        beta_error.nrmse, f_serial_error.nrmse, g_serial_error.nrmse,
        beta_serial_error.nrmse, wave8_ms, baseline_ms / wave8_ms,
        f_wave8_exact ? 1 : 0, g_wave8_exact ? 1 : 0,
        beta_wave8_exact ? 1 : 0, f_wave8_error.nrmse, g_wave8_error.nrmse,
        beta_wave8_error.nrmse, wave4_ms, baseline_ms / wave4_ms,
        f_wave4_exact ? 1 : 0, g_wave4_exact ? 1 : 0,
        beta_wave4_exact ? 1 : 0, f_wave4_error.nrmse, g_wave4_error.nrmse,
        beta_wave4_error.nrmse, fg_ms, baseline_ms / fg_ms,
        f_fg_exact ? 1 : 0, g_fg_exact ? 1 : 0, beta_fg_exact ? 1 : 0,
        f_fg_error.nrmse, g_fg_error.nrmse, beta_fg_error.nrmse,
        control_ms, baseline_ms / control_ms, f_control_exact ? 1 : 0,
        g_control_exact ? 1 : 0, f_control_error.nrmse,
        g_control_error.nrmse, fast_ms, baseline_ms / fast_ms,
        f_fast_exact ? 1 : 0, g_fast_exact ? 1 : 0,
        beta_fast_exact ? 1 : 0, f_fast_error.nrmse, g_fast_error.nrmse,
        beta_fast_error.nrmse, f_fast_error.max_abs, g_fast_error.max_abs,
        beta_fast_error.max_abs);
    if (!f_exact || !g_exact || !beta_exact || !f_serial_exact ||
        !g_serial_exact || !beta_serial_exact || !f_wave8_exact ||
        !g_wave8_exact || !beta_wave8_exact || !f_wave4_exact ||
        !g_wave4_exact || !beta_wave4_exact || !f_fg_exact || !g_fg_exact ||
        !beta_fg_exact || !f_control_exact || !g_control_exact ||
        !beta_fast_exact)
        fail("shared-A skinny bit identity");

    hip_ok(hipFree(d_g_control), "free f/g control g");
    hip_ok(hipFree(d_f_control), "free f/g control f");
    hip_ok(hipFree(d_g_fast), "free fast f/g candidate g");
    hip_ok(hipFree(d_f_fast), "free fast f/g candidate f");
    hip_ok(hipFree(d_beta_fast), "free fast beta candidate");
    hip_ok(hipFree(d_beta_fg), "free f/g candidate beta");
    hip_ok(hipFree(d_g_fg), "free f/g candidate g");
    hip_ok(hipFree(d_f_fg), "free f/g candidate f");
    hip_ok(hipFree(d_beta_wave4), "free wave4 candidate beta");
    hip_ok(hipFree(d_g_wave4), "free wave4 candidate g");
    hip_ok(hipFree(d_f_wave4), "free wave4 candidate f");
    hip_ok(hipFree(d_beta_wave8), "free wave candidate beta");
    hip_ok(hipFree(d_g_wave8), "free wave candidate g");
    hip_ok(hipFree(d_f_wave8), "free wave candidate f");
    hip_ok(hipFree(d_beta_serial), "free serial candidate beta");
    hip_ok(hipFree(d_g_serial), "free serial candidate g");
    hip_ok(hipFree(d_f_serial), "free serial candidate f");
    hip_ok(hipFree(d_beta_candidate), "free candidate beta");
    hip_ok(hipFree(d_g_candidate), "free candidate g");
    hip_ok(hipFree(d_f_candidate), "free candidate f");
    hip_ok(hipFree(d_beta), "free skinny beta");
    hip_ok(hipFree(d_g), "free skinny g");
    hip_ok(hipFree(d_f), "free skinny f");
    hip_ok(hipFree(d_weight_beta), "free beta weight");
    hip_ok(hipFree(d_weight_g), "free g weight");
    hip_ok(hipFree(d_weight_f), "free f weight");
    hip_ok(hipFree(d_x), "free skinny x");
}

}  // namespace

int main() {
    hipDeviceProp_t properties = {};
    hip_ok(hipGetDeviceProperties(&properties, 0), "query device");
    std::printf("device=%s\n", properties.name);
    run_case<16u>(4096u, 128u, 32u, 256u);
    run_case<8u>(4096u, 128u, 32u, 256u);
    run_case<16u>(4096u, 128u, 32u, 263u);
    run_case<16u>(4096u, 131u, 32u, 263u);
    run_case<16u>(4096u, 128u, 64u, 256u);
    std::puts("PASS BF16 shared-A skinny diagnostic");
    return 0;
}
