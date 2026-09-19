#ifndef DS4_GLM5_EXPERT_PAIRS_H
#define DS4_GLM5_EXPERT_PAIRS_H

#include <math.h>
#include <stdint.h>
#include <string.h>
#include "ds4_gpu.h"

/* Six tokens, eight distinct routes per token. Descriptors contain indices
 * into the original token/slot order, never a copy or conversion of weights. */
typedef struct {
    int32_t expert;
    uint32_t pair[2];
    uint32_t count;
} ds4_glm5_expert_group;
typedef struct {
    ds4_glm5_expert_group groups[48];
    uint32_t singles, doubles;
} ds4_glm5_expert_groups;

static inline int ds4_glm5_expert_routes_valid(const int32_t *ids, const float *weights) {
    if (!ids || !weights) return 0;
    for (uint32_t i = 0; i < 48; ++i) {
        if (ids[i] < 0 || ids[i] >= 288 || !isfinite(weights[i]) || weights[i] < 0.0f)
            return 0;
        for (uint32_t j = (i / 8) * 8; j < i; ++j)
            if (ids[i] == ids[j]) return 0;
    }
    return 1;
}

static inline int ds4_glm5_expert_groups_valid(const ds4_glm5_expert_groups *g,
        const int32_t *ids, const float *weights) {
    if (!g || !ds4_glm5_expert_routes_valid(ids, weights) ||
        g->singles > 48 || g->doubles > 24 || g->singles + 2 * g->doubles != 48)
        return 0;
    uint64_t seen = 0;
    for (uint32_t i = 0; i < g->singles + g->doubles; ++i) {
        const ds4_glm5_expert_group *v = &g->groups[i];
        if (v->count != (i < g->singles ? 1u : 2u) ||
            v->expert < 0 || v->expert >= 288 || (v->count == 1 && v->pair[1] != 0))
            return 0;
        for (uint32_t p = 0; p < v->count; ++p) {
            if (v->pair[p] >= 48 || (seen & (UINT64_C(1) << v->pair[p])) ||
                ids[v->pair[p]] != v->expert) return 0;
            seen |= UINT64_C(1) << v->pair[p];
        }
    }
    return seen == ((UINT64_C(1) << 48) - 1);
}

static inline int ds4_glm5_expert_groups_build(ds4_glm5_expert_groups *g,
        const int32_t *ids, const float *weights, uint32_t mode) {
    if (!g) return 0;
    memset(g, 0, sizeof(*g));
    if ((mode != 1 && mode != 2) || !ds4_glm5_expert_routes_valid(ids, weights)) return 0;
    ds4_glm5_expert_group two[24];
    uint64_t seen = 0;
    for (uint32_t i = 0; i < 48; ++i) {
        if (seen & (UINT64_C(1) << i)) continue;
        ds4_glm5_expert_group v = {ids[i], {i, 0}, 1};
        seen |= UINT64_C(1) << i;
        if (mode == 2) for (uint32_t j = i + 1; j < 48; ++j) {
            if (ids[j] == ids[i] && !(seen & (UINT64_C(1) << j))) {
                v.pair[1] = j; v.count = 2; seen |= UINT64_C(1) << j; break;
            }
        }
        if (v.count == 1) g->groups[g->singles++] = v;
        else two[g->doubles++] = v;
    }
    memcpy(g->groups + g->singles, two, g->doubles * sizeof(*two));
    return ds4_glm5_expert_groups_valid(g, ids, weights);
}

/* Borrowed, stream0-only layer state. Admission never allocates or enqueues.
 * Caller owns all buffers and must drain after a partial launch failure.
 * Readback IDs/weights must be the same arrays already admitted and agreed by
 * the two ranks. Smaller verifier tails use the incumbent API. */
typedef struct {
    ds4_gpu_tensor *out, *mid, *input_q8, *mid_q8, *descriptors;
    ds4_gpu_tensor *input, *selected, *weights;
    const void *model_map;
    uint64_t model_size, gate_offset, up_offset, down_offset;
    uint32_t rank, rows;
} ds4_glm5_expert_six_args;
typedef struct {
    ds4_glm5_expert_six_args args;
    const void *packed[3];
} ds4_glm5_expert_six_plan;

#ifdef __cplusplus
extern "C" {
#endif
int ds4_rocm_glm5_expert_six_admit(ds4_glm5_expert_six_plan *, const ds4_glm5_expert_six_args *);
int ds4_rocm_glm5_expert_six_begin(const ds4_glm5_expert_six_plan *,
    const ds4_glm5_expert_groups *, const int32_t *, const float *);
int ds4_rocm_glm5_expert_six_down_row(const ds4_glm5_expert_six_plan *, uint32_t);
#ifdef __cplusplus
}
#endif
#endif
