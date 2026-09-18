#ifndef DS4_GLM5_KDA_H
#define DS4_GLM5_KDA_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "ds4_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    DS4_GLM5_KDA_CHANNELS = 8192,
    DS4_GLM5_KDA_HISTORY = 3,
    DS4_GLM5_KDA_HEADS = 64,
    DS4_GLM5_KDA_HEAD_DIM = 128,
    DS4_GLM5_KDA_MAX_SLOTS = 1,
};

typedef struct {
    uint32_t layer;
    bool is_kda;
} ds4_glm5_layer_kind;

struct ds4_glm5_kda_slot;
struct ds4_glm5_kda_replay;

typedef struct {
    ds4_gpu_tensor *q_history;
    ds4_gpu_tensor *k_history;
    ds4_gpu_tensor *v_history;
    ds4_gpu_tensor *recurrent;
    uint64_t token_count;
    uint32_t pending_tokens;
    bool valid;
    struct ds4_glm5_kda_slot *owner_slot;
    /* Optional accepted-prefix journal, owned by this persistent layer. */
    struct ds4_glm5_kda_replay *replay;
} ds4_glm5_kda_layer_state;

typedef struct ds4_glm5_kda_slot {
    ds4_glm5_kda_layer_state *layer;
    uint32_t layer_count;
    uint32_t kda_count;
    uint32_t pending_verifications;
    uint64_t bytes;
    bool valid;
} ds4_glm5_kda_slot;

typedef struct {
    uint64_t attn_norm, q, k, v, output;
    uint64_t q_conv, k_conv, v_conv;
    uint64_t f_a, f_b, g_a, g_b, beta, o_norm, dt_bias, a_log;
    /* Production bindings carry exact GGUF types for every quantizable KDA
     * matrix. Zero retains the original BF16 synthetic-test contract. */
    uint32_t q_type, k_type, v_type, output_type;
    uint32_t f_a_type, f_b_type, g_a_type, g_b_type, beta_type;
} ds4_glm5_kda_weight_offsets;

typedef struct {
    ds4_gpu_tensor *norm, *q, *k, *v, *f_low, *g_low, *forget;
    ds4_gpu_tensor *beta, *recurrent_out;
    /* Optional, single M256/K4096 hi/lo activation panel, reused per call. */
    ds4_gpu_tensor *qkv_activation_panel;
    /* Set only after the opt-in fused norm-to-panel producer completes. */
    bool qkv_activation_panel_valid;
    uint32_t capacity_tokens;
    uint64_t bytes;
} ds4_glm5_kda_workspace;

/* Internal backend payload; callers use the owned replay interface below. */
typedef struct {
    ds4_gpu_tensor *raw_q, *raw_k, *raw_v;
    ds4_gpu_tensor *k, *v, *gate, *beta;
} ds4_glm5_kda_replay_buffers;

typedef struct {
    const ds4_glm5_kda_weight_offsets *weights;
    const void *model_map;
    uint64_t model_size;
    ds4_glm5_kda_layer_state *state;
    ds4_glm5_kda_workspace *workspace;
    const ds4_gpu_tensor *input;
    ds4_gpu_tensor *gated_output;
    ds4_gpu_tensor *output;
    uint32_t n_tokens;
    uint32_t head_start;
    uint32_t n_heads;
    float norm_eps;
    const ds4_glm5_kda_replay_buffers *replay;
} ds4_glm5_kda_device_args;

/* Test-only state comparison payload. Producing it performs device readback;
 * ordinary inference must never call this path. */
typedef struct {
    uint64_t output_fnv64;
    uint64_t q_history_fnv64;
    uint64_t k_history_fnv64;
    uint64_t v_history_fnv64;
    uint64_t recurrent_fnv64;
    uint64_t token_count;
} ds4_glm5_kda_digest;

int ds4_glm5_kda_state_bytes(uint64_t kda_count, uint32_t slot_count,
                             uint64_t *bytes);
int ds4_glm5_kda_build_schedule(ds4_glm5_layer_kind *out,
                                uint32_t capacity,
                                const bool *has_kda_q,
                                const bool *has_mla_q,
                                uint32_t layer_count,
                                uint32_t *kda_count);
int ds4_glm5_kda_slot_init(ds4_glm5_kda_slot *slot,
                           const ds4_glm5_layer_kind *schedule,
                           uint32_t layer_count,
                           uint32_t slot_count,
                           FILE *accounting);
int ds4_glm5_kda_slot_reset(ds4_glm5_kda_slot *slot);
void ds4_glm5_kda_slot_invalidate(ds4_glm5_kda_slot *slot);
void ds4_glm5_kda_slot_free(ds4_glm5_kda_slot *slot);
int ds4_glm5_kda_workspace_init(ds4_glm5_kda_workspace *workspace,
                                uint32_t capacity_tokens);
int ds4_glm5_kda_workspace_bytes(uint32_t capacity_tokens, uint64_t *bytes);
void ds4_glm5_kda_workspace_free(ds4_glm5_kda_workspace *workspace);
/* Research-only accepted-prefix transaction for one TP half of an owned,
 * full 64-head layer state. Reserve explicitly (capacity 2/4/8; rank 0/1).
 * Verify leaves live histories/recurrent bytes and token_count unchanged,
 * blocking ordinary begin/finish/commit until verify_finish. The count in
 * finish is consumed input rows, not predicted draft tokens: zero discards.
 * Invalid preconditions preserve the transaction; backend failure invalidates
 * the sequence. Reset discards saved work; slot_free releases the journal.
 * There is no weight cache, ordinary allocation or automatic speculation. */
int ds4_glm5_kda_replay_reserve(ds4_glm5_kda_layer_state *state,
                               uint32_t capacity_tokens, uint32_t rank);
uint64_t ds4_glm5_kda_replay_bytes(const ds4_glm5_kda_layer_state *state);
int ds4_glm5_kda_verify_ready(const ds4_glm5_kda_layer_state *state,
                               uint32_t n_tokens, uint32_t rank);
int ds4_glm5_kda_verify_begin(ds4_glm5_kda_layer_state *state,
                              ds4_glm5_kda_workspace *workspace,
                              const ds4_glm5_kda_weight_offsets *weights,
                              const void *model_map, uint64_t model_size,
                              const ds4_gpu_tensor *input,
                              ds4_gpu_tensor *gated_output,
                              uint32_t n_tokens, float norm_eps);
int ds4_glm5_kda_verify_finish(ds4_glm5_kda_layer_state *state,
                               uint32_t accepted_inputs);
int ds4_glm5_kda_layer_forward(ds4_glm5_kda_layer_state *state,
                               ds4_glm5_kda_workspace *workspace,
                               const ds4_glm5_kda_weight_offsets *weights,
                               const void *model_map,
                               uint64_t model_size,
                               const ds4_gpu_tensor *input,
                               ds4_gpu_tensor *output,
                               uint32_t n_tokens,
                               float norm_eps);
/* Split execution boundary used by exact TP head sharding. begin() mutates
 * only the supplied head-local recurrent state and writes packed
 * [n_tokens, n_heads*128] gated-normalized rows. finish() applies the
 * unchanged full 8192->4096 output projection and commits token_count.
 * Exactly one matching finish() is required after each successful begin();
 * a caller that cannot reach finish() must abort() the state. */
int ds4_glm5_kda_layer_begin(ds4_glm5_kda_layer_state *state,
                             ds4_glm5_kda_workspace *workspace,
                             const ds4_glm5_kda_weight_offsets *weights,
                             const void *model_map,
                             uint64_t model_size,
                             const ds4_gpu_tensor *input,
                             ds4_gpu_tensor *gated_output,
                             uint32_t n_tokens,
                             float norm_eps,
                             uint32_t head_start,
                             uint32_t n_heads);
int ds4_glm5_kda_layer_finish(ds4_glm5_kda_layer_state *state,
                              const ds4_glm5_kda_weight_offsets *weights,
                              const void *model_map,
                              uint64_t model_size,
                              const ds4_gpu_tensor *full_gated,
                              ds4_gpu_tensor *output,
                              uint32_t n_tokens);
/* Commit a successful externally composed output suffix.  This is used by
 * the TP K-slice path only after both rank partials have been exchanged and
 * added; failure retains the same fail-closed state semantics as finish(). */
int ds4_glm5_kda_layer_commit(ds4_glm5_kda_layer_state *state,
                             uint32_t n_tokens);
void ds4_glm5_kda_layer_abort(ds4_glm5_kda_layer_state *state);
int ds4_glm5_kda_compose_head_halves(ds4_gpu_tensor *full,
                                     const ds4_gpu_tensor *rank0,
                                     const ds4_gpu_tensor *rank1,
                                     uint32_t n_tokens);
int ds4_glm5_kda_layer_digest(const ds4_glm5_kda_layer_state *state,
                              const ds4_gpu_tensor *output,
                              uint64_t output_floats,
                              ds4_glm5_kda_digest *digest);
int ds4_glm5_kda_digest_equal(const ds4_glm5_kda_digest *rank0,
                              const ds4_glm5_kda_digest *rank1);

/* Internal backend adapter. Non-ROCm builds resolve the weak fail-closed
 * implementation in ds4_glm5_kda.c. */
int ds4_rocm_glm5_kda_layer_begin(
        const ds4_glm5_kda_device_args *args);
int ds4_rocm_glm5_kda_verify_begin(const ds4_glm5_kda_device_args *args);
int ds4_rocm_glm5_kda_replay_commit(ds4_glm5_kda_layer_state *state,
                                    const ds4_glm5_kda_replay_buffers *buffers,
                                    uint32_t accepted_inputs, uint32_t rank);
int ds4_rocm_glm5_kda_layer_finish(
        const ds4_glm5_kda_device_args *args,
        const ds4_gpu_tensor *full_gated);
int ds4_rocm_glm5_kda_compose_head_halves(
        ds4_gpu_tensor *full,
        const ds4_gpu_tensor *rank0,
        const ds4_gpu_tensor *rank1,
        uint32_t n_tokens);

#if defined(DS4_GLM5_KDA_TEST_HOOKS)
enum {
    DS4_GLM5_KDA_FAIL_NONE = 0,
    DS4_GLM5_KDA_FAIL_INPUT_NORM,
    DS4_GLM5_KDA_FAIL_Q_PROJECTION,
    DS4_GLM5_KDA_FAIL_K_PROJECTION,
    DS4_GLM5_KDA_FAIL_V_PROJECTION,
    DS4_GLM5_KDA_FAIL_Q_CONV,
    DS4_GLM5_KDA_FAIL_K_CONV,
    DS4_GLM5_KDA_FAIL_V_CONV,
    DS4_GLM5_KDA_FAIL_GATE_PREP,
    DS4_GLM5_KDA_FAIL_RECURRENCE,
    DS4_GLM5_KDA_FAIL_GATED_NORM,
    DS4_GLM5_KDA_FAIL_OUTPUT_PROJECTION,
};
void ds4_glm5_kda_test_fail_after(uint32_t stage);
int ds4_glm5_kda_test_should_fail(uint32_t stage);
#else
static inline int ds4_glm5_kda_test_should_fail(uint32_t stage) {
    (void)stage;
    return 0;
}
#endif

#ifdef __cplusplus
}
#endif

#endif
