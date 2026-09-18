#include "ds4_glm5_kda.h"

#include <inttypes.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#ifndef DS4_GLM5_KDA_SCHEDULE_ONLY
static int mul_u64(uint64_t a, uint64_t b, uint64_t *result) {
    if (!result || (a != 0 && b > UINT64_MAX / a)) return 0;
    *result = a * b;
    return 1;
}

static int add_u64(uint64_t a, uint64_t b, uint64_t *result) {
    if (!result || b > UINT64_MAX - a) return 0;
    *result = a + b;
    return 1;
}

static int layer_bytes(uint64_t *history_bytes,
                       uint64_t *recurrent_bytes,
                       uint64_t *total_bytes) {
    uint64_t history_values = 0;
    uint64_t recurrent_values = 0;
    uint64_t all_histories = 0;
    if (!mul_u64(DS4_GLM5_KDA_CHANNELS, DS4_GLM5_KDA_HISTORY,
                 &history_values) ||
        !mul_u64(history_values, sizeof(float), history_bytes) ||
        !mul_u64(DS4_GLM5_KDA_HEADS, DS4_GLM5_KDA_HEAD_DIM,
                 &recurrent_values) ||
        !mul_u64(recurrent_values, DS4_GLM5_KDA_HEAD_DIM,
                 &recurrent_values) ||
        !mul_u64(recurrent_values, sizeof(float), recurrent_bytes) ||
        !mul_u64(*history_bytes, 3, &all_histories) ||
        !add_u64(all_histories, *recurrent_bytes, total_bytes)) {
        return 0;
    }
    return 1;
}

int ds4_glm5_kda_state_bytes(uint64_t kda_count, uint32_t slot_count,
                             uint64_t *bytes) {
    uint64_t history = 0;
    uint64_t recurrent = 0;
    uint64_t per_layer = 0;
    uint64_t per_slot = 0;
    if (!bytes || slot_count != DS4_GLM5_KDA_MAX_SLOTS ||
        !layer_bytes(&history, &recurrent, &per_layer) ||
        !mul_u64(per_layer, kda_count, &per_slot) ||
        !mul_u64(per_slot, slot_count, bytes)) {
        return 0;
    }
    return 1;
}
#endif

int ds4_glm5_kda_build_schedule(ds4_glm5_layer_kind *out,
                                uint32_t capacity,
                                const bool *has_kda_q,
                                const bool *has_mla_q,
                                uint32_t layer_count,
                                uint32_t *kda_count) {
    if (!out || !has_kda_q || !has_mla_q || !kda_count ||
        layer_count == 0u || capacity < layer_count) {
        return 0;
    }
    uint32_t count = 0;
    for (uint32_t layer = 0; layer < layer_count; ++layer) {
        if (has_kda_q[layer] == has_mla_q[layer]) return 0;
        if (has_kda_q[layer]) ++count;
    }
    for (uint32_t layer = 0; layer < layer_count; ++layer) {
        out[layer].layer = layer;
        out[layer].is_kda = has_kda_q[layer];
    }
    *kda_count = count;
    return 1;
}

#ifndef DS4_GLM5_KDA_SCHEDULE_ONLY
struct ds4_glm5_kda_replay {
    ds4_glm5_kda_layer_state *owner;
    ds4_glm5_kda_replay_buffers buffers;
    ds4_gpu_tensor *source[4];
    uint64_t frontier;
    uint64_t bytes;
    uint32_t capacity, rank, tokens;
    bool active;
};

static void replay_discard(ds4_glm5_kda_layer_state *s) {
    if (s && s->replay && s->replay->owner == s) s->replay->active = false;
}

static void replay_free(ds4_glm5_kda_layer_state *s) {
    if (!s || !s->replay) return;
    struct ds4_glm5_kda_replay *r = s->replay;
    if (r->owner == s) {
        ds4_gpu_tensor_free(r->buffers.raw_q);
        ds4_gpu_tensor_free(r->buffers.raw_k);
        ds4_gpu_tensor_free(r->buffers.raw_v);
        ds4_gpu_tensor_free(r->buffers.k);
        ds4_gpu_tensor_free(r->buffers.v);
        ds4_gpu_tensor_free(r->buffers.gate);
        ds4_gpu_tensor_free(r->buffers.beta);
        free(r);
    }
    s->replay = NULL;
}

static int replay_pending(const ds4_glm5_kda_layer_state *s) {
    return s && ((s->replay && s->replay->active) ||
        (s->owner_slot && s->owner_slot->pending_verifications));
}

static int replay_sources_valid(const ds4_glm5_kda_layer_state *s) {
    if (!s || !s->valid || !s->owner_slot || !s->owner_slot->valid ||
        !s->owner_slot->layer) return 0;
    bool owned = false;
    for (uint32_t i = 0; i < s->owner_slot->layer_count; ++i)
        if (s == &s->owner_slot->layer[i]) owned = true;
    if (!owned) return 0;
    return ds4_gpu_tensor_bytes(s->q_history) == 8192u * 3u * 4u &&
           ds4_gpu_tensor_bytes(s->k_history) == 8192u * 3u * 4u &&
           ds4_gpu_tensor_bytes(s->v_history) == 8192u * 3u * 4u &&
           ds4_gpu_tensor_bytes(s->recurrent) == 64u * 128u * 128u * 4u;
}

static int replay_matches(const ds4_glm5_kda_layer_state *s) {
    const struct ds4_glm5_kda_replay *r = s ? s->replay : NULL;
    return r && r->owner == s && replay_sources_valid(s) &&
           r->source[0] == s->q_history && r->source[1] == s->k_history &&
           r->source[2] == s->v_history && r->source[3] == s->recurrent;
}

int ds4_glm5_kda_replay_reserve(ds4_glm5_kda_layer_state *s,
                               uint32_t capacity, uint32_t rank) {
    if (!replay_sources_valid(s) || s->pending_tokens || rank > 1u ||
        (capacity != 2u && capacity != 4u && capacity != 8u)) return 0;
    if (s->replay) return replay_matches(s) && !s->replay->active &&
        s->replay->capacity == capacity && s->replay->rank == rank;
    struct ds4_glm5_kda_replay *r = calloc(1, sizeof(*r));
    if (!r) return 0;
    r->owner = s;
    r->capacity = capacity;
    r->rank = rank;
    r->source[0] = s->q_history; r->source[1] = s->k_history;
    r->source[2] = s->v_history; r->source[3] = s->recurrent;
    const uint64_t vector_bytes = (uint64_t)capacity * 4096u * 4u;
    const uint64_t beta_bytes = (uint64_t)capacity * 32u * 4u;
    r->bytes = 6u * vector_bytes + beta_bytes;
    r->buffers.raw_q = ds4_gpu_tensor_alloc(vector_bytes);
    r->buffers.raw_k = ds4_gpu_tensor_alloc(vector_bytes);
    r->buffers.raw_v = ds4_gpu_tensor_alloc(vector_bytes);
    r->buffers.k = ds4_gpu_tensor_alloc(vector_bytes);
    r->buffers.v = ds4_gpu_tensor_alloc(vector_bytes);
    r->buffers.gate = ds4_gpu_tensor_alloc(vector_bytes);
    r->buffers.beta = ds4_gpu_tensor_alloc(beta_bytes);
    s->replay = r;
    if (!r->buffers.raw_q || !r->buffers.raw_k || !r->buffers.raw_v ||
        !r->buffers.k || !r->buffers.v || !r->buffers.gate || !r->buffers.beta) {
        replay_free(s);
        return 0;
    }
    return 1;
}

uint64_t ds4_glm5_kda_replay_bytes(const ds4_glm5_kda_layer_state *s) {
    return s && s->replay && s->replay->owner == s ? s->replay->bytes : 0;
}

int ds4_glm5_kda_verify_begin(ds4_glm5_kda_layer_state *s,
                              ds4_glm5_kda_workspace *workspace,
                              const ds4_glm5_kda_weight_offsets *weights,
                              const void *model_map, uint64_t model_size,
                              const ds4_gpu_tensor *input,
                              ds4_gpu_tensor *gated_output,
                              uint32_t tokens, float norm_eps) {
    if (!replay_matches(s) || s->replay->active || s->pending_tokens ||
        s->owner_slot->pending_verifications >= s->owner_slot->kda_count ||
        !workspace || !weights || !model_map || !model_size ||
        (tokens != 2u && tokens != 4u && tokens != 8u) ||
        tokens > s->replay->capacity || tokens > workspace->capacity_tokens ||
        s->token_count > UINT64_MAX - tokens || norm_eps != 1.0e-5f ||
        ds4_gpu_tensor_bytes(input) < (uint64_t)tokens * 4096u * 4u ||
        ds4_gpu_tensor_bytes(gated_output) < (uint64_t)tokens * 4096u * 4u)
        return 0;
    const uint32_t types[] = {weights->q_type, weights->k_type, weights->v_type,
        weights->f_a_type, weights->f_b_type, weights->g_a_type,
        weights->g_b_type, weights->beta_type};
    for (uint32_t i = 0; i < sizeof(types)/sizeof(types[0]); ++i)
        if (types[i] != 30u) return 0;
    const char *exact = getenv("DS4_ROCM_GLM5_BF16_SMALL_M_EXACT");
    if (!exact || strcmp(exact, "1") != 0) return 0;
    const ds4_glm5_kda_device_args args = {
        .weights=weights, .model_map=model_map, .model_size=model_size,
        .state=s, .workspace=workspace, .input=input, .gated_output=gated_output,
        .n_tokens=tokens, .head_start=s->replay->rank * 32u, .n_heads=32u,
        .norm_eps=norm_eps, .replay=&s->replay->buffers,
    };
    if (!ds4_rocm_glm5_kda_verify_begin(&args)) {
        ds4_glm5_kda_layer_abort(s);
        return 0;
    }
    s->replay->frontier = s->token_count;
    s->replay->tokens = tokens;
    s->replay->active = true;
    s->pending_tokens = tokens;
    ++s->owner_slot->pending_verifications;
    return 1;
}

int ds4_glm5_kda_verify_finish(ds4_glm5_kda_layer_state *s,
                               uint32_t accepted) {
    if (!replay_matches(s) || !s->replay->active ||
        !s->owner_slot->pending_verifications ||
        s->pending_tokens != s->replay->tokens ||
        s->token_count != s->replay->frontier || accepted > s->replay->tokens)
        return 0;
    if (accepted && !ds4_rocm_glm5_kda_replay_commit(
            s, &s->replay->buffers, accepted, s->replay->rank)) {
        ds4_glm5_kda_layer_abort(s);
        return 0;
    }
    s->token_count += accepted;
    s->pending_tokens = 0;
    s->replay->active = false;
    --s->owner_slot->pending_verifications;
    return 1;
}

void ds4_glm5_kda_slot_free(ds4_glm5_kda_slot *slot) {
    if (!slot) return;
    if (slot->layer) {
        for (uint32_t i = 0; i < slot->layer_count; ++i) {
            replay_free(&slot->layer[i]);
            ds4_gpu_tensor_free(slot->layer[i].q_history);
            ds4_gpu_tensor_free(slot->layer[i].k_history);
            ds4_gpu_tensor_free(slot->layer[i].v_history);
            ds4_gpu_tensor_free(slot->layer[i].recurrent);
        }
        free(slot->layer);
    }
    memset(slot, 0, sizeof(*slot));
}

void ds4_glm5_kda_slot_invalidate(ds4_glm5_kda_slot *slot) {
    if (!slot || (slot->layer_count != 0u && !slot->layer)) return;
    slot->valid = false;
    slot->pending_verifications = 0;
    for (uint32_t i = 0; i < slot->layer_count; ++i) {
        if (slot->layer[i].recurrent) {
            replay_discard(&slot->layer[i]);
            slot->layer[i].pending_tokens = 0u;
            slot->layer[i].valid = false;
        }
    }
}

int ds4_glm5_kda_slot_reset(ds4_glm5_kda_slot *slot) {
    if (!slot || (slot->layer_count != 0 && !slot->layer)) return 0;
    slot->pending_verifications = 0;
    const uint64_t history_values =
        (uint64_t)DS4_GLM5_KDA_CHANNELS * DS4_GLM5_KDA_HISTORY;
    const uint64_t recurrent_values =
        (uint64_t)DS4_GLM5_KDA_HEADS * DS4_GLM5_KDA_HEAD_DIM *
        DS4_GLM5_KDA_HEAD_DIM;
    for (uint32_t i = 0; i < slot->layer_count; ++i) {
        ds4_glm5_kda_layer_state *state = &slot->layer[i];
        if (!state->recurrent) continue;
        replay_discard(state);
        if (!ds4_gpu_tensor_fill_f32(state->q_history, 0.0f, history_values) ||
            !ds4_gpu_tensor_fill_f32(state->k_history, 0.0f, history_values) ||
            !ds4_gpu_tensor_fill_f32(state->v_history, 0.0f, history_values) ||
            !ds4_gpu_tensor_fill_f32(state->recurrent, 0.0f,
                                     recurrent_values)) {
            ds4_glm5_kda_slot_invalidate(slot);
            return 0;
        }
        state->token_count = 0;
        state->pending_tokens = 0u;
        state->valid = true;
    }
    slot->valid = true;
    return 1;
}

int ds4_glm5_kda_slot_init(ds4_glm5_kda_slot *slot,
                           const ds4_glm5_layer_kind *schedule,
                           uint32_t layer_count,
                           uint32_t slot_count,
                           FILE *accounting) {
    uint64_t history_bytes = 0;
    uint64_t recurrent_bytes = 0;
    uint64_t per_layer = 0;
    uint64_t total_bytes = 0;
    uint64_t kda_count = 0;
    if (!slot || (layer_count != 0 && !schedule) || slot->layer ||
        !layer_bytes(&history_bytes, &recurrent_bytes, &per_layer)) {
        return 0;
    }
    for (uint32_t i = 0; i < layer_count; ++i) {
        if (schedule[i].is_kda) ++kda_count;
    }
    if (kda_count > UINT32_MAX ||
        !ds4_glm5_kda_state_bytes(kda_count, slot_count, &total_bytes)) {
        return 0;
    }
    FILE *stream = accounting ? accounting : stderr;
    fprintf(stream,
            "ds4: GLM5 KDA resident state: slots=%u layers=%" PRIu64
            " bytes=%" PRIu64 " MiB=%.2f\n",
            slot_count, kda_count, total_bytes,
            (double)total_bytes / (1024.0 * 1024.0));
    fflush(stream);

    slot->layer = calloc(layer_count ? layer_count : 1u, sizeof(*slot->layer));
    if (!slot->layer) return 0;
    slot->layer_count = layer_count;
    slot->kda_count = (uint32_t)kda_count;
    slot->bytes = total_bytes;
    for (uint32_t i = 0; i < layer_count; ++i) {
        if (!schedule[i].is_kda) continue;
        ds4_glm5_kda_layer_state *state = &slot->layer[i];
        state->owner_slot = slot;
        state->q_history = ds4_gpu_tensor_alloc(history_bytes);
        if (!state->q_history) goto fail;
        state->k_history = ds4_gpu_tensor_alloc(history_bytes);
        if (!state->k_history) goto fail;
        state->v_history = ds4_gpu_tensor_alloc(history_bytes);
        if (!state->v_history) goto fail;
        state->recurrent = ds4_gpu_tensor_alloc(recurrent_bytes);
        if (!state->recurrent) goto fail;
    }
    if (!ds4_glm5_kda_slot_reset(slot)) goto fail;
    return 1;

fail:
    ds4_glm5_kda_slot_free(slot);
    return 0;
}

void ds4_glm5_kda_workspace_free(ds4_glm5_kda_workspace *workspace) {
    if (!workspace) return;
    ds4_gpu_tensor_free(workspace->norm);
    ds4_gpu_tensor_free(workspace->q);
    ds4_gpu_tensor_free(workspace->k);
    ds4_gpu_tensor_free(workspace->v);
    ds4_gpu_tensor_free(workspace->f_low);
    if (workspace->g_low != workspace->f_low)
        ds4_gpu_tensor_free(workspace->g_low);
    ds4_gpu_tensor_free(workspace->forget);
    ds4_gpu_tensor_free(workspace->beta);
    ds4_gpu_tensor_free(workspace->recurrent_out);
    ds4_gpu_tensor_free(workspace->qkv_activation_panel);
    memset(workspace, 0, sizeof(*workspace));
}

static int kda_six_workspace_requested(void) {
    const char *decode = getenv(
        "DS4_ROCM_GLM5_BF16_KDA_SIX_DECODE_MULTIPTR");
    /* Keep the old name as a source-compatible alias for existing diagnostic
     * launchers.  New launchers use the explicit decode selector so enabling
     * prefill fusion cannot implicitly alter decode workspace admission. */
    if (!decode)
        decode = getenv("DS4_ROCM_GLM5_BF16_KDA_SIX_MULTIPTR");
    const char *prefill = getenv("DS4_ROCM_GLM5_BF16_KDA_SIX_PREFILL");
    return (decode && strcmp(decode, "1") == 0) ||
           (prefill && strcmp(prefill, "1") == 0);
}

static int kda_activation_panel_requested(uint32_t capacity_tokens) {
    const char *value = getenv("DS4_ROCM_GLM5_BF16_QKV_ACTIVATION_PANEL");
    return capacity_tokens >= 256u && value && strcmp(value, "1") == 0;
}

int ds4_glm5_kda_workspace_bytes(uint32_t capacity_tokens, uint64_t *bytes) {
    /* Physical rows only. The six-pointer experiment opts into a second
     * low-rank buffer; the ordinary path aliases g_low to f_low. */
    const uint64_t low_buffers = kda_six_workspace_requested() ? 2u : 1u;
    const uint64_t floats_per_token =
        4096u + 3u * DS4_GLM5_KDA_CHANNELS + low_buffers * 128u +
        DS4_GLM5_KDA_CHANNELS + DS4_GLM5_KDA_HEADS +
        DS4_GLM5_KDA_CHANNELS;
    uint64_t values = 0;
    return bytes && capacity_tokens != 0u &&
           mul_u64(capacity_tokens, floats_per_token, &values) &&
           mul_u64(values, sizeof(float), bytes) &&
           add_u64(*bytes, kda_activation_panel_requested(capacity_tokens)
                   ? UINT64_C(256) * 4096u * sizeof(uint32_t) : 0u, bytes);
}

static ds4_gpu_tensor *workspace_alloc_rows(uint32_t tokens,
                                             uint32_t width) {
    uint64_t values = 0;
    uint64_t bytes = 0;
    if (!mul_u64(tokens, width, &values) ||
        !mul_u64(values, sizeof(float), &bytes)) {
        return NULL;
    }
    return ds4_gpu_tensor_alloc(bytes);
}

int ds4_glm5_kda_workspace_init(ds4_glm5_kda_workspace *workspace,
                                uint32_t capacity_tokens) {
    uint64_t expected_bytes = 0;
    if (!workspace || workspace->capacity_tokens != 0u ||
        !ds4_glm5_kda_workspace_bytes(capacity_tokens, &expected_bytes)) {
        return 0;
    }
    workspace->norm = workspace_alloc_rows(capacity_tokens, 4096u);
    workspace->q = workspace_alloc_rows(capacity_tokens,
                                         DS4_GLM5_KDA_CHANNELS);
    workspace->k = workspace_alloc_rows(capacity_tokens,
                                         DS4_GLM5_KDA_CHANNELS);
    workspace->v = workspace_alloc_rows(capacity_tokens,
                                         DS4_GLM5_KDA_CHANNELS);
    workspace->f_low = workspace_alloc_rows(capacity_tokens, 128u);
    workspace->g_low = workspace->f_low;
    if (kda_six_workspace_requested())
        workspace->g_low = workspace_alloc_rows(capacity_tokens, 128u);
    workspace->forget = workspace_alloc_rows(capacity_tokens,
                                              DS4_GLM5_KDA_CHANNELS);
    workspace->beta = workspace_alloc_rows(capacity_tokens,
                                            DS4_GLM5_KDA_HEADS);
    workspace->recurrent_out = workspace_alloc_rows(
        capacity_tokens, DS4_GLM5_KDA_CHANNELS);
    const int panel_requested = kda_activation_panel_requested(capacity_tokens);
    if (panel_requested)
        workspace->qkv_activation_panel = workspace_alloc_rows(256u, 4096u);
    if (!workspace->norm || !workspace->q || !workspace->k ||
        !workspace->v || !workspace->f_low || !workspace->g_low ||
        !workspace->forget ||
        !workspace->beta || !workspace->recurrent_out ||
        (panel_requested && !workspace->qkv_activation_panel)) {
        ds4_glm5_kda_workspace_free(workspace);
        return 0;
    }
    workspace->capacity_tokens = capacity_tokens;
    workspace->bytes = expected_bytes;
    return 1;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
int ds4_rocm_glm5_kda_layer_begin(
        const ds4_glm5_kda_device_args *args) {
    (void)args;
    return 0;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
int ds4_rocm_glm5_kda_verify_begin(const ds4_glm5_kda_device_args *args) {
    (void)args;
    return 0;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
int ds4_rocm_glm5_kda_replay_commit(ds4_glm5_kda_layer_state *state,
                                    const ds4_glm5_kda_replay_buffers *buffers,
                                    uint32_t accepted, uint32_t rank) {
    (void)state; (void)buffers; (void)accepted; (void)rank;
    return 0;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
int ds4_rocm_glm5_kda_layer_finish(
        const ds4_glm5_kda_device_args *args,
        const ds4_gpu_tensor *full_gated) {
    (void)args;
    (void)full_gated;
    return 0;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
int ds4_rocm_glm5_kda_compose_head_halves(
        ds4_gpu_tensor *full,
        const ds4_gpu_tensor *rank0,
        const ds4_gpu_tensor *rank1,
        uint32_t n_tokens) {
    (void)full;
    (void)rank0;
    (void)rank1;
    (void)n_tokens;
    return 0;
}

#if defined(DS4_GLM5_KDA_TEST_HOOKS)
static uint32_t g_glm5_kda_test_fail_stage;

void ds4_glm5_kda_test_fail_after(uint32_t stage) {
    g_glm5_kda_test_fail_stage = stage;
}

int ds4_glm5_kda_test_should_fail(uint32_t stage) {
    return stage != DS4_GLM5_KDA_FAIL_NONE &&
           stage == g_glm5_kda_test_fail_stage;
}
#endif

void ds4_glm5_kda_layer_abort(ds4_glm5_kda_layer_state *state) {
    if (!state) return;
    replay_discard(state);
    state->pending_tokens = 0u;
    if (state->owner_slot) ds4_glm5_kda_slot_invalidate(state->owner_slot);
    else state->valid = false;
}

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
                             uint32_t n_heads) {
    if (replay_pending(state) ||
        (state && state->owner_slot && !state->owner_slot->valid)) return 0;
    if (!state || !state->valid || !state->q_history || !state->k_history ||
        !state->v_history || !state->recurrent || !workspace || !weights ||
        !model_map || model_size == 0u || !input || !gated_output ||
        n_tokens == 0u || n_tokens > workspace->capacity_tokens ||
        state->pending_tokens != 0u ||
        state->token_count > UINT64_MAX - n_tokens ||
        (n_heads != DS4_GLM5_KDA_HEADS &&
         n_heads != DS4_GLM5_KDA_HEADS / 2u) ||
        head_start > DS4_GLM5_KDA_HEADS - n_heads ||
        (head_start % n_heads) != 0u ||
        !(norm_eps > 0.0f)) {
        return 0;
    }
    const uint64_t input_bytes =
        (uint64_t)n_tokens * 4096u * sizeof(float);
    const uint64_t gated_bytes =
        (uint64_t)n_tokens * n_heads * DS4_GLM5_KDA_HEAD_DIM * sizeof(float);
    if (ds4_gpu_tensor_bytes(input) < input_bytes ||
        ds4_gpu_tensor_bytes(gated_output) < gated_bytes) {
        return 0;
    }
    const ds4_glm5_kda_device_args args = {
        .weights = weights,
        .model_map = model_map,
        .model_size = model_size,
        .state = state,
        .workspace = workspace,
        .input = input,
        .gated_output = gated_output,
        .output = NULL,
        .n_tokens = n_tokens,
        .head_start = head_start,
        .n_heads = n_heads,
        .norm_eps = norm_eps,
    };
    if (!ds4_rocm_glm5_kda_layer_begin(&args)) {
        ds4_glm5_kda_layer_abort(state);
        return 0;
    }
    state->pending_tokens = n_tokens;
    return 1;
}

int ds4_glm5_kda_layer_finish(ds4_glm5_kda_layer_state *state,
                              const ds4_glm5_kda_weight_offsets *weights,
                              const void *model_map,
                              uint64_t model_size,
                              const ds4_gpu_tensor *full_gated,
                              ds4_gpu_tensor *output,
                              uint32_t n_tokens) {
    if (replay_pending(state) ||
        (state && state->owner_slot && !state->owner_slot->valid)) return 0;
    const uint64_t full_bytes =
        (uint64_t)n_tokens * DS4_GLM5_KDA_CHANNELS * sizeof(float);
    const uint64_t output_bytes =
        (uint64_t)n_tokens * 4096u * sizeof(float);
    if (!state || !state->valid || !weights || !model_map ||
        model_size == 0u || !full_gated || !output || n_tokens == 0u ||
        state->pending_tokens != n_tokens ||
        state->token_count > UINT64_MAX - n_tokens ||
        ds4_gpu_tensor_bytes(full_gated) < full_bytes ||
        ds4_gpu_tensor_bytes(output) < output_bytes) {
        ds4_glm5_kda_layer_abort(state);
        return 0;
    }
    const ds4_glm5_kda_device_args args = {
        .weights = weights,
        .model_map = model_map,
        .model_size = model_size,
        .state = state,
        .workspace = NULL,
        .input = NULL,
        .gated_output = NULL,
        .output = output,
        .n_tokens = n_tokens,
        .head_start = 0u,
        .n_heads = DS4_GLM5_KDA_HEADS,
        .norm_eps = 0.0f,
    };
    if (!ds4_rocm_glm5_kda_layer_finish(&args, full_gated)) {
        ds4_glm5_kda_layer_abort(state);
        return 0;
    }
    return ds4_glm5_kda_layer_commit(state, n_tokens);
}

int ds4_glm5_kda_layer_commit(ds4_glm5_kda_layer_state *state,
                             uint32_t n_tokens) {
    if (replay_pending(state) ||
        (state && state->owner_slot && !state->owner_slot->valid)) return 0;
    if (!state || !state->valid || n_tokens == 0u ||
        state->pending_tokens != n_tokens ||
        state->token_count > UINT64_MAX - n_tokens) {
        ds4_glm5_kda_layer_abort(state);
        return 0;
    }
    state->pending_tokens = 0u;
    state->token_count += n_tokens;
    return 1;
}

int ds4_glm5_kda_compose_head_halves(ds4_gpu_tensor *full,
                                     const ds4_gpu_tensor *rank0,
                                     const ds4_gpu_tensor *rank1,
                                     uint32_t n_tokens) {
    if (!full || !rank0 || !rank1 || n_tokens == 0u) return 0;
    return ds4_rocm_glm5_kda_compose_head_halves(
        full, rank0, rank1, n_tokens);
}

int ds4_glm5_kda_layer_forward(ds4_glm5_kda_layer_state *state,
                               ds4_glm5_kda_workspace *workspace,
                               const ds4_glm5_kda_weight_offsets *weights,
                               const void *model_map,
                               uint64_t model_size,
                               const ds4_gpu_tensor *input,
                               ds4_gpu_tensor *output,
                               uint32_t n_tokens,
                               float norm_eps) {
    return state && workspace &&
           ds4_glm5_kda_layer_begin(
               state, workspace, weights, model_map, model_size, input,
               workspace->recurrent_out, n_tokens, norm_eps, 0u,
               DS4_GLM5_KDA_HEADS) &&
           ds4_glm5_kda_layer_finish(
               state, weights, model_map, model_size,
               workspace->recurrent_out, output, n_tokens);
}

static int tensor_fnv64(const ds4_gpu_tensor *tensor, uint64_t bytes,
                        uint64_t *digest) {
    enum { CHUNK_BYTES = 1u << 20 };
    if (!tensor || !digest || bytes == 0u ||
        ds4_gpu_tensor_bytes(tensor) < bytes) return 0;
    unsigned char *buffer = malloc(CHUNK_BYTES);
    if (!buffer) return 0;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (uint64_t offset = 0; offset < bytes;) {
        const uint64_t remaining = bytes - offset;
        const uint64_t count = remaining < CHUNK_BYTES ? remaining : CHUNK_BYTES;
        if (!ds4_gpu_tensor_read(tensor, offset, buffer, count)) {
            free(buffer);
            return 0;
        }
        for (uint64_t i = 0; i < count; ++i) {
            hash ^= buffer[i];
            hash *= UINT64_C(1099511628211);
        }
        offset += count;
    }
    free(buffer);
    *digest = hash;
    return 1;
}

int ds4_glm5_kda_layer_digest(const ds4_glm5_kda_layer_state *state,
                              const ds4_gpu_tensor *output,
                              uint64_t output_floats,
                              ds4_glm5_kda_digest *digest) {
    uint64_t output_bytes = 0;
    uint64_t history_bytes = 0;
    uint64_t recurrent_bytes = 0;
    uint64_t ignored = 0;
    if (!state || !state->valid || !state->q_history || !state->k_history ||
        !state->v_history || !state->recurrent || !output || !digest ||
        output_floats == 0u ||
        !mul_u64(output_floats, sizeof(float), &output_bytes) ||
        !layer_bytes(&history_bytes, &recurrent_bytes, &ignored)) return 0;
    ds4_glm5_kda_digest value = {0};
    if (!tensor_fnv64(output, output_bytes, &value.output_fnv64) ||
        !tensor_fnv64(state->q_history, history_bytes,
                      &value.q_history_fnv64) ||
        !tensor_fnv64(state->k_history, history_bytes,
                      &value.k_history_fnv64) ||
        !tensor_fnv64(state->v_history, history_bytes,
                      &value.v_history_fnv64) ||
        !tensor_fnv64(state->recurrent, recurrent_bytes,
                      &value.recurrent_fnv64)) return 0;
    value.token_count = state->token_count;
    *digest = value;
    return 1;
}

int ds4_glm5_kda_digest_equal(const ds4_glm5_kda_digest *rank0,
                              const ds4_glm5_kda_digest *rank1) {
    return rank0 && rank1 &&
           rank0->output_fnv64 == rank1->output_fnv64 &&
           rank0->q_history_fnv64 == rank1->q_history_fnv64 &&
           rank0->k_history_fnv64 == rank1->k_history_fnv64 &&
           rank0->v_history_fnv64 == rank1->v_history_fnv64 &&
           rank0->recurrent_fnv64 == rank1->recurrent_fnv64 &&
           rank0->token_count == rank1->token_count;
}
#endif
