#include "ds4_glm5_next_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef DS4_GLM5_TARGET_COMMIT_TEST
#include "ds4_glm5_next_exec.h"
#include "ds4_tp.h"
#endif

struct ds4_gpu_tensor {
    uint64_t bytes;
};

static int alloc_calls;
static int free_calls;
static int fill_calls;
static int fail_alloc_call;
static int copy_calls, fail_copy_call;
static FILE *accounting_stream;
static int accounting_seen_before_alloc;

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    ++alloc_calls;
    if (accounting_stream) {
        fflush(accounting_stream);
        accounting_seen_before_alloc = ftell(accounting_stream) > 0;
    }
    if (fail_alloc_call > 0 && alloc_calls == fail_alloc_call) return NULL;
    ds4_gpu_tensor *tensor = calloc(1, sizeof(*tensor));
    if (tensor) tensor->bytes = bytes;
    return tensor;
}

void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    ++free_calls;
    free(tensor);
}

int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value,
                            uint64_t count) {
    (void)value;
    if (!tensor || count > tensor->bytes / sizeof(float)) return 0;
    ++fill_calls;
    return 1;
}

uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0u;
}

int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                         const ds4_gpu_tensor *src, uint64_t src_offset,
                         uint64_t bytes) {
    ++copy_calls;
    if (fail_copy_call && copy_calls == fail_copy_call) return 0;
    return dst && src && dst_offset <= dst->bytes &&
        bytes <= dst->bytes - dst_offset && src_offset <= src->bytes &&
        bytes <= src->bytes - src_offset;
}

int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset,
                        void *data, uint64_t bytes) {
    (void)tensor;
    (void)offset;
    (void)data;
    (void)bytes;
    return 0;
}

#define CHECK(expr, message) do { \
    if (!(expr)) { fprintf(stderr, "FAIL %s\n", message); return 0; } \
} while (0)

static void reset_fakes(void) {
    alloc_calls = free_calls = fill_calls = 0;
    fail_alloc_call = 0;
    copy_calls = fail_copy_call = 0;
    accounting_stream = NULL;
    accounting_seen_before_alloc = 0;
}

static void fill_words(void *value, size_t bytes, uint64_t *next) {
    uint64_t *word = value;
    for (size_t i = 0u; i < bytes / sizeof(uint64_t); ++i)
        word[i] = (*next)++;
}

static void make_valid(ds4_glm5_next_model_offsets *model) {
    memset(model, 0, sizeof(*model));
    uint64_t next = 1u;
    model->token_embd = next++;
    model->output_norm = next++;
    model->output = next++;
    model->nextn_eh_proj = next++;
    model->layer_count = DS4_GLM5_NEXT_LAYER_COUNT;
    model->trunk_count = DS4_GLM5_NEXT_TRUNK_COUNT;
    model->nextn_count = 1u;
    model->rms_norm_eps = 1.0e-5f;
    model->hc_eps = 1.0e-6f;
    for (uint32_t il = 0u; il < model->layer_count; ++il) {
        ds4_glm5_next_layer_offsets *layer = &model->layer[il];
        layer->layer = il;
        layer->is_trunk = il < model->trunk_count;
        layer->attn_norm = next++;
        layer->ffn_norm = next++;
        const bool mla = il == DS4_GLM5_NEXT_TRUNK_COUNT ||
                         (il & 3u) == 3u;
        layer->attention = mla ? DS4_GLM5_NEXT_ATTN_MLA :
                                 DS4_GLM5_NEXT_ATTN_KDA;
        if (mla) fill_words(&layer->mla, sizeof(layer->mla), &next);
        else {
            fill_words(&layer->kda, sizeof(layer->kda), &next);
            layer->kda.q_type = layer->kda.k_type = 30u;
            layer->kda.v_type = layer->kda.output_type = 30u;
            layer->kda.f_a_type = layer->kda.f_b_type = 30u;
            layer->kda.g_a_type = layer->kda.g_b_type = 30u;
            layer->kda.beta_type = 30u;
        }
        if (il < DS4_GLM5_NEXT_LEADING_DENSE) {
            layer->ffn = DS4_GLM5_NEXT_FFN_DENSE;
            layer->ffn_weight.gate = next++;
            layer->ffn_weight.up = next++;
            layer->ffn_weight.down = next++;
        } else {
            layer->ffn = DS4_GLM5_NEXT_FFN_ROUTED;
            fill_words(&layer->ffn_weight.gate_exps,
                       sizeof(layer->ffn_weight) - 3u * sizeof(uint64_t),
                       &next);
        }
        if (layer->is_trunk)
            fill_words(&layer->hc, sizeof(layer->hc), &next);
    }
}

static int test_bytes(void) {
    ds4_glm5_next_model_offsets model;
    make_valid(&model);
    uint64_t bytes = 0u;
    CHECK(ds4_glm5_next_state_bytes(&model, 8u, &bytes),
          "8-token state size accepted");
    CHECK(bytes == UINT64_C(152870680), "8-token compact state size exact");
    CHECK(ds4_glm5_next_state_bytes(&model, 9u, &bytes) &&
          bytes == UINT64_C(152899104),
          "9-token state uses ceil pool capacity exactly");
    CHECK(ds4_glm5_next_state_bytes(&model, 262144u, &bytes),
          "256K state size accepted");
    CHECK(bytes == UINT64_C(6453309440), "256K compact state size exact");
    CHECK(!ds4_glm5_next_state_bytes(&model, 0u, &bytes),
          "zero context rejected");
    model.layer[3].attention = DS4_GLM5_NEXT_ATTN_KDA;
    CHECK(!ds4_glm5_next_state_bytes(&model, 8u, &bytes),
          "invalid schedule rejected before accounting");
    return 1;
}

static int test_lifecycle(void) {
    reset_fakes();
    ds4_glm5_next_model_offsets model;
    make_valid(&model);
    ds4_glm5_next_state state = {0};
    char *text = NULL;
    size_t text_size = 0u;
    accounting_stream = open_memstream(&text, &text_size);
    CHECK(accounting_stream, "open accounting stream");
    CHECK(ds4_glm5_next_state_init(&state, &model, 8u,
                                   accounting_stream),
          "initialize complete mixed-attention state");
    fflush(accounting_stream);
    CHECK(accounting_seen_before_alloc, "accounting precedes allocation");
    CHECK(strstr(text, "context=8 kda=34 mla=11 bytes=152870680") != NULL,
          "combined accounting exact");
    CHECK(state.valid && state.kda.valid && state.layer_count == 45u &&
          state.mla_count == 11u && state.context_capacity == 8u,
          "complete state starts valid");
    CHECK(alloc_calls == 213 && fill_calls == 147,
          "34 KDA and 11 four-buffer compact MLA states allocated once");
    CHECK(state.mla[3].valid && state.mla[3].owner == &state &&
          state.mla[3].capacity_tokens == 8u &&
          state.mla[3].capacity_pools == 2u &&
          state.mla[3].index_pool && state.mla[3].index_tail &&
          state.mla[3].pool_gate_tail && state.mla[3].index_valid_keys &&
          !state.mla[45].compact_kv,
          "trunk MLA owned and nextn MLA excluded");
    uint32_t rejected_tail = 0u, rejected_pool = 0u;
    bool rejected_publish = false;
    state.mla[3].first_valid = 1u;
    CHECK(!ds4_glm5_next_mla_append_plan(
              &state.mla[3], &rejected_tail, &rejected_pool,
              &rejected_publish),
          "compact append rejects a shifted sequence origin");
    state.mla[3].first_valid = 0u;
    state.mla[3].tail_count = 1u;
    CHECK(!ds4_glm5_next_mla_append_plan(
              &state.mla[3], &rejected_tail, &rejected_pool,
              &rejected_publish),
          "compact append rejects inconsistent counters");
    state.mla[3].tail_count = 0u;
    ds4_gpu_tensor *append_tail = state.mla[3].index_tail;
    state.mla[3].index_tail = NULL;
    CHECK(!ds4_glm5_next_mla_append_plan(
              &state.mla[3], &rejected_tail, &rejected_pool,
              &rejected_publish),
          "compact append rejects a missing tail buffer");
    state.mla[3].index_tail = append_tail;
    for (uint32_t pos = 0u; pos < 8u; ++pos) {
        uint32_t tail_slot = UINT32_MAX, pool_index = UINT32_MAX;
        bool publish_pool = false;
        CHECK(ds4_glm5_next_mla_append_plan(
                  &state.mla[3], &tail_slot, &pool_index, &publish_pool) &&
              tail_slot == pos % 4u && pool_index == pos / 4u &&
              publish_pool == (pos % 4u == 3u),
              "compact MLA append plan follows pool/tail lifecycle");
        CHECK(ds4_glm5_next_mla_append_commit(&state.mla[3]) &&
              state.mla[3].token_count == pos + 1u &&
              state.mla[3].complete_pools == (pos + 1u) / 4u &&
              state.mla[3].tail_count == (pos + 1u) % 4u,
              "compact MLA append commits atomically");
    }
    uint32_t tail_slot = 0u, pool_index = 0u;
    bool publish_pool = false;
    CHECK(!ds4_glm5_next_mla_append_plan(
              &state.mla[3], &tail_slot, &pool_index, &publish_pool),
          "compact MLA append fails closed at context capacity");
    CHECK(ds4_glm5_next_state_reset(&state),
          "state resets after compact append lifecycle test");
    CHECK(!ds4_glm5_next_state_init(&state, &model, 8u,
                                    accounting_stream),
          "live state cannot be initialized twice");
    state.mla[3].token_count = 7u;
    state.mla[3].complete_pools = 1u;
    state.mla[3].tail_count = 3u;
    ds4_glm5_next_state_invalidate(&state);
    CHECK(!state.valid && !state.kda.valid && !state.mla[3].valid,
          "mixed state invalidates atomically");
    CHECK(ds4_glm5_next_state_reset(&state) && state.valid &&
          state.kda.valid && state.mla[3].valid &&
          state.mla[3].token_count == 0u &&
          state.mla[3].complete_pools == 0u &&
          state.mla[3].tail_count == 0u &&
          state.mla[3].first_valid == 0u && fill_calls == 419,
          "mixed state resets atomically");
    state.mla[3].owner = NULL;
    CHECK(!ds4_glm5_next_state_reset(&state) && !state.valid,
          "corrupt MLA ownership fails closed");
    state.mla[3].owner = &state;
    CHECK(ds4_glm5_next_state_reset(&state),
          "restored MLA ownership resets");
    ds4_gpu_tensor *saved_compact = state.mla[3].compact_kv;
    state.mla[3].compact_kv = NULL;
    CHECK(!ds4_glm5_next_state_reset(&state) && !state.valid,
          "missing MLA buffer fails closed");
    state.mla[3].compact_kv = saved_compact;
    ds4_gpu_tensor *saved_pool = state.mla[3].index_pool;
    state.mla[3].index_pool = NULL;
    CHECK(!ds4_glm5_next_state_reset(&state) && !state.valid,
          "missing compact MLA pool buffer fails closed");
    state.mla[3].index_pool = saved_pool;
    CHECK(ds4_glm5_next_state_reset(&state),
          "restored compact MLA pool buffer resets");
    ds4_glm5_next_state_free(&state);
    CHECK(free_calls == 213 && !state.valid && !state.kda.layer &&
          !state.mla[3].compact_kv,
          "complete mixed state freed exactly once");
    ds4_glm5_next_state_free(&state);
    CHECK(free_calls == 213 && !ds4_glm5_next_state_reset(&state),
          "double free is harmless and reset-after-free fails closed");
    fclose(accounting_stream);
    accounting_stream = NULL;
    free(text);
    return 1;
}

static int test_partial_failure(void) {
    reset_fakes();
    ds4_glm5_next_model_offsets model;
    make_valid(&model);
    ds4_glm5_next_state state = {0};
    fail_alloc_call = 150;
    FILE *stream = tmpfile();
    CHECK(stream, "open failure accounting stream");
    CHECK(!ds4_glm5_next_state_init(&state, &model, 8u, stream),
          "injected MLA allocation failure propagates");
    CHECK(alloc_calls == 150 && free_calls == 149 && !state.valid &&
          !state.kda.layer && state.bytes == 0u,
          "partial KDA/MLA state is fully unwound");
    fclose(stream);

    reset_fakes();
    make_valid(&model);
    fail_alloc_call = 10;
    stream = tmpfile();
    CHECK(stream, "open KDA failure accounting stream");
    CHECK(!ds4_glm5_next_state_init(&state, &model, 8u, stream),
          "injected KDA allocation failure propagates");
    CHECK(alloc_calls == 10 && free_calls == 9 && !state.valid &&
          !state.kda.layer && state.bytes == 0u,
          "partial KDA state is fully unwound");
    fclose(stream);
    return 1;
}

static int test_mla_replay(void) {
    reset_fakes();
    ds4_glm5_next_model_offsets model;
    make_valid(&model);
    ds4_glm5_next_state state = {0};
    FILE *stream = tmpfile();
    CHECK(stream && ds4_glm5_next_state_init(&state, &model, 16u, stream),
          "initialize replay owner");
    ds4_glm5_next_mla_state *s = &state.mla[3], *view = NULL;
    CHECK(!ds4_glm5_next_mla_replay_bytes(s) &&
          !ds4_glm5_next_mla_replay_reserve(s, 3u), "explicit legal capacity");
    ds4_glm5_next_mla_state alias = *s;
    CHECK(!ds4_glm5_next_mla_replay_reserve(&alias, 4u), "reject borrowed owner");
    for (int fail = 1; fail <= 4; ++fail) {
        const int before = free_calls;
        fail_alloc_call = alloc_calls + fail;
        CHECK(!ds4_glm5_next_mla_replay_reserve(s, 4u) && !s->replay &&
              free_calls == before + 3 && state.valid,
              "all journal allocation failures unwind without state damage");
    }
    fail_alloc_call = 0;
    CHECK(ds4_glm5_next_mla_replay_reserve(s, 4u) &&
          ds4_glm5_next_mla_replay_bytes(s) == 8192u &&
          ds4_glm5_next_mla_replay_reserve(s, 4u) &&
          !ds4_glm5_next_mla_replay_reserve(s, 8u), "fixed bounded reservation");
    const int reserved_allocs = alloc_calls;
    CHECK(ds4_glm5_next_mla_verify_ready(s, 2u) &&
          ds4_glm5_next_mla_verify_ready(s, 4u) &&
          !ds4_glm5_next_mla_verify_ready(s, 8u) &&
          !ds4_glm5_next_mla_verify_ready(s, 3u) &&
          !ds4_glm5_next_mla_verify_ready(NULL, 4u), "MLA readiness bounds");
    CHECK(!ds4_glm5_next_mla_verify_begin(s, 8u, &view) && !view &&
          ds4_glm5_next_mla_verify_begin(s, 4u, &view) &&
          view != s && state.pending_mla_verifications == 1u &&
          !ds4_glm5_next_mla_verify_begin(s, 4u, &view), "single pending view");
    CHECK(!ds4_glm5_next_mla_verify_ready(s, 4u) &&
          !ds4_glm5_next_mla_verify_ready(view, 4u), "active and borrowed views are not ready");
    CHECK(!ds4_glm5_next_mla_verify_pending(s, 0u, 4u),
          "unstaged pending view cannot commit");
    CHECK(!ds4_glm5_next_mla_append_commit(s) &&
          !ds4_glm5_next_mla_append_commit(&state.mla[7]) &&
          !ds4_glm5_next_mla_append_commit(view) &&
          !ds4_glm5_next_mla_verify_finish(s, 0u),
          "ordinary, unrecorded and incomplete commits refused");
    alias = *view;
    CHECK(!ds4_glm5_next_mla_append_commit(&alias), "copied view refused");
    CHECK(!ds4_glm5_next_mla_verify_record(view, 1u, s->index_tail, 0u,
              s->pool_gate_tail, 0u, 4u) &&
          !ds4_glm5_next_mla_verify_record(view, 0u, s->index_tail, 1u,
              s->pool_gate_tail, 0u, 4u) &&
          !ds4_glm5_next_mla_verify_record(view, 0u, s->index_tail, 0u,
              s->pool_gate_tail, UINT64_MAX, 1u), "record order and bounds");
    CHECK(ds4_glm5_next_mla_verify_record(view, 0u, s->index_tail, 0u,
              s->pool_gate_tail, 0u, 4u), "batch recording precedes commits");
    for (uint32_t i = 0; i < 4u; ++i)
        CHECK(ds4_glm5_next_mla_append_commit(view), "advance private counters");
    CHECK(s->token_count == 0u && !ds4_glm5_next_mla_append_commit(view) &&
          !ds4_glm5_next_mla_verify_finish(s, 5u), "bound speculative frontier");
    CHECK(ds4_glm5_next_mla_verify_pending(s, 0u, 4u) &&
          !ds4_glm5_next_mla_verify_pending(s, 1u, 4u) &&
          !ds4_glm5_next_mla_verify_pending(s, 0u, 2u) &&
          !ds4_glm5_next_mla_verify_pending(view, 0u, 4u),
          "commit preflight binds owner, frontier and tokens");
    CHECK(ds4_glm5_next_mla_verify_finish(s, 3u) && s->token_count == 3u &&
          s->tail_count == 3u && !state.pending_mla_verifications &&
          !ds4_glm5_next_mla_append_commit(view) &&
          !ds4_glm5_next_mla_verify_finish(s, 3u), "accept prefix and retire view");
    CHECK(ds4_glm5_next_mla_verify_begin(s, 4u, &view) &&
          ds4_glm5_next_state_reset(&state) && !state.pending_mla_verifications &&
          !ds4_glm5_next_mla_verify_finish(s, 0u) &&
          !ds4_glm5_next_mla_append_commit(view) &&
          ds4_glm5_next_mla_replay_bytes(s) == 8192u,
          "reset retires view while retaining reservation");
    CHECK(alloc_calls == reserved_allocs, "begin/record/commit/reset allocate nothing");

    /* Copy submission failures at begin, both record copies and both commit
     * copies must poison the whole state. The byte-level GPU test is separate. */
    for (int stage = 0; stage < 3; ++stage) for (int fail = 1; fail <= 2; ++fail) {
        CHECK(ds4_glm5_next_state_reset(&state), "recover after backend failure");
        if (stage == 0) fail_copy_call = copy_calls + fail;
        if (stage == 0) {
            CHECK(!ds4_glm5_next_mla_verify_begin(s, 4u, &view), "begin copy failure");
        } else {
            CHECK(ds4_glm5_next_mla_verify_begin(s, 4u, &view), "begin failure fixture");
            if (stage == 1) fail_copy_call = copy_calls + fail;
            const int recorded = ds4_glm5_next_mla_verify_record(
                view, 0u, s->index_tail, 0u, s->pool_gate_tail, 0u, 4u);
            if (stage == 1) CHECK(!recorded, "record copy failure");
            else {
                CHECK(recorded, "record commit-failure fixture");
                for (int i = 0; i < 4; ++i)
                    CHECK(ds4_glm5_next_mla_append_commit(view), "private append fixture");
                fail_copy_call = copy_calls + fail;
                CHECK(!ds4_glm5_next_mla_verify_finish(s, 1u), "commit copy failure");
            }
        }
        CHECK(!state.valid && !state.kda.valid && !s->valid &&
              !state.pending_mla_verifications &&
              !ds4_glm5_next_mla_verify_finish(s, 0u) &&
              !ds4_glm5_next_mla_append_commit(view), "failure invalidates all aliases");
        fail_copy_call = 0;
    }
    CHECK(ds4_glm5_next_state_reset(&state), "final reset");
    s->token_count = 14u; s->complete_pools = 3u; s->tail_count = 2u;
    CHECK(!ds4_glm5_next_mla_verify_begin(s, 4u, &view), "capacity refusal");
    ds4_glm5_next_state_free(&state);
    CHECK(free_calls == alloc_calls - 4, "all successful allocations freed");
    fclose(stream);
    return 1;
}

#ifdef DS4_GLM5_TARGET_COMMIT_TEST
/* Exercise the production all-layer finish with deterministic host backend
 * failures. This is lifecycle/atomicity evidence, not numerical GPU evidence. */
struct ds4_tp { uint32_t rank; };
static int replay_calls, fail_replay_call, fail_sync;
int ds4_tp_rank(const ds4_tp *p) { return (int)p->rank; }
bool ds4_tp_is_rdma(const ds4_tp *p) { return p != NULL; }
bool ds4_tp_big_gate_is_rdma_capable(const ds4_tp *p) { return p != NULL; }
bool ds4_tp_big_gate_is_direct(const ds4_tp *p, const void *a, const void *b, uint64_t n) {
    return p && a && b && n;
}
uint32_t ds4_tp_runtime_features(const ds4_tp *p) { (void)p; return 0; }
uint64_t ds4_tp_prefill_config(const ds4_tp *p) { (void)p; return 0; }
int ds4_gpu_synchronize(void) { return !fail_sync; }
int ds4_rocm_glm5_kda_verify_begin(const ds4_glm5_kda_device_args *a) { return a != NULL; }
int ds4_rocm_glm5_kda_replay_commit(ds4_glm5_kda_layer_state *s,
        const ds4_glm5_kda_replay_buffers *b, uint32_t n, uint32_t rank) {
    (void)s; (void)b; (void)n; (void)rank;
    return ++replay_calls != fail_replay_call;
}

static int stage_target(ds4_glm5_next_exec_ctx *ctx, ds4_glm5_next_state *s) {
    const uint32_t n = 4;
    ds4_glm5_kda_workspace w = {.capacity_tokens=n};
    ds4_gpu_tensor input = {.bytes=4u*4096u*4u};
    for (unsigned il = 0; il < 45; ++il) {
        const int reserved = il % 4u != 3u ?
            ds4_glm5_kda_replay_reserve(&s->kda.layer[il], n, ctx->tp_rank) :
            ds4_glm5_next_mla_replay_reserve(&s->mla[il], n);
        CHECK(reserved, "reserve all journals before beginning target");
    }
    for (unsigned il = 0; il < 45; ++il) {
        if (il % 4u != 3u) {
            CHECK(ds4_glm5_kda_verify_begin(&s->kda.layer[il], &w,
                &ctx->model->layer[il].kda, ctx->model_map, ctx->model_size,
                &input, &input, n, 1.0e-5f), "stage KDA with backend stub");
            CHECK(ds4_glm5_kda_verify_pending(&s->kda.layer[il], 0, n, ctx->tp_rank) &&
                !ds4_glm5_kda_verify_pending(&s->kda.layer[il], 1, n, ctx->tp_rank) &&
                !ds4_glm5_kda_verify_pending(&s->kda.layer[il], 0, 2, ctx->tp_rank) &&
                !ds4_glm5_kda_verify_pending(&s->kda.layer[il], 0, n, 1u-ctx->tp_rank),
                "KDA commit preflight binds rank/frontier/tokens");
        } else {
            ds4_glm5_next_mla_state *live = &s->mla[il], *view = NULL;
            CHECK(ds4_glm5_next_mla_verify_begin(live, n, &view), "stage MLA");
            CHECK(ds4_glm5_next_mla_verify_record(view, 0, &input, 0, &input, 0, n), "record MLA");
            for (unsigned t = 0; t < n; ++t)
                CHECK(ds4_glm5_next_mla_append_commit(view), "advance private MLA view");
        }
    }
    s->verification = (ds4_glm5_next_verification){
        .model=ctx->model, .model_map=ctx->model_map, .model_size=ctx->model_size,
        .tp=ctx->tp, .tp_slab=ctx->tp_slab, .tp_big_out=ctx->tp_big_out,
        .tp_big_in=ctx->tp_big_in, .tp_big_out_host=ctx->tp_big_out_host,
        .tp_big_in_host=ctx->tp_big_in_host, .tp_sequence=ctx->tp_sequence,
        .sequence_end=*ctx->tp_sequence, .rank=ctx->tp_rank,
        .tokens=n, .next_layer=45, .complete=true,
    };
    return 1;
}

static int test_target_commit(void) {
    reset_fakes();
    setenv("DS4_ROCM_GLM5_BF16_SMALL_M_EXACT", "1", 1);
    ds4_glm5_next_model_offsets model;
    make_valid(&model);
    ds4_glm5_next_state state = {0};
    FILE *quiet = fopen("/dev/null", "w");
    CHECK(quiet && ds4_glm5_next_state_init(&state, &model, 8, quiet), "target state init");
    ds4_tp peer = {0};
    ds4_gpu_tensor out = {.bytes=65536}, in = {.bytes=65536};
    uint64_t sequence = 81;
    ds4_glm5_next_exec_ctx ctx = {
        .model=&model, .model_map=&model, .model_size=1, .tp=&peer,
        .tp_big_out=&out, .tp_big_in=&in, .tp_big_out_host=&out,
        .tp_big_in_host=&in, .tp_sequence=&sequence,
    };
    for (unsigned accepted = 0; accepted <= 4; ++accepted) {
        CHECK(stage_target(&ctx, &state), "complete pending target");
        const int copies = copy_calls, commits = replay_calls;
        state.kda.layer[44].pending_tokens = 2;
        CHECK(!ds4_glm5_next_target_verify_finish(&ctx, &state, accepted) &&
              state.valid && copy_calls == copies && replay_calls == commits,
              "last-layer mismatch prevents every commit");
        state.kda.layer[44].pending_tokens = 4;
        state.mla[43].token_count = 1; state.mla[43].tail_count = 1;
        CHECK(!ds4_glm5_next_target_verify_finish(&ctx, &state, accepted) &&
              copy_calls == copies && replay_calls == commits, "late MLA mismatch prevents commit");
        state.mla[43].token_count = state.mla[43].tail_count = 0;
        state.verification.complete = false;
        CHECK(!ds4_glm5_next_target_verify_finish(&ctx, &state, accepted), "incomplete logits refused");
        state.verification.complete = true;
        state.verification.next_layer = 44;
        CHECK(!ds4_glm5_next_target_verify_finish(&ctx, &state, accepted), "incomplete trunk refused");
        state.verification.next_layer = 45;
        ctx.model_size++;
        CHECK(!ds4_glm5_next_target_verify_finish(&ctx, &state, accepted), "changed source refused");
        ctx.model_size--;
        CHECK(!ds4_glm5_next_target_verify_finish(&ctx, &state, 5), "accepted bound");
        CHECK(ds4_glm5_next_target_verify_finish(&ctx, &state, accepted), "uniform accepted prefix");
        CHECK(!state.verification.tokens && !state.kda.pending_verifications &&
              !state.pending_mla_verifications && sequence == 81, "retire without rewinding transport");
        for (unsigned il = 0; il < 45; ++il)
            CHECK((il % 4 == 3 ? state.mla[il].token_count : state.kda.layer[il].token_count) ==
                  accepted, "every layer consumes the same prefix");
        CHECK(!ds4_glm5_next_target_verify_finish(&ctx, &state, accepted), "duplicate finish refused");
        CHECK(ds4_glm5_next_state_reset(&state), "reset accepted target");
    }
    for (unsigned mode = 0; mode < 3; ++mode) {
        CHECK(stage_target(&ctx, &state), "stage failing target");
        if (mode == 0) fail_replay_call = replay_calls + 4; /* after MLA layer 3 */
        if (mode == 1) fail_copy_call = copy_calls + 1; /* after KDA layers 0..2 */
        if (mode == 2) fail_sync = 1; /* every launch succeeded */
        CHECK(!ds4_glm5_next_target_verify_finish(&ctx, &state, 2) &&
              !state.valid && !state.kda.valid && !state.verification.tokens &&
              !state.kda.pending_verifications && !state.pending_mla_verifications,
              "partial commit or final async failure invalidates entire sequence");
        CHECK(!ds4_glm5_next_target_verify_finish(&ctx, &state, 0), "failed state cannot be reused");
        fail_replay_call = fail_copy_call = fail_sync = 0;
        CHECK(ds4_glm5_next_state_reset(&state), "reset invalid target");
    }
    CHECK(stage_target(&ctx, &state) && ds4_glm5_next_state_reset(&state) &&
          !state.verification.tokens && !ds4_glm5_next_target_verify_finish(&ctx, &state, 0),
          "reset cancels pending full target");
    ds4_glm5_next_state_free(&state);
    CHECK(free_calls == alloc_calls, "all reserved journals freed");
    fclose(quiet);
    fprintf(stderr, "PASS full-target commit preflight and partial-failure atomicity (host stubs)\n");
    return 1;
}
#endif

int main(void) {
    int ok = test_bytes();
    ok &= test_lifecycle();
    ok &= test_partial_failure();
    ok &= test_mla_replay();
#ifdef DS4_GLM5_TARGET_COMMIT_TEST
    ok &= test_target_commit();
#endif
    if (ok) fprintf(stderr, "PASS GLM5-next atomic resident state lifecycle\n");
    return ok ? 0 : 1;
}
