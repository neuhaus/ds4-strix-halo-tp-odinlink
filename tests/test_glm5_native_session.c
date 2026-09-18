/* Production session orchestration + real two-thread socket agreement.
 * Arithmetic doubles make acceptance and failures controllable. This proves
 * scheduling/publication, not model arithmetic, quality, RDMA, or throughput. */
#include "ds4.h"
#include "ds4_glm5_next_exec.h"
#include "ds4_tp.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>

#define DS4_N_EMBD 4u
#define DS4_N_VOCAB 64u
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); abort(); } } while (0)
enum { F_NONE, F_VIEW, F_BEGIN, F_DRAFT, F_DRAFT_NAN, F_PROPOSAL,
       F_VERIFY, F_LOGITS_NAN, F_PREDICTION, F_NORMALIZE, F_REFRESH, F_FINISH, F_SYNC };
struct ds4_gpu_tensor { unsigned char *data; uint64_t bytes; int owns; };
struct ds4_glm5_next_workspace { uint32_t rows; };
struct ds4_engine { struct { int active; ds4_tp *ctx; } tp; };
struct ds4_session {
    ds4_engine *engine;
    ds4_glm5_next_exec_ctx glm5_next_exec;
    ds4_glm5_next_state glm5_next_state, glm5_native_state;
    ds4_glm5_next_workspace *glm5_next_ws, *glm5_native_ws, *glm5_native_warm_ws;
    ds4_glm5_next_workspace *glm5_native_verify_ws[3];
    ds4_gpu_tensor *glm5_native_previous, *glm5_native_chain, *glm5_native_hidden;
    ds4_gpu_tensor *glm5_native_normalized, *glm5_native_shifted;
    ds4_gpu_tensor *glm5_native_hc[2], *glm5_native_logits;
    ds4_gpu_tensor *glm5_next_cur, *glm5_next_logits;
    float *glm5_native_host_logits, *sample_probs, *logits;
    ds4_tokens checkpoint;
    int ctx_size;
    uint32_t glm5_native_rows;
    uint64_t glm5_native_cycle, tp_session_id;
    bool glm5_next_ready, checkpoint_valid, glm5_native_previous_valid;
};
typedef struct {
    ds4_session s;
    ds4_engine e;
    ds4_tp_native_cycle command;
    unsigned wanted, fault, step, views, phases, start, warm_count;
    uint32_t verified[8], warm_ids[4096];
    float warm_previous[4096];
    int accepted[8], result, refreshed, finished;
    char error[160];
} fixture;
static __thread fixture *active;
static unsigned cases;
static const uint64_t row_bytes = DS4_N_EMBD * sizeof(float);

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    ds4_gpu_tensor *t = calloc(1, sizeof(*t)); CHECK(t);
    t->data = calloc(1, bytes); CHECK(t->data); t->bytes = bytes; t->owns = 1; return t;
}
ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (active && active->fault == F_VIEW && ++active->views == 1u) return NULL;
    if (!base || offset > base->bytes || bytes > base->bytes - offset) return NULL;
    ds4_gpu_tensor *t = calloc(1, sizeof(*t)); CHECK(t);
    t->data = base->data + offset; t->bytes = bytes; return t;
}
void ds4_gpu_tensor_free(ds4_gpu_tensor *t) { if (t) { if (t->owns) free(t->data); free(t); } }
int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t d, const ds4_gpu_tensor *src,
                        uint64_t s, uint64_t bytes) {
    if (!dst || !src || d > dst->bytes || bytes > dst->bytes - d ||
        s > src->bytes || bytes > src->bytes - s) return 0;
    memcpy(dst->data + d, src->data + s, bytes); return 1;
}
int ds4_gpu_tensor_read(const ds4_gpu_tensor *t, uint64_t offset, void *out, uint64_t bytes) {
    if (!t || offset > t->bytes || bytes > t->bytes - offset) return 0;
    memcpy(out, t->data + offset, bytes); return 1;
}
int ds4_gpu_synchronize(void) { return !(active && active->fault == F_SYNC && active->finished); }
uint32_t ds4_glm5_next_workspace_capacity(const ds4_glm5_next_workspace *w) { return w ? w->rows : 0; }
ds4_glm5_next_workspace *ds4_glm5_next_draft_workspace_create_rows(uint32_t rows, uint32_t ctx) {
    CHECK(ctx >= rows); ds4_glm5_next_workspace *w = malloc(sizeof(*w)); CHECK(w); w->rows = rows; return w;
}
void ds4_glm5_next_workspace_destroy(ds4_glm5_next_workspace *w) { free(w); }
void ds4_glm5_next_workspace_begin_decode(ds4_glm5_next_workspace *w) { CHECK(w); }
void ds4_glm5_next_state_invalidate(ds4_glm5_next_state *s) { s->valid = false; }
void ds4_session_invalidate(ds4_session *s) {
    s->checkpoint.len = 0; s->checkpoint_valid = s->glm5_native_previous_valid = false;
    ds4_glm5_next_state_invalidate(&s->glm5_next_state);
    ds4_glm5_next_state_invalidate(&s->glm5_native_state);
}
static void token_vec_push(ds4_tokens *v, int token) { CHECK(v->len < v->cap); v->v[v->len++] = token; }

int ds4_glm5_next_draft_target_hidden(const ds4_glm5_next_exec_ctx *x,
        ds4_glm5_next_workspace *w, const ds4_gpu_tensor *hc, ds4_gpu_tensor *out) {
    return ds4_glm5_next_draft_target_hidden_rows(x, w, hc, out, w ? w->rows : 0u);
}
int ds4_glm5_next_draft_target_hidden_rows(const ds4_glm5_next_exec_ctx *x,
        ds4_glm5_next_workspace *w, const ds4_gpu_tensor *hc, ds4_gpu_tensor *out, uint32_t rows) {
    (void)x;
    if (active->fault == F_NORMALIZE || !w || !hc || !out || !rows || rows > w->rows ||
        hc->bytes != rows * row_bytes * 4u || out->bytes != rows * row_bytes) return 0;
    for (unsigned i = 0; i < rows; ++i)
        memcpy(out->data + i * row_bytes, hc->data + i * row_bytes * 4u, row_bytes);
    return 1;
}
int ds4_glm5_next_draft_warm_rows(const ds4_glm5_next_exec_ctx *x,
        ds4_glm5_next_state *s, ds4_glm5_next_workspace *w,
        const ds4_gpu_tensor *hidden, const uint32_t *ids, uint32_t rows) {
    (void)x; CHECK(w->rows == rows && hidden->bytes == rows * row_bytes && s->valid);
    for (unsigned t = 0; t < rows; ++t) {
        CHECK(active->warm_count < 4096u);
        active->warm_ids[active->warm_count] = ids[t];
        active->warm_previous[active->warm_count++] = ((float *)hidden->data)[t * DS4_N_EMBD];
        ++s->mla[45].token_count;
    }
    return 1;
}
int ds4_glm5_next_mla_verify_begin(ds4_glm5_next_mla_state *s, uint32_t rows,
                                   ds4_glm5_next_mla_state **view) {
    CHECK(rows == active->command.rows - 1u);
    if (active->fault == F_BEGIN) return 0;
    *view = s; return 1;
}
static void fill_logits(float *out, unsigned token) {
    for (unsigned i = 0; i < DS4_N_VOCAB; ++i) out[i] = i == token ? 2.0f : -1.0f;
}
int ds4_glm5_next_draft_step(const ds4_glm5_next_exec_ctx *x, ds4_glm5_next_mla_state *s,
        ds4_glm5_next_workspace *w, const ds4_gpu_tensor *previous, uint32_t token,
        ds4_gpu_tensor *hidden, ds4_gpu_tensor *logits) {
    (void)x; CHECK(w->rows == 1u);
    const unsigned t = active->step++;
    const float want = t ? -(float)t : 1000.0f + active->start - 1u;
    CHECK(((float *)previous->data)[0] == want);
    if (active->fault == F_DRAFT && t == 0u) return 0;
    for (unsigned j = 0; j < DS4_N_EMBD; ++j) ((float *)hidden->data)[j] = -(float)(t + 1u);
    fill_logits((float *)logits->data, (token + 1u + (active->fault == F_PROPOSAL)) % DS4_N_VOCAB);
    if (active->fault == F_DRAFT_NAN) ((float *)logits->data)[0] = NAN;
    ++s->token_count; return 1;
}
int ds4_glm5_next_target_verify(const ds4_glm5_next_exec_ctx *x,
        ds4_glm5_next_state *s, ds4_glm5_next_workspace *w, ds4_glm5_next_workspace *scalar,
        const uint32_t *tokens, uint32_t rows, ds4_gpu_tensor *h0,
        ds4_gpu_tensor *h1, ds4_gpu_tensor *logits) {
    (void)x; (void)s; (void)h0; CHECK(w->rows == rows && scalar->rows == 1u);
    if (active->fault == F_VERIFY) return 0;
    memcpy(active->verified, tokens, rows * sizeof(*tokens));
    for (unsigned t = 0; t < rows; ++t) {
        unsigned next = t + 1u < active->wanted ? tokens[t + 1u] : 1u;
        if (active->fault == F_PREDICTION) next = 2u;
        fill_logits((float *)logits->data + t * DS4_N_VOCAB, next);
        for (unsigned j = 0; j < DS4_N_EMBD * 4u; ++j)
            ((float *)h1->data)[t * DS4_N_EMBD * 4u + j] = 1000.0f + active->start + t;
    }
    if (active->fault == F_LOGITS_NAN) ((float *)logits->data)[0] = NAN;
    return 1;
}
int ds4_glm5_next_draft_refresh(const ds4_glm5_next_exec_ctx *x,
        ds4_glm5_next_state *s, ds4_glm5_next_workspace *w, const ds4_gpu_tensor *hidden,
        const uint32_t *tokens, uint32_t prefix, uint32_t rows, uint32_t accepted,
        ds4_gpu_tensor *previous) {
    (void)x; CHECK(w->rows == 1u && rows == active->command.rows && accepted > 0u);
    CHECK(s->mla[45].token_count == prefix - 1u + rows - 1u);
    CHECK(((float *)previous->data)[0] == 1000.0f + prefix - 1u);
    CHECK(!active->finished && active->s.checkpoint.len == (int)prefix);
    if (active->fault == F_REFRESH) return 0;
    for (unsigned t = 0; t < accepted; ++t) {
        CHECK(tokens[t] == active->verified[t]);
        CHECK(((float *)hidden->data)[t * DS4_N_EMBD] == 1000.0f + prefix + t);
    }
    s->mla[45].token_count = prefix + accepted - 1u;
    memcpy(previous->data, hidden->data + (accepted - 1u) * row_bytes, row_bytes);
    active->refreshed = 1; return 1;
}
int ds4_glm5_next_target_verify_finish(const ds4_glm5_next_exec_ctx *x,
        ds4_glm5_next_state *s, uint32_t accepted) {
    (void)x; CHECK(active->refreshed);
    if (active->fault == F_FINISH) return 0;
    s->mla[3].token_count = active->start + accepted; active->finished = 1; return 1;
}
static int observed_agree(ds4_tp *tp, const ds4_tp_native_cycle *c, uint32_t phase,
        uint32_t accepted, const uint32_t tokens[8], int ok, char *err, size_t size) {
    CHECK(active->s.checkpoint.len == (int)active->start);
    for (unsigned i = 0; i < 8; ++i) CHECK(active->accepted[i] == -1);
    if (phase < 9u) CHECK(!active->finished && !active->refreshed);
    if (phase == 9u && ok) CHECK(active->finished && active->refreshed);
    ++active->phases;
    return ds4_tp_native_agree(tp, c, phase, accepted, tokens, ok, err, size);
}
#define ds4_tp_native_agree observed_agree
#include "ds4_glm5_native_session.inc"
#undef ds4_tp_native_agree

static fixture *create(int fd, unsigned rank, unsigned rows, unsigned prefix) {
    fixture *f = calloc(1, sizeof(*f)); CHECK(f); active = f;
    ds4_session *s = &f->s; s->engine = &f->e;
    f->e.tp.active = 1; f->e.tp.ctx = ds4_tp_test_control_create(fd, rank); CHECK(f->e.tp.ctx);
    s->glm5_next_exec.tp = f->e.tp.ctx; s->glm5_next_exec.tp_rank = rank;
    s->ctx_size = 12288; s->tp_session_id = 17; s->glm5_native_rows = rows;
    s->glm5_next_ready = s->checkpoint_valid = true; s->glm5_native_previous_valid = prefix != 0;
    s->glm5_next_state.valid = s->glm5_native_state.valid = true;
    s->glm5_next_state.mla[3].token_count = prefix;
    s->glm5_native_state.mla[45].token_count = prefix ? prefix - 1 : 0;
    s->checkpoint.len = s->checkpoint.cap = (int)prefix;
    s->checkpoint.v = calloc(prefix ? prefix : 1u, sizeof(int)); CHECK(s->checkpoint.v);
    s->glm5_next_ws = ds4_glm5_next_draft_workspace_create_rows(1, s->ctx_size);
    s->glm5_native_ws = ds4_glm5_next_draft_workspace_create_rows(1, s->ctx_size);
    for (unsigned i = 0; i < 3; ++i)
        s->glm5_native_verify_ws[i] = ds4_glm5_next_draft_workspace_create_rows(2u << i, s->ctx_size);
    s->glm5_native_previous = ds4_gpu_tensor_alloc(row_bytes);
    s->glm5_native_chain = ds4_gpu_tensor_alloc(row_bytes);
    s->glm5_native_hidden = ds4_gpu_tensor_alloc(row_bytes);
    s->glm5_native_normalized = ds4_gpu_tensor_alloc(256u * row_bytes);
    s->glm5_native_shifted = ds4_gpu_tensor_alloc(256u * row_bytes);
    s->glm5_native_hc[0] = ds4_gpu_tensor_alloc(rows * row_bytes * 4u);
    s->glm5_native_hc[1] = ds4_gpu_tensor_alloc(rows * row_bytes * 4u);
    s->glm5_native_logits = ds4_gpu_tensor_alloc(rows * DS4_N_VOCAB * sizeof(float));
    s->glm5_next_cur = ds4_gpu_tensor_alloc(row_bytes * 4u);
    s->glm5_next_logits = ds4_gpu_tensor_alloc(DS4_N_VOCAB * sizeof(float));
    s->glm5_native_host_logits = calloc(rows * DS4_N_VOCAB, sizeof(float));
    s->sample_probs = calloc(DS4_N_VOCAB, sizeof(float)); s->logits = calloc(DS4_N_VOCAB, sizeof(float));
    CHECK(s->glm5_native_host_logits && s->sample_probs && s->logits);
    ((float *)s->glm5_native_previous->data)[0] = 1000.0f + prefix - 1u;
    return f;
}
static void destroy(fixture *f) {
    ds4_session *s = &f->s;
    ds4_tp_test_control_destroy(f->e.tp.ctx);
    ds4_gpu_tensor *t[] = {s->glm5_native_previous, s->glm5_native_chain, s->glm5_native_hidden,
        s->glm5_native_normalized, s->glm5_native_shifted, s->glm5_native_hc[0], s->glm5_native_hc[1],
        s->glm5_native_logits, s->glm5_next_cur, s->glm5_next_logits};
    for (unsigned i = 0; i < sizeof(t) / sizeof(t[0]); ++i) ds4_gpu_tensor_free(t[i]);
    free(s->glm5_native_ws); free(s->glm5_native_warm_ws); free(s->glm5_next_ws);
    for (unsigned i = 0; i < 3; ++i) free(s->glm5_native_verify_ws[i]);
    free(s->checkpoint.v); free(s->glm5_native_host_logits); free(s->sample_probs); free(s->logits); free(f);
    active = NULL;
}
static void prepare(fixture *f, unsigned rows, unsigned accepted, unsigned fault, int root, int eos) {
    f->wanted = accepted; f->fault = fault; f->step = f->views = f->phases = 0;
    f->refreshed = f->finished = 0; f->start = f->s.checkpoint.len;
    f->command = (ds4_tp_native_cycle){17u, f->s.glm5_native_cycle + 1u, f->start, rows, root, eos};
    for (unsigned i = 0; i < 8; ++i) f->accepted[i] = -1;
}
static void *run(void *p) {
    fixture *f = p; active = f;
    f->result = ds4_session_glm5_native_cycle_run(&f->s, &f->command, f->accepted, f->error, sizeof(f->error));
    return NULL;
}
static void run_pair(fixture *a, fixture *b, unsigned expected) {
    pthread_t peer; CHECK(pthread_create(&peer, NULL, run, b) == 0); run(a);
    CHECK(pthread_join(peer, NULL) == 0);
    fixture *both[] = {a, b};
    for (unsigned i = 0; i < 2; ++i) {
        fixture *f = both[i]; ds4_session *s = &f->s;
        CHECK(f->result == (expected ? (int)expected : -1));
        if (expected) {
            CHECK(s->checkpoint_valid && s->glm5_next_state.valid && s->glm5_native_state.valid);
            CHECK(s->checkpoint.len == (int)(f->start + expected));
            CHECK(s->glm5_native_state.mla[45].token_count + 1u == (unsigned)s->checkpoint.len);
            CHECK(s->glm5_next_state.mla[3].token_count == (unsigned)s->checkpoint.len);
            CHECK(f->phases == f->command.rows + 2u && f->finished && f->refreshed);
            for (unsigned t = 0; t < expected; ++t) {
                CHECK(f->accepted[t] == f->command.root + (int)t);
                CHECK(s->checkpoint.v[f->start + t] == f->accepted[t]);
            }
            float oracle[DS4_N_VOCAB];
            const unsigned next = expected < f->wanted ? (unsigned)f->command.root + expected : 1u;
            fill_logits(oracle, next); CHECK(memcmp(oracle, s->logits, sizeof(oracle)) == 0);
            CHECK(((float *)s->glm5_next_cur->data)[0] == 1000.0f + f->start + expected - 1u);
        } else {
            CHECK(!s->checkpoint_valid && !s->glm5_native_previous_valid && s->checkpoint.len == 0);
            CHECK(!s->glm5_next_state.valid && !s->glm5_native_state.valid && ds4_tp_failed(f->e.tp.ctx));
            for (unsigned t = 0; t < 8; ++t) CHECK(f->accepted[t] == -1);
        }
    }
    ++cases;
}
static void cycle_cases(void) {
    const unsigned prefixes[] = {1, 3, 8191, 8192};
    for (unsigned p = 0; p < 4; ++p) for (unsigned rows = 2; rows <= 8; rows *= 2)
        for (unsigned k = 1; k <= rows; ++k) {
            int fd[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
            fixture *a = create(fd[0], 0, rows, prefixes[p]), *b = create(fd[1], 1, rows, prefixes[p]);
            int root = 9;
            for (unsigned cycle = 0; cycle < 3; ++cycle) {
                prepare(a, rows, k, F_NONE, root, -1); prepare(b, rows, k, F_NONE, root, -1);
                run_pair(a, b, k); root = glm5_native_argmax(a->s.logits);
            }
            destroy(a); destroy(b);
        }
    for (unsigned fault = F_VIEW; fault <= F_SYNC; ++fault) for (unsigned rank = 0; rank < 2; ++rank) {
        int fd[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
        fixture *a = create(fd[0], 0, 8, 8192), *b = create(fd[1], 1, 8, 8192);
        prepare(a, 8, 4, rank == 0 ? fault : F_NONE, 9, -1);
        prepare(b, 8, 4, rank == 1 ? fault : F_NONE, 9, -1);
        run_pair(a, b, 0); destroy(a); destroy(b);
    }
    for (unsigned fault = 0; fault < 9; ++fault) for (unsigned rank = 0; rank < 2; ++rank) {
        int fd[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
        fixture *a = create(fd[0], 0, 8, 8192), *b = create(fd[1], 1, 8, 8192);
        prepare(a, 8, 4, F_NONE, 9, -1); prepare(b, 8, 4, F_NONE, 9, -1);
        fixture *bad = rank ? b : a;
        switch (fault) {
        case 0: bad->s.checkpoint_valid = false; break;
        case 1: bad->s.glm5_native_previous_valid = false; break;
        case 2: bad->s.glm5_next_state.valid = false; break;
        case 3: bad->s.glm5_native_state.valid = false; break;
        case 4: --bad->s.glm5_native_state.mla[45].token_count; break;
        case 5: ++bad->s.glm5_native_cycle; break;
        case 6: ++bad->s.tp_session_id; break;
        case 7: bad->s.ctx_size = 8193; break;
        case 8: bad->s.glm5_native_rows = 4; break;
        }
        run_pair(a, b, 0); CHECK(!a->step && !b->step); destroy(a); destroy(b);
    }
    int fd[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
    fixture *a = create(fd[0], 0, 8, 3), *b = create(fd[1], 1, 8, 3);
    prepare(a, 8, 8, F_NONE, 9, 12); prepare(b, 8, 8, F_NONE, 9, 12);
    /* CLI stops on EOS, while fixed-length benchmarks choose the best non-EOS
     * root. Leave that decision at the caller's next ordinary frontier. */
    run_pair(a, b, 3);
    CHECK(glm5_native_argmax(a->s.logits) == 12 && glm5_native_argmax(b->s.logits) == 12);
    CHECK(a->accepted[3] == -1 && b->accepted[3] == -1);
    destroy(a); destroy(b);
}
static void teacher_cases(void) {
    ds4_session disabled = {0};
    CHECK(ds4_session_glm5_native_teach(&disabled, NULL, NULL, NULL, 0)); ++cases;
    const unsigned chunks[] = {1, 2, 17, 256, 1, 255, 7, 1024, 513};
    for (unsigned seeded = 0; seeded < 2; ++seeded) {
        fixture *f = create(-1, 0, 8, seeded ? 8191 : 0); active = f;
        for (unsigned c = 0; c < sizeof(chunks) / sizeof(chunks[0]); ++c) {
            const unsigned rows = chunks[c], prefix = f->s.checkpoint.len, before = f->warm_count;
            ds4_glm5_next_workspace w = {rows}; int ids[1024];
            ds4_gpu_tensor *hc = ds4_gpu_tensor_alloc(rows * row_bytes * 4u);
            for (unsigned t = 0; t < rows; ++t) {
                ids[t] = 3 + (prefix + t) % 40;
                for (unsigned j = 0; j < DS4_N_EMBD * 4; ++j)
                    ((float *)hc->data)[t * DS4_N_EMBD * 4u + j] = 1000.0f + prefix + t;
            }
            CHECK(ds4_session_glm5_native_teach(&f->s, &w, hc, ids, rows));
            const unsigned skip = prefix ? 0 : 1;
            CHECK(f->warm_count == before + rows - skip);
            for (unsigned t = skip; t < rows; ++t) {
                CHECK(f->warm_ids[before + t - skip] == (unsigned)ids[t]);
                CHECK(f->warm_previous[before + t - skip] == 1000.0f + prefix + t - 1u);
            }
            CHECK(((float *)f->s.glm5_native_previous->data)[0] == 1000.0f + prefix + rows - 1u);
            f->s.checkpoint.len += rows;
            CHECK(f->s.glm5_native_state.mla[45].token_count + 1u == (unsigned)f->s.checkpoint.len);
            ds4_gpu_tensor_free(hc); ++cases;
        }
        destroy(f);
    }
    fixture *f = create(-1, 0, 8, 3); active = f;
    ds4_glm5_next_workspace w = {1}; const int ids[] = {4, 5};
    ds4_gpu_tensor *hc = ds4_gpu_tensor_alloc(2u * row_bytes * 4u);
    CHECK(!ds4_session_glm5_native_teach(&f->s, &w, hc, ids, 2));
    CHECK(!f->s.glm5_next_state.valid && !f->s.glm5_native_state.valid);
    ds4_gpu_tensor_free(hc); destroy(f); ++cases;
}
int main(void) {
    teacher_cases(); cycle_cases();
    printf("PASS native session cases=%u production_orchestration=1 arithmetic_doubles=1 real_socket_agreement=1\n", cases);
    return 0;
}
