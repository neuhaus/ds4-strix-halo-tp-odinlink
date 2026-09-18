/* Diagnostic link wrappers. No engine/GPU object is rebuilt for this probe.
 * Build with --wrap for the names below; enabled only by DS4_TP_TRANSPORT_PROBE=1.
 * Wall time inside transport includes peer compute skew. Socket time is a
 * subset, never an extra cost to add. No GPU fences or hot-path log writes.
 * The measured host inference path is synchronous and single-threaded. */
#include "ds4.h"
#include "ds4_tp.h"
#include "ds4_glm5_next_exec.h"
#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

enum { OUTSIDE, PREFILL, ORDINARY, NATIVE, DRAFT, VERIFY, REFRESH, PHASES };
enum { GATE, AUX, BULK, LOGITS, HASH, AGREE_START, AGREE_DRAFT, AGREE_FINISH, ROUTE, COMPUTE, COMMAND, ACK, GPU_SYNC, BATCH, WAVES, OPS };
static const char *phase_names[] = {"outside","prefill","ordinary","native_other","draft","verify","refresh"};
static const char *op_names[] = {"gate","aux","bulk","logits","route_hash","native_start","native_proposal","native_finish","layer_route","layer_compute","command_send","command_ack","gpu_sync","unexpected_batch","unexpected_waves"};
typedef struct { unsigned long long calls, failures, bytes; double wall, socket; } stat;
static stat stats[PHASES][OPS];
static unsigned long long phase_calls[PHASES];
static unsigned long long phase_units[PHASES];
static double phase_wall[PHASES];
static int enabled, probe_rank = -1;
static pthread_t owner;
static __thread int phase, depth;
static __thread double socket_time;
static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static void report(void) {
    for (int p=0; p<PHASES; ++p) {
        if (phase_calls[p]) fprintf(stderr,
            "TP_PROBE_SCOPE rank=%d phase=%s calls=%llu units=%llu wall_ms=%.6f\n",
            probe_rank,p==NATIVE ? "native_cycle_inclusive" : phase_names[p],
            phase_calls[p],phase_units[p],phase_wall[p]*1000);
        for (int o=0; o<OPS; ++o) {
            const stat *s=&stats[p][o]; if (!s->calls) continue;
            fprintf(stderr,
                "TP_PROBE rank=%d phase=%s op=%s calls=%llu failures=%llu tx_payload_bytes=%llu wall_ms=%.6f socket_ms=%.6f\n",
                probe_rank,phase_names[p],op_names[o],s->calls,s->failures,s->bytes,s->wall*1000,s->socket*1000);
        }
    }
}
__attribute__((constructor)) static void init(void) {
    const char *e=getenv("DS4_TP_TRANSPORT_PROBE");
    enabled=e && !strcmp(e,"1");
    owner=pthread_self();
    if (enabled) atexit(report);
}
static int active(void) {
    if (enabled && !pthread_equal(owner,pthread_self())) {
        fprintf(stderr,"TP_PROBE invalid: transport/scope called from another thread\n");
        _Exit(3);
    }
    return enabled;
}
#define SCOPE(name, params, args, scope, units) \
    extern int __real_##name params; \
    int __wrap_##name params { \
        if (!active()) return __real_##name args; \
        const int saved=phase; phase=scope; const double probe_start=now(); \
        const int result=__real_##name args; const int saved_errno=errno; \
        phase_wall[scope]+=now()-probe_start; ++phase_calls[scope]; \
        phase_units[scope]+=(units); phase=saved; \
        errno=saved_errno; return result; \
    }
SCOPE(ds4_session_sync,
    (ds4_session *s,const ds4_tokens *p,char *e,size_t n),(s,p,e,n),PREFILL,0)
SCOPE(ds4_session_eval,
    (ds4_session *s,int t,char *e,size_t n),(s,t,e,n),ORDINARY,1)
SCOPE(ds4_session_eval_speculative_argmax,
    (ds4_session *s,int t,int m,int eos,int *a,int cap,char *e,size_t n),
    (s,t,m,eos,a,cap,e,n),NATIVE,0)
SCOPE(ds4_session_tp_glm5_native_cycle,
    (ds4_session *s,uint64_t id,uint64_t c,uint32_t p,int root,uint32_t rows,int eos,char *e,size_t n),
    (s,id,c,p,root,rows,eos,e,n),NATIVE,0)
SCOPE(ds4_glm5_next_draft_step,
    (const ds4_glm5_next_exec_ctx *x,ds4_glm5_next_mla_state *s,ds4_glm5_next_workspace *w,
     const ds4_gpu_tensor *prev,uint32_t token,ds4_gpu_tensor *h,ds4_gpu_tensor *l),
    (x,s,w,prev,token,h,l),DRAFT,1)
SCOPE(ds4_glm5_next_target_verify,
    (const ds4_glm5_next_exec_ctx *x,ds4_glm5_next_state *s,ds4_glm5_next_workspace *b,
     ds4_glm5_next_workspace *w,const uint32_t *t,uint32_t n,ds4_gpu_tensor *a,
     ds4_gpu_tensor *h,ds4_gpu_tensor *l),(x,s,b,w,t,n,a,h,l),VERIFY,n)
SCOPE(ds4_glm5_next_draft_refresh,
    (const ds4_glm5_next_exec_ctx *x,ds4_glm5_next_state *s,ds4_glm5_next_workspace *w,
     const ds4_gpu_tensor *h,const uint32_t *t,uint32_t p,uint32_t n,uint32_t k,ds4_gpu_tensor *prev),
    (x,s,w,h,t,p,n,k,prev),REFRESH,k)

#define CALL(name, params, args, op, payload) \
    extern int __real_##name params; \
    int __wrap_##name params { \
        if (!active() || depth) return __real_##name args; \
        if (tp) probe_rank=ds4_tp_rank(tp); \
        const double old_socket=socket_time, probe_start=now(); ++depth; \
        const int result=__real_##name args; const int saved_errno=errno; \
        const double elapsed=now()-probe_start; --depth; \
        stat *s=&stats[phase][op]; ++s->calls; s->failures+=(result!=1); \
        s->bytes+=(payload); s->wall+=elapsed; s->socket+=socket_time-old_socket; \
        errno=saved_errno; return result; \
    }
CALL(ds4_tp_gate_exchange,
    (ds4_tp *tp,uint32_t l,uint32_t g,uint64_t q),(tp,l,g,q),GATE,ds4_tp_vec_bytes(tp))
CALL(ds4_tp_gate_exchange_from_registered,
    (ds4_tp *tp,uint32_t l,uint32_t g,uint64_t q,const void *p),(tp,l,g,q,p),GATE,ds4_tp_vec_bytes(tp))
CALL(ds4_tp_native_gate_exchange_next,
    (ds4_tp *tp,uint32_t l,uint32_t g,const void *p),(tp,l,g,p),GATE,ds4_tp_vec_bytes(tp))
CALL(ds4_tp_aux_gate_exchange,
    (ds4_tp *tp,uint32_t l),(tp,l),AUX,ds4_tp_aux_payload_bytes(tp))
CALL(ds4_tp_big_gate_exchange,
    (ds4_tp *tp,uint32_t l,uint64_t q,const void *o,void *i,uint64_t b),(tp,l,q,o,i,b),BULK,b)
CALL(ds4_tp_batch_gate_exchange,
    (ds4_tp *tp,uint32_t l,uint32_t r,uint64_t q),(tp,l,r,q),BATCH,r*ds4_tp_vec_bytes(tp))
CALL(ds4_tp_big_gate_exchange_waves,
    (ds4_tp *tp,uint32_t l,uint64_t q,const void *o,void *i,uint64_t b,uint64_t wb,
     uint32_t w,ds4_tp_big_wave_ready_fn fn,void *ud),(tp,l,q,o,i,b,wb,w,fn,ud),WAVES,b)
CALL(ds4_tp_exchange_logits_halves,
    (ds4_tp *tp,float *l,uint32_t n),(tp,l,n),LOGITS,(uint64_t)n*sizeof(float))
CALL(ds4_tp_hash_check,
    (ds4_tp *tp,uint64_t q,uint64_t h,char *e,size_t n),(tp,q,h,e,n),HASH,0)
CALL(ds4_tp_native_agree,
    (ds4_tp *tp,const ds4_tp_native_cycle *c,uint32_t p,uint32_t a,const uint32_t t[8],int ok,char *e,size_t n),
    (tp,c,p,a,t,ok,e,n),(p==0 ? AGREE_START : p<8 ? AGREE_DRAFT : AGREE_FINISH),0)
CALL(ds4_tp_verify_layer_agree,
    (ds4_tp *tp,uint64_t q,uint32_t l,uint32_t f,uint32_t r,uint32_t p,uint64_t h,int ok,char *e,size_t n),
    (tp,q,l,f,r,p,h,ok,e,n),(p ? COMPUTE : ROUTE),0)
CALL(ds4_tp_send_native_cycle,
    (ds4_tp *tp,const ds4_tp_native_cycle *c),(tp,c),COMMAND,0)
CALL(ds4_tp_send_eval,
    (ds4_tp *tp,uint64_t id,uint64_t q,int t),(tp,id,q,t),COMMAND,0)
CALL(ds4_tp_send_sync,
    (ds4_tp *tp,uint64_t id,const int *t,uint32_t n),(tp,id,t,n),COMMAND,0)
CALL(ds4_tp_wait_command_ack,
    (ds4_tp *tp,uint64_t id,const char *o,char *e,size_t n),(tp,id,o,e,n),ACK,0)
/* GPU work completed at existing boundaries is reported separately from TP.
 * It is not all synchronization overhead: most is useful preceding compute. */
extern int __real_ds4_gpu_synchronize(void);
int __wrap_ds4_gpu_synchronize(void) {
    if (!active() || depth) return __real_ds4_gpu_synchronize();
    const double t=now(); const int r=__real_ds4_gpu_synchronize(), saved_errno=errno;
    stat *s=&stats[phase][GPU_SYNC]; ++s->calls; s->failures+=(r!=1); s->wall+=now()-t;
    errno=saved_errno; return r;
}
#define SOCKET(name, type, params, args) \
    extern type __real_##name params; \
    type __wrap_##name params { \
        if (!depth) return __real_##name args; \
        const double t=now(); const type r=__real_##name args; const int e=errno; \
        socket_time+=now()-t; errno=e; return r; \
    }
SOCKET(read,ssize_t,(int fd,void *p,size_t n),(fd,p,n))
SOCKET(write,ssize_t,(int fd,const void *p,size_t n),(fd,p,n))
SOCKET(writev,ssize_t,(int fd,const struct iovec *p,int n),(fd,p,n))
SOCKET(send,ssize_t,(int fd,const void *p,size_t n,int f),(fd,p,n,f))
SOCKET(recv,ssize_t,(int fd,void *p,size_t n,int f),(fd,p,n,f))
SOCKET(poll,int,(struct pollfd *p,nfds_t n,int timeout),(p,n,timeout))
