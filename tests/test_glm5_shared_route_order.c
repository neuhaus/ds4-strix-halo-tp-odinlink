// Exercise the actual executor seam; discard unrelated executor functions at
// link time. GPU operations and peer checks are deterministic boundary mocks.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <assert.h>
#include "../ds4_glm5_next_exec.c"

struct ds4_tp { int rank, failed, peer_result; bool rdma; };
static char events[32];
static unsigned event_count, fail_op;
static uint64_t expected_hash, expected_seq;
static ds4_glm5_next_workspace *workspace;
static ds4_glm5_next_exec_ctx *context;
static unsigned shared_calls;
static int event(char ch) {
    assert(event_count+1 < sizeof(events));
    events[event_count++] = ch; events[event_count] = 0;
    return !fail_op || event_count != fail_op;
}
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *t) { return t ? t->bytes : 0; }
int ds4_gpu_tensor_read(const ds4_gpu_tensor *t, uint64_t off, void *data, uint64_t bytes) {
    if (!t || !t->ptr || off>t->bytes || bytes>t->bytes-off) return 0;
    if (!event(t==workspace->router_selected ? 'I' : 'W')) return 0;
    memcpy(data,(const char *)t->ptr+off,bytes); return 1;
}
int ds4_tp_rank(const ds4_tp *tp) { return tp->rank; }
bool ds4_tp_is_rdma(const ds4_tp *tp) { return tp->rdma; }
bool ds4_tp_big_gate_is_rdma_capable(const ds4_tp *tp) { return tp->rdma; }
bool ds4_tp_big_gate_is_direct(const ds4_tp *tp, const void *a, const void *b, uint64_t n) {
    return tp->rdma && a && b && n==4096*sizeof(float);
}
void ds4_tp_mark_failed(ds4_tp *tp) { tp->failed=1; }
int ds4_tp_hash_check(ds4_tp *tp, uint64_t seq, uint64_t hash, char *err, size_t len) {
    assert(seq==expected_seq && hash==expected_hash);
    (void)err; (void)len; event('H'); return tp->peer_result;
}
int ds4_gpu_matmul_q8_0_tensor(ds4_gpu_tensor *out, const void *map,
        uint64_t size, uint64_t offset, uint64_t k, uint64_t n,
        const ds4_gpu_tensor *x, uint64_t m) {
    const bool gate = out==workspace->shared_gate;
    assert(gate || out==workspace->shared_up);
    const ds4_glm5_next_ffn_offsets *f = &context->model->layer[3].ffn_weight;
    assert(offset==(gate ? f->gate_shexp : f->up_shexp)+context->tp_rank*1024ull*4352);
    assert(map==context->model_map && size==context->model_size);
    assert(k==4096 && n==1024 && m==1 && x==workspace->ffn_hidden);
    ++shared_calls; return event(gate ? 'G' : 'U');
}
int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *g,
        const ds4_gpu_tensor *u, uint32_t n, float clamp, float weight) {
    assert(out==workspace->shared_mid && g==workspace->shared_gate && u==workspace->shared_up);
    assert(n==1024 && clamp==10.0f && weight==1.0f);
    ++shared_calls; return event('S');
}
int ds4_gpu_matmul_q8_0_kslice_tensor(ds4_gpu_tensor *out, const void *map,
        uint64_t size, uint64_t offset, uint64_t k, uint64_t base,
        uint64_t count, uint64_t n, const ds4_gpu_tensor *x, uint64_t xbase) {
    assert(out==workspace->shared_out && x==workspace->shared_mid && xbase==0);
    assert(offset==context->model->layer[3].ffn_weight.down_shexp);
    assert(map==context->model_map && size==context->model_size);
    assert(k==2048 && base==context->tp_rank*1024 && count==1024 && n==4096);
    ++shared_calls; return event('D');
}
static int invoke(bool early) {
    memset(events,0,sizeof(events)); event_count=0; shared_calls=0; context->tp->failed=0;
    return route_agrees(context,3,8192,workspace->router_selected,
                        workspace->router_weights,NULL,NULL,early ? workspace : NULL);
}
int main(void) {
    ds4_tp tp = {.rdma=true,.peer_result=1};
    int32_t ids[8]={0,1,2,3,4,5,6,287};
    float weights[8]={0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f};
    ds4_gpu_tensor selected={.ptr=ids,.bytes=sizeof(ids)}, rw={.ptr=weights,.bytes=sizeof(weights)};
    ds4_gpu_tensor shared[5]={{0}}, slab={.bytes=16384};
    ds4_glm5_next_workspace w={.decode_phase=true,.router_selected=&selected,
        .router_weights=&rw,.shared_gate=&shared[0],.shared_up=&shared[1],
        .shared_mid=&shared[2],.shared_out=&shared[3],.ffn_hidden=&shared[4]};
    ds4_glm5_next_model_offsets model={0};
    model.layer[3].ffn_weight.gate_shexp=123;
    model.layer[3].ffn_weight.up_shexp=456;
    model.layer[3].ffn_weight.down_shexp=789;
    uint64_t sequence=73;
    ds4_glm5_next_exec_ctx ctx={.model=&model,.model_map=&model,.model_size=10000000,
        .tp=&tp,.tp_sequence=&sequence,.tp_big_out=&slab,.tp_big_in=&slab,
        .tp_big_out_host=&slab,.tp_big_in_host=&slab};
    workspace=&w; context=&ctx;
    expected_hash=fnv64_continue(UINT64_C(1469598103934665603),ids,sizeof(ids));
    expected_hash=fnv64_continue(expected_hash,weights,sizeof(weights));
    const uint64_t fields[]={UINT64_C(0x474c4d3500000000),73,3,8192,1};
    expected_seq=fnv64_continue(UINT64_C(1469598103934665603),fields,sizeof(fields));
    unsetenv("DS4_ROCM_GLM5_SHARED_ROUTE_OVERLAP");
    unsetenv("DS4_ROCM_GLM5_SHARED_Q8_PAIR_DECODE");
    unsetenv("DS4_ROCM_GLM5_WINDOW_OVERLAP");
    unsetenv("DS4_ROCM_GLM5_WINDOW_SCRATCH");
    assert(shared_route_overlap_mode(&ctx,&w,1)==0);
    setenv("DS4_ROCM_GLM5_SHARED_ROUTE_OVERLAP","1",1);
    for (unsigned rank=0; rank<2; ++rank) {
        ctx.tp_rank=rank; tp.rank=rank;
        assert(shared_route_overlap_mode(&ctx,&w,1)==1);
        assert(invoke(false)==1 && strcmp(events,"IWH")==0 && !shared_calls);
        assert(invoke(true)==1 && strcmp(events,"IWGUSDH")==0 && shared_calls==4);
        assert(sequence==73);
        for (unsigned bad=0; bad<4; ++bad) {
            const int32_t old_id=ids[7]; const float old_weight=weights[7];
            if (bad==0) ids[7]=288;
            if (bad==1) ids[7]=ids[0];
            if (bad==2) weights[7]=NAN;
            if (bad==3) weights[7]=-0.1f;
            assert(invoke(true)==0 && shared_calls==0 && strchr(events,'H')==NULL);
            ids[7]=old_id; weights[7]=old_weight;
        }
        for (fail_op=1; fail_op<=6; ++fail_op) {
            assert(invoke(true)==0 && strchr(events,'H')==NULL);
        }
        fail_op=0;
        tp.peer_result=-1;
        assert(invoke(true)==0 && strcmp(events,"IWGUSDH")==0 && tp.failed);
        tp.peer_result=1;
    }
    assert(shared_route_overlap_mode(&ctx,&w,0)==-1);
    assert(shared_route_overlap_mode(&ctx,&w,-1)==-1);
    tp.rdma=false; assert(shared_route_overlap_mode(&ctx,&w,1)==-1); tp.rdma=true;
    setenv("DS4_ROCM_GLM5_SHARED_Q8_PAIR_DECODE","1",1);
    assert(shared_route_overlap_mode(&ctx,&w,1)==-1);
    unsetenv("DS4_ROCM_GLM5_SHARED_Q8_PAIR_DECODE");
    setenv("DS4_ROCM_GLM5_WINDOW_OVERLAP", "1", 1);
    setenv("DS4_ROCM_GLM5_WINDOW_SCRATCH", "1", 1);
    w.decode_phase=true;
    assert(shared_route_overlap_mode(&ctx,&w,1)==-1);
    unsetenv("DS4_ROCM_GLM5_WINDOW_SCRATCH");
    assert(shared_route_overlap_mode(&ctx,&w,1)==1);
    unsetenv("DS4_ROCM_GLM5_WINDOW_OVERLAP");
    w.decode_phase=false; assert(shared_route_overlap_mode(&ctx,&w,1)==0);
    setenv("DS4_ROCM_GLM5_SHARED_ROUTE_OVERLAP","yes",1);
    assert(shared_route_overlap_mode(&ctx,&w,1)==-1);
    puts("PASS shared route ordering, both rank slices, invalid routes, enqueue failures, peer disagreement and admission");
    return 0;
}
