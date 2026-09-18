// Local numerical integration test. The peer contribution is explicitly
// simulated by echoing the local payload; no network is used. This cannot
// establish TP correctness, zero-fallback transport, model quality or t/s.
#include "ds4_glm5_next_exec.h"
#include "ds4_gpu_mgpu.h"
extern "C" {
#include "ds4_tp.h"
}
#include "glm5_next_real_offsets.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <string>

#define REQUIRE(x) do { if (!(x)) { \
    std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); std::exit(1); \
} } while (0)

struct ds4_tp {
    unsigned rank = 0;
    uint32_t features = DS4_TP_FEATURE_GLM5_KDA_TP |
        DS4_TP_FEATURE_GLM5_KDA_OUTPUT_ROWSLICE | DS4_TP_FEATURE_GLM5_SMALL_GATE;
    unsigned char *slab = nullptr;
    bool capable = true, failed = false;
    unsigned calls = 0, fail_call = 0, bulk_calls = 0, aux_calls = 0;
    unsigned layer_agrees[2] = {}, handoff_bulk_calls = 0, handoff_pending = 0;
    int fail_agree_phase = -1;
    bool fail_handoff_bulk = false;
    uint64_t handoff_sequence = 0, handoff_hash = 0;
    uint32_t handoff_layer = 0, handoff_frontier = 0, handoff_rows = 0;
    uint64_t prefill_config = 0;
    uint64_t latency_seq = 0;
};
extern "C" {
void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}
int ds4_tp_rank(const ds4_tp *p) { return p->rank; }
bool ds4_tp_is_rdma(const ds4_tp *p) { return p->capable; }
bool ds4_tp_big_gate_is_rdma_capable(const ds4_tp *p) { return p->capable; }
bool ds4_tp_big_gate_is_direct(const ds4_tp *p,const void *,const void *,uint64_t) { return p->capable; }
uint32_t ds4_tp_runtime_features(const ds4_tp *p) { return p->features; }
uint64_t ds4_tp_prefill_config(const ds4_tp *p) { return p->prefill_config; }
uint64_t ds4_tp_vec_bytes(const ds4_tp *) { return 16384; }
uint64_t ds4_tp_aux_payload_bytes(const ds4_tp *) { return 8192; }
uint64_t ds4_tp_slab_out_offset(const ds4_tp *,uint32_t,uint32_t) { return 0; }
uint64_t ds4_tp_slab_in_offset(const ds4_tp *,uint32_t,uint32_t) { return 16384; }
uint64_t ds4_tp_slab_aux_out_payload_offset(const ds4_tp *,uint32_t) { return 32768; }
uint64_t ds4_tp_slab_aux_in_payload_offset(const ds4_tp *,uint32_t) { return 40960; }
void ds4_tp_mark_failed(ds4_tp *p) { p->failed=true; p->handoff_pending=0; }
int ds4_tp_hash_check(ds4_tp *,uint64_t,uint64_t,char *,size_t) { return 1; }
int ds4_tp_verify_layer_agree(ds4_tp *p,uint64_t sequence,uint32_t layer,
        uint32_t frontier,uint32_t rows,uint32_t phase,uint64_t hash,
        int local_ok,char *,size_t) {
    REQUIRE(layer>=3 && layer<45 && layer%4!=3 && phase<2 &&
        ds4_tp_glm5_native_width_valid(rows) && frontier<=UINT32_MAX-rows &&
        (p->prefill_config & DS4_TP_CONFIG_GLM5_VERIFY_FFN_HANDOFF));
    ++p->layer_agrees[phase];
    if (!local_ok || p->failed || p->fail_agree_phase==(int)phase) {
        ds4_tp_mark_failed(p); return 0;
    }
    REQUIRE(p->handoff_pending==phase);
    if (!phase) {
        p->handoff_sequence=sequence; p->handoff_hash=hash;
        p->handoff_layer=layer; p->handoff_frontier=frontier; p->handoff_rows=rows;
    } else REQUIRE(p->handoff_sequence==sequence && p->handoff_hash==hash &&
        p->handoff_layer==layer && p->handoff_frontier==frontier && p->handoff_rows==rows);
    ++p->handoff_pending;
    return 1;
}
int ds4_tp_gate_exchange(ds4_tp *p,uint32_t,uint32_t,uint64_t) {
    if (++p->calls==p->fail_call) return 0;
    std::memcpy(p->slab+16384,p->slab,16384); return 1;
}
int ds4_tp_gate_exchange_from_registered(ds4_tp *p,uint32_t,uint32_t,uint64_t,const void *out) {
    if (++p->calls==p->fail_call) return 0;
    std::memcpy(p->slab+16384,out,16384); return 1;
}
int ds4_tp_native_gate_exchange_next(ds4_tp *p,uint32_t layer,uint32_t gate,const void *out) {
    const uint64_t seq = ++p->latency_seq;
    const unsigned ordinal = (unsigned)((seq - 1u) % 87u);
    REQUIRE((ordinal < 3u ? ordinal * 2u : ordinal + 3u) == layer * 2u + gate);
    return out ? ds4_tp_gate_exchange_from_registered(p,layer,gate,seq,out) :
                 ds4_tp_gate_exchange(p,layer,gate,seq);
}
int ds4_tp_aux_gate_exchange(ds4_tp *p,uint32_t) {
    ++p->aux_calls;
    if (++p->calls==p->fail_call) return 0;
    std::memcpy(p->slab+40960,p->slab+32768,8192); return 1;
}
int ds4_tp_big_gate_exchange(ds4_tp *p,uint32_t layer,uint64_t sequence,const void *out,void *in,uint64_t bytes) {
    if (p->handoff_pending) {
        REQUIRE(p->handoff_pending==2 && layer==p->handoff_layer &&
            sequence==p->handoff_sequence+1 && bytes==p->handoff_rows*16384u);
        p->handoff_pending=0; ++p->handoff_bulk_calls;
        if (p->fail_handoff_bulk) { ds4_tp_mark_failed(p); return 0; }
    }
    ++p->bulk_calls;
    if (++p->calls==p->fail_call) return 0;
    std::memcpy(in,out,bytes); return 1;
}
}

static constexpr uint64_t hc_row = 16384u*4u;
static constexpr unsigned context = 8464;
static uint64_t compared_values;
static unsigned cases;

struct Tensor {
    ds4_gpu_tensor *p;
    explicit Tensor(uint64_t bytes,bool host=false) : p(host?
        ds4_gpu_tensor_alloc_rdma_host(bytes):ds4_gpu_tensor_alloc(bytes)) { REQUIRE(p); }
    ~Tensor() { ds4_gpu_tensor_free(p); }
    operator ds4_gpu_tensor *() const { return p; }
};
struct State {
    ds4_glm5_next_state s = {};
    explicit State(const ds4_glm5_next_model_offsets &m, bool draft=false) {
        FILE *quiet=std::fopen("/dev/null","w"); REQUIRE(quiet);
        REQUIRE(draft ? ds4_glm5_next_draft_state_init(&s,&m,context,quiet) :
            ds4_glm5_next_state_init(&s,&m,context,quiet));
        std::fclose(quiet);
    }
    ~State() { ds4_glm5_next_state_free(&s); }
};
struct Workspace {
    ds4_glm5_next_workspace *p;
    explicit Workspace(unsigned m, bool draft=false) : p(draft ?
        ds4_glm5_next_draft_workspace_create_rows(m,context) :
        ds4_glm5_next_workspace_create_capacity_context(m,context)) {
        REQUIRE(p); ds4_glm5_next_workspace_begin_decode(p);
    }
    ~Workspace() { ds4_glm5_next_workspace_destroy(p); }
    operator ds4_glm5_next_workspace *() const { return p; }
};

static std::vector<float> read(ds4_gpu_tensor *t,uint64_t count) {
    std::vector<float> v(count);
    if (count) REQUIRE(ds4_gpu_tensor_read(t,0,v.data(),count*4u));
    for (float x:v) REQUIRE(std::isfinite(x));
    return v;
}
static void equal(const char *name,ds4_gpu_tensor *a,ds4_gpu_tensor *b,uint64_t count) {
    const auto av=read(a,count), bv=read(b,count);
    uint64_t different=0;
    for (size_t i=0;i<av.size();++i) different+=std::memcmp(&av[i],&bv[i],4)!=0;
    if (different) std::fprintf(stderr,"DIFF %s count=%llu differing=%llu\n",name,
        (unsigned long long)count,(unsigned long long)different);
    REQUIRE(!different); compared_values+=count;
}
static void equal_layer(State &a,State &b,unsigned il) {
    REQUIRE(a.s.valid && b.s.valid);
    if (il<45 && il%4!=3) {
        auto &x=a.s.kda.layer[il], &y=b.s.kda.layer[il];
        REQUIRE(x.token_count==y.token_count);
        equal("recurrent",x.recurrent,y.recurrent,64u*128*128);
        equal("q_history",x.q_history,y.q_history,8192*3);
        equal("k_history",x.k_history,y.k_history,8192*3);
        equal("v_history",x.v_history,y.v_history,8192*3);
    } else {
        auto &x=a.s.mla[il], &y=b.s.mla[il];
        REQUIRE(x.token_count==y.token_count && x.complete_pools==y.complete_pools && x.tail_count==y.tail_count);
        equal("index_tail",x.index_tail,y.index_tail,4*128);
        equal("gate_tail",x.pool_gate_tail,y.pool_gate_tail,4*128);
        equal("compact_kv",x.compact_kv,y.compact_kv,(uint64_t)x.token_count*512);
        equal("index_pool",x.index_pool,y.index_pool,x.complete_pools*128);
    }
}

static void seed_mla(State &state,unsigned il,unsigned prefix) {
    auto &s=state.s.mla[il];
    REQUIRE(ds4_gpu_tensor_fill_f32(s.compact_kv,0.03125f,(uint64_t)context*512));
    REQUIRE(ds4_gpu_tensor_fill_f32(s.index_pool,0.0625f,s.capacity_pools*128));
    REQUIRE(ds4_gpu_tensor_fill_f32(s.index_tail,0.09375f,4*128));
    REQUIRE(ds4_gpu_tensor_fill_f32(s.pool_gate_tail,0.015625f,4*128));
    REQUIRE(ds4_gpu_tensor_fill_f32(s.index_pool_valid,1.0f,s.capacity_pools));
    std::vector<int32_t> ids(s.capacity_pools*4);
    for (unsigned i=0;i<ids.size();++i) ids[i]=(int32_t)i;
    REQUIRE(ds4_gpu_tensor_write(s.index_pool_ids,0,ids.data(),ids.size()*4));
    s.token_count=prefix; s.complete_pools=prefix/4; s.tail_count=prefix%4;
}

static void serial_row(ds4_glm5_next_exec_ctx &x,unsigned il,State &s,Workspace &w,
                        ds4_gpu_tensor *inputs,unsigned input_row,ds4_gpu_tensor *output) {
    auto *in=ds4_gpu_tensor_view(inputs,input_row*hc_row,hc_row);
    REQUIRE(in && ds4_glm5_next_layer_forward(&x,il,&s.s,w,in,output));
    ds4_gpu_tensor_free(in);
}

static void run_case(ds4_glm5_next_exec_ctx &x,unsigned il,unsigned m,unsigned prefix,unsigned accepted) {
    State base(*x.model), reference(*x.model), candidate(*x.model);
    Workspace scalar(1), batch(m);
    Tensor inputs((m+5)*hc_row), serial(m*hc_row), got(m*hc_row), next_a(hc_row), next_b(hc_row);
    std::vector<float> host((m+5)*16384u);
    for (size_t i=0;i<host.size();++i)
        host[i]=float(int((i*193+(i/16384)*761)%997)-498)/1001.3f;
    REQUIRE(ds4_gpu_tensor_write(inputs,0,host.data(),host.size()*4));
    if (il%4==3) {
        seed_mla(base,il,prefix); seed_mla(reference,il,prefix); seed_mla(candidate,il,prefix);
    } else for (unsigned i=0;i<prefix;++i) {
        serial_row(x,il,base,scalar,inputs,m+i,next_a);
        serial_row(x,il,reference,scalar,inputs,m+i,next_a);
        serial_row(x,il,candidate,scalar,inputs,m+i,next_a);
    }
    REQUIRE(ds4_glm5_next_layer_verify_reserve(&x,il,&candidate.s,m));
    auto *verify_inputs=ds4_gpu_tensor_view(inputs,0,m*hc_row); REQUIRE(verify_inputs);
    const unsigned aux_before=x.tp->aux_calls;
    const unsigned bulk_before=x.tp->bulk_calls;
    REQUIRE(ds4_glm5_next_layer_verify(&x,il,&candidate.s,batch,scalar,verify_inputs,got,m));
    REQUIRE(x.tp->aux_calls==aux_before && x.tp->bulk_calls>bulk_before);
    REQUIRE(!ds4_glm5_next_layer_verify(&x,il,&candidate.s,batch,scalar,verify_inputs,got,m) && candidate.s.valid);
    equal_layer(base,candidate,il);
    auto *one=ds4_gpu_tensor_view(inputs,0,hc_row); REQUIRE(one);
    REQUIRE(!ds4_glm5_next_layer_forward(&x,il,&candidate.s,scalar,one,next_a));
    ds4_gpu_tensor_free(one);
    for (unsigned i=0;i<m;++i) {
        auto *out=ds4_gpu_tensor_view(serial,i*hc_row,hc_row); REQUIRE(out);
        serial_row(x,il,base,scalar,inputs,i,out);
        ds4_gpu_tensor_free(out);
    }
    equal("layer_output",serial,got,m*16384u);
    for (unsigned i=0;i<accepted;++i) serial_row(x,il,reference,scalar,inputs,i,next_a);
    REQUIRE(ds4_glm5_next_layer_verify_finish(&x,il,&candidate.s,accepted));
    REQUIRE(!ds4_glm5_next_layer_verify_finish(&x,il,&candidate.s,accepted));
    equal_layer(reference,candidate,il);
    for (unsigned i=0;i<2;++i) {
        serial_row(x,il,reference,scalar,inputs,m+3+i,next_a);
        serial_row(x,il,candidate,scalar,inputs,m+3+i,next_b);
        equal("continuation_output",next_a,next_b,16384);
        equal_layer(reference,candidate,il);
    }
    ds4_gpu_tensor_free(verify_inputs);
    REQUIRE(ds4_gpu_synchronize());
    ++cases;
    std::printf("LAYER_VERIFY layer=%u rank=%u m=%u prefix=%u accepted=%u simulated_peer=echo PASS\n",
        il,x.tp_rank,m,prefix,accepted); std::fflush(stdout);
}

static void refusal_and_failure(ds4_glm5_next_exec_ctx &x) {
    State state(*x.model);
    Workspace scalar(1), batch(4);
    Tensor input(4*hc_row), output(4*hc_row);
    REQUIRE(ds4_gpu_tensor_fill_f32(input,0.125f,4*16384));
    REQUIRE(!ds4_glm5_next_layer_verify_reserve(&x,0,&state.s,3));
    REQUIRE(ds4_glm5_next_layer_verify_reserve(&x,0,&state.s,4));
    x.tp->capable=false;
    const unsigned before=x.tp->calls;
    REQUIRE(!ds4_glm5_next_layer_verify(&x,0,&state.s,batch,scalar,input,output,4));
    REQUIRE(state.s.valid && x.tp->calls==before);
    x.tp->capable=true;
    const unsigned reserved_rank=x.tp_rank;
    x.tp_rank=x.tp->rank=1u-reserved_rank;
    REQUIRE(!ds4_glm5_next_layer_verify(&x,0,&state.s,batch,scalar,input,output,4));
    REQUIRE(state.s.valid && x.tp->calls==before);
    x.tp_rank=x.tp->rank=reserved_rank;
    for (unsigned fail=1;fail<=2;++fail) {
        x.tp->failed=false;
        REQUIRE(ds4_glm5_next_state_reset(&state.s));
        x.tp->fail_call=x.tp->calls+fail;
        REQUIRE(!ds4_glm5_next_layer_verify(&x,0,&state.s,batch,scalar,input,output,4));
        REQUIRE(!state.s.valid && !state.s.kda.pending_verifications &&
            !ds4_glm5_next_layer_verify_finish(&x,0,&state.s,1));
        x.tp->fail_call=0;
    }
    x.tp->failed=false;
}

static void serial_target_reserved(ds4_glm5_next_exec_ctx &x,State &s,Workspace &w,
                          unsigned token,ds4_gpu_tensor *scratch,
                          ds4_gpu_tensor *hidden,ds4_gpu_tensor *logits) {
    REQUIRE(ds4_glm5_next_embed_token(&x,token,scratch));
    ds4_gpu_tensor *in=scratch, *out=hidden;
    for (unsigned il=0;il<45;++il) {
        REQUIRE(ds4_glm5_next_layer_forward(&x,il,&s.s,w,in,out));
        auto *swap=in; in=out; out=swap;
    }
    REQUIRE(ds4_glm5_next_output_logits(&x,w,hidden,logits) && ds4_gpu_synchronize());
}

static void serial_target(ds4_glm5_next_exec_ctx &x,State &s,Workspace &w,
                          unsigned token,ds4_gpu_tensor *hidden,ds4_gpu_tensor *logits) {
    Tensor scratch(hc_row);
    serial_target_reserved(x,s,w,token,scratch,hidden,logits);
}

static void equal_target(State &a,State &b) {
    for (unsigned il=0;il<45;++il) equal_layer(a,b,il);
}

static void target_case(ds4_glm5_next_exec_ctx &x,unsigned m,unsigned prefix,unsigned accepted,
                        bool synthetic_frontier=false) {
    std::printf("TARGET_VERIFY begin rank=%u m=%u prefix=%u accepted=%u simulated_peer=echo synthetic_frontier=%u\n",
        x.tp_rank,m,prefix,accepted,synthetic_frontier); std::fflush(stdout);
    constexpr uint64_t logit_row=154880u*4u;
    const uint32_t tokens[8]={300,1234,57,902,341,765,88,42};
    State base(*x.model), reference(*x.model), candidate(*x.model);
    Workspace scalar(1), batch(m);
    Tensor scratch(m*hc_row), got(m*hc_row), logits(m*logit_row);
    Tensor serial(m*hc_row), serial_logits(m*logit_row);
    Tensor next_a(hc_row), next_b(hc_row), logit_a(logit_row), logit_b(logit_row);
    // Match even physically inactive tail entries without changing model data.
    for (unsigned il=3;il<45;il+=4) {
        const unsigned p=synthetic_frontier?prefix:0u;
        seed_mla(base,il,p); seed_mla(reference,il,p); seed_mla(candidate,il,p);
    }
    if (synthetic_frontier) for (unsigned il=0;il<45;++il) if (il%4!=3) {
        base.s.kda.layer[il].token_count=prefix;
        reference.s.kda.layer[il].token_count=prefix;
        candidate.s.kda.layer[il].token_count=prefix;
    }
    for (unsigned t=0;!synthetic_frontier && t<prefix;++t) {
        serial_target(x,base,scalar,991+t,next_a,logit_a);
        serial_target(x,reference,scalar,991+t,next_a,logit_a);
        serial_target(x,candidate,scalar,991+t,next_a,logit_a);
    }
    REQUIRE(ds4_glm5_next_target_verify_reserve(&x,&candidate.s,m));
    const auto frontier=candidate.s.kda.layer[0].token_count;
    const auto before=x.tp->calls;
    uint32_t bad[8]; std::memcpy(bad,tokens,sizeof(bad)); bad[m-1]=154880;
    REQUIRE(!ds4_glm5_next_target_verify(&x,&candidate.s,batch,scalar,bad,m,scratch,got,logits));
    REQUIRE(candidate.s.valid && x.tp->calls==before && !candidate.s.verification.tokens);
    candidate.s.kda.layer[44].token_count++;
    REQUIRE(!ds4_glm5_next_target_verify(&x,&candidate.s,batch,scalar,tokens,m,scratch,got,logits));
    candidate.s.kda.layer[44].token_count--;
    REQUIRE(x.tp->calls==before);
    auto *alias=ds4_gpu_tensor_view(scratch,0,m*hc_row); REQUIRE(alias);
    REQUIRE(!ds4_glm5_next_target_verify(&x,&candidate.s,batch,scalar,tokens,m,scratch,alias,logits));
    ds4_gpu_tensor_free(alias);
    const auto aux_before=x.tp->aux_calls;
    const auto agree_before=x.tp->layer_agrees[0];
    const auto compute_before=x.tp->layer_agrees[1];
    const auto handoff_before=x.tp->handoff_bulk_calls;
    REQUIRE(ds4_glm5_next_target_verify(&x,&candidate.s,batch,scalar,tokens,m,scratch,got,logits));
    const unsigned handoffs=(x.tp->prefill_config & DS4_TP_CONFIG_GLM5_VERIFY_FFN_HANDOFF)?31u:0u;
    REQUIRE(x.tp->layer_agrees[0]==agree_before+handoffs &&
        x.tp->layer_agrees[1]==compute_before+handoffs &&
        x.tp->handoff_bulk_calls==handoff_before+handoffs && !x.tp->handoff_pending);
    REQUIRE(candidate.s.verification.complete && candidate.s.verification.next_layer==45 &&
        candidate.s.kda.pending_verifications==34 && candidate.s.pending_mla_verifications==11 &&
        x.tp->aux_calls==aux_before);
    const auto sequence_after=*x.tp_sequence;
    REQUIRE(!std::memcmp(candidate.s.verification.input_tokens,tokens,m*4));
    equal_target(base,candidate);
    REQUIRE(!ds4_glm5_next_target_verify(&x,&candidate.s,batch,scalar,tokens,m,scratch,got,logits));
    REQUIRE(!ds4_glm5_next_layer_verify_finish(&x,0,&candidate.s,accepted));
    REQUIRE(!ds4_glm5_next_layer_forward(&x,0,&candidate.s,scalar,next_a,next_b));
    ds4_glm5_next_exec_ctx other=x;
    auto other_model=*x.model; other.model=&other_model;
    REQUIRE(!ds4_glm5_next_target_verify_finish(&other,&candidate.s,accepted));
    other=x; uint64_t other_sequence=*x.tp_sequence; other.tp_sequence=&other_sequence;
    REQUIRE(!ds4_glm5_next_target_verify_finish(&other,&candidate.s,accepted));
    other=x; other.tp_rank=1-x.tp_rank;
    REQUIRE(!ds4_glm5_next_target_verify_finish(&other,&candidate.s,accepted));
    // Mutate one binding field at a time on a real completed transaction.
    // Every refusal must preserve all live state, journals and transport calls.
    const auto finish_calls=x.tp->calls;
    const auto binding=candidate.s.verification;
    auto refused=[&](const ds4_glm5_next_exec_ctx &changed) {
        REQUIRE(!ds4_glm5_next_target_verify_finish(&changed,&candidate.s,accepted));
        REQUIRE(candidate.s.valid && x.tp->calls==finish_calls &&
            !std::memcmp(&binding,&candidate.s.verification,sizeof(binding)) &&
            candidate.s.kda.pending_verifications==34 &&
            candidate.s.pending_mla_verifications==11);
        equal_target(base,candidate);
    };
    other=x; other.model_map=(const unsigned char *)x.model_map+1; refused(other);
    other=x; --other.model_size; refused(other);
    ds4_tp other_peer=*x.tp; other=x; other.tp=&other_peer; refused(other);
    auto *slab_view=ds4_gpu_tensor_view(x.tp_slab,0,ds4_gpu_tensor_bytes(x.tp_slab));
    auto *out_view=ds4_gpu_tensor_view(x.tp_big_out,0,ds4_gpu_tensor_bytes(x.tp_big_out));
    auto *in_view=ds4_gpu_tensor_view(x.tp_big_in,0,ds4_gpu_tensor_bytes(x.tp_big_in));
    REQUIRE(slab_view && out_view && in_view);
    other=x; other.tp_slab=slab_view; refused(other);
    other=x; other.tp_big_out=out_view; refused(other);
    other=x; other.tp_big_in=in_view; refused(other);
    other=x; other.tp_big_out_host=(unsigned char *)x.tp_big_out_host+4; refused(other);
    other=x; other.tp_big_in_host=(unsigned char *)x.tp_big_in_host+4; refused(other);
    ds4_gpu_tensor_free(slab_view); ds4_gpu_tensor_free(out_view); ds4_gpu_tensor_free(in_view);
    ++*x.tp_sequence; refused(x); --*x.tp_sequence;
    const auto features=x.tp->features;
    x.tp->features^=DS4_TP_FEATURE_GLM5_SMALL_GATE; refused(x); x.tp->features=features;
    ++x.tp->prefill_config; refused(x); --x.tp->prefill_config;
    candidate.s.kda.layer[44].pending_tokens=1;
    REQUIRE(!ds4_glm5_next_target_verify_finish(&x,&candidate.s,accepted));
    candidate.s.kda.layer[44].pending_tokens=m;
    REQUIRE(candidate.s.valid && candidate.s.kda.layer[0].token_count==frontier);
    REQUIRE(ds4_glm5_next_target_verify_finish(&x,&candidate.s,accepted));
    REQUIRE(*x.tp_sequence==sequence_after && !candidate.s.verification.tokens);
    // The simulated transport is shared by these local independent states;
    // run references only after finish so they cannot interleave a TP pass.
    for (unsigned t=0;t<m;++t) {
        auto *h=ds4_gpu_tensor_view(serial,t*hc_row,hc_row);
        auto *l=ds4_gpu_tensor_view(serial_logits,t*logit_row,logit_row);
        REQUIRE(h && l); serial_target(x,base,scalar,tokens[t],h,l);
        ds4_gpu_tensor_free(h); ds4_gpu_tensor_free(l);
    }
    equal("target_hidden",serial,got,m*16384u);
    equal("target_logits",serial_logits,logits,m*154880u);
    for (unsigned t=0;t<accepted;++t)
        serial_target(x,reference,scalar,tokens[t],next_a,logit_a);
    equal_target(reference,candidate);
    for (unsigned t=0;t<2;++t) {
        serial_target(x,reference,scalar,789+t,next_a,logit_a);
        serial_target(x,candidate,scalar,789+t,next_b,logit_b);
        equal("target_continuation_hidden",next_a,next_b,16384);
        equal("target_continuation_logits",logit_a,logit_b,154880);
        equal_target(reference,candidate);
    }
    ++cases;
    std::printf("TARGET_VERIFY rank=%u m=%u prefix=%u accepted=%u PASS\n",
        x.tp_rank,m,prefix,accepted); std::fflush(stdout);
}

static void target_failure(ds4_glm5_next_exec_ctx &x) {
    State state(*x.model);
    Workspace scalar(1), batch(2);
    Tensor scratch(2*hc_row), hidden(2*hc_row), logits(2u*154880*4u);
    const uint32_t tokens[2]={300,1234};
    REQUIRE(ds4_glm5_next_target_verify_reserve(&x,&state.s,2));
    // Two attention exchanges and one FFN exchange per input: fail after
    // several journals have succeeded, including the first MLA layer.
    x.tp->fail_call=x.tp->calls+20;
    REQUIRE(!ds4_glm5_next_target_verify(&x,&state.s,batch,scalar,tokens,2,scratch,hidden,logits));
    REQUIRE(!state.s.valid && !state.s.verification.tokens && !state.s.kda.pending_verifications &&
        !state.s.pending_mla_verifications && !ds4_glm5_next_target_verify_finish(&x,&state.s,0));
    x.tp->fail_call=0; x.tp->failed=false;
    REQUIRE(ds4_glm5_next_state_reset(&state.s));
    std::puts("TARGET_VERIFY injected mid-pass exchange failure invalidates whole state PASS");
    const char *shared=std::getenv("DS4_ROCM_GLM5_VERIFY_SHARED_Q8");
    if (shared && !std::strcmp(shared,"1")) {
        REQUIRE(setenv("DS4_ROCM_GLM5_VERIFY_SHARED_Q8","invalid",1)==0);
        REQUIRE(!ds4_glm5_next_target_verify(&x,&state.s,batch,scalar,tokens,2,scratch,hidden,logits));
        REQUIRE(!state.s.valid && !state.s.verification.tokens &&
            !state.s.kda.pending_verifications && !state.s.pending_mla_verifications);
        REQUIRE(ds4_glm5_next_state_reset(&state.s));
        REQUIRE(setenv("DS4_ROCM_GLM5_VERIFY_SHARED_Q8","1",1)==0);
        const char *pair=std::getenv("DS4_ROCM_GLM5_SHARED_Q8_PAIR_DECODE");
        const bool had_pair=pair!=nullptr;
        const std::string pair_value=pair?pair:"";
        REQUIRE(setenv("DS4_ROCM_GLM5_SHARED_Q8_PAIR_DECODE","1",1)==0);
        REQUIRE(!ds4_glm5_next_target_verify(&x,&state.s,batch,scalar,tokens,2,scratch,hidden,logits));
        REQUIRE(!state.s.valid && !state.s.verification.tokens &&
            !state.s.kda.pending_verifications && !state.s.pending_mla_verifications);
        REQUIRE(ds4_glm5_next_state_reset(&state.s));
        REQUIRE(had_pair?setenv("DS4_ROCM_GLM5_SHARED_Q8_PAIR_DECODE",pair_value.c_str(),1)==0:
            unsetenv("DS4_ROCM_GLM5_SHARED_Q8_PAIR_DECODE")==0);
        std::puts("TARGET_VERIFY invalid shared selector and incompatible paired arithmetic refuse PASS");
    }
    // Each failure above represents a terminated session. A following case
    // uses a new logical echo peer; do not carry its poisoned link forward.
    REQUIRE(!x.tp->handoff_pending);
    x.tp->failed=false;
}

static void target_handoff_failure(ds4_glm5_next_exec_ctx &x) {
    State state(*x.model);
    Workspace scalar(1), batch(2);
    Tensor scratch(2*hc_row), hidden(2*hc_row), logits(2u*154880*4u);
    const uint32_t tokens[2]={300,1234};
    REQUIRE(ds4_glm5_next_target_verify_reserve(&x,&state.s,2));
    auto fails=[&]() {
        REQUIRE(!ds4_glm5_next_target_verify(&x,&state.s,batch,scalar,tokens,2,scratch,hidden,logits));
        REQUIRE(!state.s.valid && !state.s.verification.tokens &&
            !state.s.kda.pending_verifications && !state.s.pending_mla_verifications &&
            !ds4_glm5_next_target_verify_finish(&x,&state.s,1));
        ds4_tp_mark_failed(x.tp); x.tp->failed=false;
        REQUIRE(ds4_glm5_next_state_reset(&state.s));
    };
    REQUIRE(setenv("DS4_ROCM_GLM5_VERIFY_FFN_HANDOFF","invalid",1)==0); fails();
    REQUIRE(setenv("DS4_ROCM_GLM5_VERIFY_FFN_HANDOFF","1",1)==0);
    x.tp->prefill_config &= ~DS4_TP_CONFIG_GLM5_VERIFY_FFN_HANDOFF; fails();
    x.tp->prefill_config |= DS4_TP_CONFIG_GLM5_VERIFY_FFN_HANDOFF;
    REQUIRE(setenv("DS4_ROCM_GLM5_VERIFY_SHARED_Q8","0",1)==0); fails();
    REQUIRE(setenv("DS4_ROCM_GLM5_VERIFY_SHARED_Q8","1",1)==0);
    for (int phase=0;phase<2;++phase) {
        const auto before=x.tp->layer_agrees[phase];
        const auto bulk_before=x.tp->handoff_bulk_calls;
        x.tp->fail_agree_phase=phase; fails(); x.tp->fail_agree_phase=-1;
        REQUIRE(x.tp->layer_agrees[phase]==before+1 && x.tp->handoff_bulk_calls==bulk_before);
    }
    const auto bulk_before=x.tp->handoff_bulk_calls;
    x.tp->fail_handoff_bulk=true; fails(); x.tp->fail_handoff_bulk=false;
    REQUIRE(x.tp->handoff_bulk_calls==bulk_before+1);
    const char *overlap=std::getenv("DS4_ROCM_GLM5_SHARED_ROUTE_OVERLAP");
    const bool had_overlap=overlap!=nullptr;
    const std::string overlap_value=overlap?overlap:"";
    const auto before=x.tp->layer_agrees[0];
    const auto compute_before=x.tp->layer_agrees[1];
    REQUIRE(ds4_glm5_next_layer_verify_reserve(&x,4,&state.s,2));
    REQUIRE(ds4_gpu_tensor_fill_f32(scratch,0.125f,2*16384u));
    REQUIRE(setenv("DS4_ROCM_GLM5_SHARED_ROUTE_OVERLAP","invalid",1)==0);
    REQUIRE(!ds4_glm5_next_layer_verify(&x,4,&state.s,batch,scalar,scratch,hidden,2));
    REQUIRE(!state.s.valid && !state.s.kda.pending_verifications && x.tp->failed);
    x.tp->failed=false;
    REQUIRE(x.tp->layer_agrees[0]==before+1 && x.tp->layer_agrees[1]==compute_before);
    REQUIRE((had_overlap?setenv("DS4_ROCM_GLM5_SHARED_ROUTE_OVERLAP",overlap_value.c_str(),1):
        unsetenv("DS4_ROCM_GLM5_SHARED_ROUTE_OVERLAP"))==0);
    std::puts("TARGET_HANDOFF invalid selector/hello/shared, local failure, both agreement and bulk failures refuse PASS");
}

static void target_timing(ds4_glm5_next_exec_ctx &x,unsigned m,bool heads_only=false,
                         const char *switch_name="DS4_ROCM_GLM5_BF16_VERIFY_HEAD") {
    const bool dense_compare=!std::strcmp(switch_name,"DS4_ROCM_GLM5_VERIFY_DENSE_Q8");
    const bool shared_compare=!std::strcmp(switch_name,"DS4_ROCM_GLM5_VERIFY_SHARED_Q8");
    const bool handoff_compare=!std::strcmp(switch_name,"DS4_ROCM_GLM5_VERIFY_FFN_HANDOFF");
    const uint64_t original_config=x.tp->prefill_config;
    const char *tag=heads_only?(handoff_compare?"HANDOFF_BUDGET":shared_compare?"SHARED_Q8_BUDGET":
        dense_compare?"DENSE_Q8_BUDGET":"HEAD_BUDGET"):"TARGET_BUDGET";
    const char *old_option=std::getenv(switch_name);
    const bool had_option=old_option!=nullptr;
    const std::string old_value=old_option?old_option:"";
    std::printf("%s begin rank=%u m=%u simulated_peer=echo\n",
        tag,x.tp_rank,m);
    std::fflush(stdout);
    constexpr uint64_t logit_row=154880u*4u;
    const uint32_t tokens[8]={300,1234,57,902,341,765,88,42};
    State serial(*x.model), candidate(*x.model);
    Workspace scalar(1), batch(m);
    Tensor scratch(hc_row), hidden(hc_row), logits(logit_row);
    Tensor head_candidate(logit_row);
    Tensor prefix_hidden(hc_row), prefix_logits(logit_row);
    Tensor batch_scratch(m*hc_row), batch_hidden(m*hc_row), batch_logits(m*logit_row);
    REQUIRE(ds4_glm5_next_target_verify_reserve(&x,&candidate.s,m));
    if (heads_only) REQUIRE(ds4_glm5_next_target_verify_reserve(&x,&serial.s,m));
    for (unsigned il=3;il<45;il+=4) { seed_mla(serial,il,0); seed_mla(candidate,il,0); }
    std::vector<double> times[2];
    for (unsigned sample=0;sample<13;++sample) for (unsigned turn=0;turn<2;++turn) {
        const unsigned arm=turn^(sample&1u);
        State &state=arm?candidate:serial;
        REQUIRE(ds4_glm5_next_state_reset(&state.s));
        for (unsigned t=0;t<3;++t)
            serial_target_reserved(x,state,scalar,991+t,scratch,prefix_hidden,prefix_logits);
        REQUIRE(ds4_gpu_synchronize());
        if (heads_only)
            REQUIRE(setenv(switch_name,arm?"1":"0",1)==0);
        if (handoff_compare) x.tp->prefill_config =
            (original_config & ~DS4_TP_CONFIG_GLM5_VERIFY_FFN_HANDOFF) |
            (arm?DS4_TP_CONFIG_GLM5_VERIFY_FFN_HANDOFF:0u);
        const auto start=std::chrono::steady_clock::now();
        if (arm || heads_only) {
            REQUIRE(ds4_glm5_next_target_verify(&x,&state.s,batch,scalar,tokens,m,
                batch_scratch,batch_hidden,batch_logits));
            REQUIRE(ds4_glm5_next_target_verify_finish(&x,&state.s,m));
        } else for (unsigned t=0;t<m;++t)
            serial_target_reserved(x,state,scalar,tokens[t],scratch,hidden,logits);
        REQUIRE(ds4_gpu_synchronize());
        const double ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-start).count();
        if (heads_only) REQUIRE(ds4_gpu_tensor_copy(arm?head_candidate.p:logits.p,0,
            batch_logits,(m-1)*logit_row,logit_row));
        if (sample>=4) {
            times[arm].push_back(ms);
            std::printf("%s_SAMPLE rank=%u m=%u round=%u arm=%s ms=%.6f\n",
                tag,x.tp_rank,m,sample-4,
                arm?"verify_commit":heads_only?(handoff_compare?"scalar_handoff_verify":shared_compare?"scalar_shared_verify":
                    dense_compare?"scalar_dense_verify":"scalar_head_verify"):"serial",ms);
            std::fflush(stdout);
        }
    }
    equal_target(serial,candidate);
    auto *last=heads_only?ds4_gpu_tensor_view(head_candidate,0,logit_row):
        ds4_gpu_tensor_view(batch_logits,(m-1)*logit_row,logit_row);
    REQUIRE(last); equal("timed_final_logits",logits,last,154880); ds4_gpu_tensor_free(last);
    for (auto &v:times) std::sort(v.begin(),v.end());
    std::printf("%s rank=%u m=%u samples=9 control_median_ms=%.6f "
        "verify_commit_median_ms=%.6f simulated_peer=echo network_test=0 drafting_test=0\n",
        tag,x.tp_rank,m,times[0][4],times[1][4]);
    std::fflush(stdout);
    if (heads_only) REQUIRE((had_option?setenv(switch_name,old_value.c_str(),1):unsetenv(switch_name))==0);
    x.tp->prefill_config=original_config;
}

static void target_profile(ds4_glm5_next_exec_ctx &x) {
    State state(*x.model);
    Workspace scalar(1), batch(8);
    Tensor scratch(hc_row), hidden(hc_row), logits(154880u*4u);
    Tensor batch_scratch(8*hc_row), batch_hidden(8*hc_row), batch_logits(8u*154880u*4u);
    const uint32_t tokens[8]={300,1234,57,902,341,765,88,42};
    REQUIRE(ds4_glm5_next_target_verify_reserve(&x,&state.s,8));
    for (unsigned il=3;il<45;il+=4) seed_mla(state,il,0);
    for (unsigned pass=0;pass<3;++pass) {
        REQUIRE(ds4_glm5_next_state_reset(&state.s));
        for (unsigned t=0;t<3;++t)
            serial_target_reserved(x,state,scalar,991+t,scratch,hidden,logits);
        REQUIRE(ds4_gpu_synchronize());
        if (pass==2) REQUIRE(setenv("DS4_GLM5_VERIFY_PROFILE","1",1)==0);
        REQUIRE(ds4_glm5_next_target_verify(&x,&state.s,batch,scalar,tokens,8,
            batch_scratch,batch_hidden,batch_logits));
        REQUIRE(ds4_glm5_next_target_verify_finish(&x,&state.s,8));
        REQUIRE(unsetenv("DS4_GLM5_VERIFY_PROFILE")==0);
    }
    std::printf("VERIFY_PROFILE_DONE rank=%u simulated_peer=echo network_test=0 quality_test=0\n",x.tp_rank);
}

#include "glm5_native_draft_checks.hpp"

int main(int argc,char **argv) {
    const bool refresh=argc==3 && !std::strcmp(argv[1],"--native-refresh");
    const bool warm=argc==3 && !std::strcmp(argv[1],"--native-warm");
    const bool native=refresh || warm || (argc==3 && !std::strcmp(argv[1],"--native-draft"));
    const bool dense_compare=argc==3 && !std::strcmp(argv[1],"--target-resident-dense-both");
    const bool shared_compare=argc==3 && !std::strcmp(argv[1],"--target-resident-shared-both");
    const bool handoff_compare=argc==3 && !std::strcmp(argv[1],"--target-resident-handoff-both");
    const bool resident_both=dense_compare || shared_compare || handoff_compare || (argc==3 && !std::strcmp(argv[1],"--target-resident-both"));
    const bool profile=argc==3 && !std::strcmp(argv[1],"--target-resident-profile");
    const bool resident=argc==3 && (!std::strcmp(argv[1],"--target-resident") ||
        !std::strcmp(argv[1],"--target-resident-timing") || resident_both || profile || native);
    REQUIRE(!resident || !std::strcmp(argv[2],"0") || !std::strcmp(argv[2],"1"));
    const unsigned resident_rank=resident && !std::strcmp(argv[2],"1")?1u:0u;
    const bool timing=(argc==2 && !std::strcmp(argv[1],"--target-timing")) ||
        (resident && !std::strcmp(argv[1],"--target-resident-timing"));
    const bool target=resident || (argc==2 && (!std::strcmp(argv[1],"--target") ||
        !std::strcmp(argv[1],"--target-smoke")));
    const bool smoke=argc==2 && (!std::strcmp(argv[1],"--smoke") ||
        !std::strcmp(argv[1],"--target-smoke"));
    REQUIRE(argc==1 || target || smoke || timing);
    const char *path=std::getenv("DS4_GLM5_MODEL");
    REQUIRE(path && std::getenv("DS4_RESEARCH_ROOT"));
    setenv("DS4_ROCM_GLM5_BF16_SMALL_M_EXACT","1",1);
    setenv("DS4_GLM5_NEXT_ENABLE_ORDINARY","1",1);
    Glm5TestGGUF g; REQUIRE(g.open_file(path));
    ds4_glm5_next_model_offsets model={}; REQUIRE(glm5_next_bind_real_offsets(g,model));
    if (dense_compare) {
        REQUIRE(std::getenv("DS4_ROCM_GLM5_VERIFY_DENSE_Q8") &&
            !std::strcmp(std::getenv("DS4_ROCM_GLM5_VERIFY_DENSE_Q8"),"1"));
        REQUIRE(std::getenv("DS4_ROCM_GLM5_BF16_VERIFY_HEAD") &&
            !std::strcmp(std::getenv("DS4_ROCM_GLM5_BF16_VERIFY_HEAD"),"1") && model.output_type==30u);
        std::puts("TARGET_SETTINGS dense_q8=1 bf16_head=1 model_head_type=30");
    }
    if (shared_compare || handoff_compare) {
        REQUIRE(std::getenv("DS4_ROCM_GLM5_VERIFY_SHARED_Q8") &&
            !std::strcmp(std::getenv("DS4_ROCM_GLM5_VERIFY_SHARED_Q8"),"1"));
        std::puts("TARGET_SETTINGS shared_q8=1 scalar_prefix=1 weights=original");
    }
    if (handoff_compare) REQUIRE(std::getenv("DS4_ROCM_GLM5_VERIFY_FFN_HANDOFF") &&
        !std::strcmp(std::getenv("DS4_ROCM_GLM5_VERIFY_FFN_HANDOFF"),"1"));
    REQUIRE(ds4_gpu_init() && ds4_gpu_set_model_fd_for_map(g.fd,g.map));
    if (!resident) REQUIRE(ds4_gpu_set_model_map(g.map,g.size));
    {
        Tensor slab(65536,true), out(8*4096*4,true), in(8*4096*4,true);
        ds4_tp peer; peer.slab=(unsigned char *)ds4_gpu_tensor_contents(slab);
        if (handoff_compare) peer.prefill_config |= DS4_TP_CONFIG_GLM5_VERIFY_FFN_HANDOFF;
        if (resident) {
            Glm5NextKShardPlan plan;
            REQUIRE(glm5_next_build_kshard_plan(g,model,plan,native));
            uint64_t free_bytes=0,total_bytes=0;
            REQUIRE(ds4_gpu_memory_info(&free_bytes,&total_bytes) &&
                plan.dense_total_bytes+plan.packed_total_bytes+(UINT64_C(3)<<30)<free_bytes);
            ds4_gpu_set_glm_model(true);
            ds4_gpu_set_q8_cache_suppressed(1);
            peer.rank=resident_rank;
            peer.features|=DS4_TP_FEATURE_Q4K_KSHARD | DS4_TP_FEATURE_Q4K_WMMA;
            ds4_gpu_set_tp_runtime_features(peer.rank,peer.features);
            REQUIRE(ds4_gpu_q4k_kshard_install(g.map,g.size,g.fd,peer.rank,
                plan.dense_offsets.data(),plan.dense_sizes.data(),plan.dense_offsets.size(),
                plan.dense_max_tensor_bytes,plan.layers.data(),plan.layers.size()));
            ds4_gpu_q4k_kshard_windows windows={};
            REQUIRE(ds4_gpu_q4k_kshard_windows_get(&windows) && windows.rank==peer.rank &&
                windows.n_layers==42 && windows.row_count==1024 &&
                windows.down_column_byte_count==576 &&
                ds4_gpu_q4k_packed_slice_bytes()==plan.packed_total_bytes);
            std::printf("TARGET_RESIDENCY rank=%u dense_bytes=%llu packed_bytes=%llu "
                "weights=original_q4k simulated_peer=echo\n",peer.rank,
                (unsigned long long)plan.dense_total_bytes,
                (unsigned long long)plan.packed_total_bytes); std::fflush(stdout);
        }
        uint64_t sequence=0;
        ds4_glm5_next_exec_ctx x={};
        x.model=&model; x.model_map=g.map; x.model_size=g.size; x.tp=&peer;
        x.tp_slab=slab; x.tp_big_out=out; x.tp_big_in=in;
        x.tp_big_out_host=ds4_gpu_tensor_contents(out); x.tp_big_in_host=ds4_gpu_tensor_contents(in);
        x.tp_sequence=&sequence;
        const unsigned first_rank=resident?resident_rank:0u;
        const unsigned end_rank=resident?resident_rank+1u:2u;
        if (native) {
            peer.rank=x.tp_rank=resident_rank;
            if (refresh) { native_hidden_tile_checks(x); native_refresh_checks(x); }
            if (warm) native_warm_checks(x);
            if (!refresh) native_draft_checks(x);
        } else if (profile) {
            peer.rank=x.tp_rank=resident_rank;
            target_profile(x);
        } else if (timing) {
            for (unsigned rank=first_rank;rank<end_rank;++rank) {
                peer.rank=x.tp_rank=rank;
                for (unsigned m : glm5_test_verifier_widths()) target_timing(x,m);
            }
        } else if (target) {
            for (unsigned rank=first_rank;rank<(smoke?1u:end_rank);++rank) {
                peer.rank=x.tp_rank=rank;
                if (smoke) target_case(x,2,0,1);
                else for (unsigned m : glm5_test_verifier_widths())
                    for (unsigned accepted=0;accepted<=m;++accepted)
                        target_case(x,m,m==2?0u:3u,accepted);
                target_failure(x);
                if (handoff_compare) {
                    for (unsigned m : glm5_test_verifier_widths()) target_case(x,m,8192u+(m==4),m/2,true);
                    target_handoff_failure(x);
                }
                if (resident_both && !shared_compare && !handoff_compare) for (unsigned m : glm5_test_verifier_widths()) target_timing(x,m);
                if (resident_both) for (unsigned m : glm5_test_verifier_widths()) target_timing(x,m,true,
                    handoff_compare?"DS4_ROCM_GLM5_VERIFY_FFN_HANDOFF":
                    shared_compare?"DS4_ROCM_GLM5_VERIFY_SHARED_Q8":
                    dense_compare?"DS4_ROCM_GLM5_VERIFY_DENSE_Q8":"DS4_ROCM_GLM5_BF16_VERIFY_HEAD");
            }
        } else if (smoke) run_case(x,0,4,3,2);
        else {
            for (unsigned rank=0;rank<2;++rank) {
                peer.rank=x.tp_rank=rank;
                for (unsigned il : {0u,3u,44u}) for (unsigned m : glm5_test_verifier_widths())
                    for (unsigned prefix : {0u,3u}) for (unsigned accepted=0;accepted<=m;++accepted)
                        run_case(x,il,m,prefix,accepted);
                for (unsigned prefix : {2047u,2048u,8192u,8193u})
                    for (unsigned accepted=0;accepted<=4;++accepted)
                        run_case(x,3,4,prefix,accepted);
            }
        }
        if (!target && !timing) refusal_and_failure(x);
        if (handoff_compare) std::printf("HANDOFF_COUNTS rank=%u routes=%u compute=%u bulk=%u simulated_peer=echo\n",
            peer.rank,peer.layer_agrees[0],peer.layer_agrees[1],peer.handoff_bulk_calls);
    }
    std::printf("PASS verification cases=%u compared_float_values=%llu simulated_peer=echo "
        "network_test=0 full_target_test=%u quality_test=0 timing_test=%u\n",
        cases,(unsigned long long)compared_values,refresh || (!native && (target||timing))?1:0,
        timing||resident_both||(native&&!refresh)?1:0);
    if (resident) ds4_gpu_q4k_kshard_release();
    ds4_gpu_cleanup();
    return 0;
}
