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

#define REQUIRE(x) do { if (!(x)) { \
    std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); std::exit(1); \
} } while (0)

struct ds4_tp {
    unsigned rank = 0;
    unsigned char *slab = nullptr;
    bool capable = true, failed = false;
    unsigned calls = 0, fail_call = 0, bulk_calls = 0, aux_calls = 0;
};
extern "C" {
void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}
int ds4_tp_rank(const ds4_tp *p) { return p->rank; }
bool ds4_tp_is_rdma(const ds4_tp *p) { return p->capable; }
bool ds4_tp_big_gate_is_rdma_capable(const ds4_tp *p) { return p->capable; }
bool ds4_tp_big_gate_is_direct(const ds4_tp *p,const void *,const void *,uint64_t) { return p->capable; }
uint32_t ds4_tp_runtime_features(const ds4_tp *) {
    return DS4_TP_FEATURE_GLM5_KDA_TP | DS4_TP_FEATURE_GLM5_KDA_OUTPUT_ROWSLICE |
        DS4_TP_FEATURE_GLM5_SMALL_GATE;
}
uint64_t ds4_tp_prefill_config(const ds4_tp *) { return 0; }
uint64_t ds4_tp_vec_bytes(const ds4_tp *) { return 16384; }
uint64_t ds4_tp_aux_payload_bytes(const ds4_tp *) { return 8192; }
uint64_t ds4_tp_slab_out_offset(const ds4_tp *,uint32_t,uint32_t) { return 0; }
uint64_t ds4_tp_slab_in_offset(const ds4_tp *,uint32_t,uint32_t) { return 16384; }
uint64_t ds4_tp_slab_aux_out_payload_offset(const ds4_tp *,uint32_t) { return 32768; }
uint64_t ds4_tp_slab_aux_in_payload_offset(const ds4_tp *,uint32_t) { return 40960; }
void ds4_tp_mark_failed(ds4_tp *p) { p->failed=true; }
int ds4_tp_hash_check(ds4_tp *,uint64_t,uint64_t,char *,size_t) { return 1; }
int ds4_tp_gate_exchange(ds4_tp *p,uint32_t,uint32_t,uint64_t) {
    if (++p->calls==p->fail_call) return 0;
    std::memcpy(p->slab+16384,p->slab,16384); return 1;
}
int ds4_tp_gate_exchange_from_registered(ds4_tp *p,uint32_t,uint32_t,uint64_t,const void *out) {
    if (++p->calls==p->fail_call) return 0;
    std::memcpy(p->slab+16384,out,16384); return 1;
}
int ds4_tp_aux_gate_exchange(ds4_tp *p,uint32_t) {
    ++p->aux_calls;
    if (++p->calls==p->fail_call) return 0;
    std::memcpy(p->slab+40960,p->slab+32768,8192); return 1;
}
int ds4_tp_big_gate_exchange(ds4_tp *p,uint32_t,uint64_t,const void *out,void *in,uint64_t bytes) {
    ++p->bulk_calls;
    if (++p->calls==p->fail_call) return 0;
    std::memcpy(in,out,bytes); return 1;
}
}

static constexpr uint64_t hc_row = 16384u*4u;
static constexpr unsigned context = 8216;
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
    explicit State(const ds4_glm5_next_model_offsets &m) {
        FILE *quiet=std::fopen("/dev/null","w"); REQUIRE(quiet);
        REQUIRE(ds4_glm5_next_state_init(&s,&m,context,quiet));
        std::fclose(quiet);
    }
    ~State() { ds4_glm5_next_state_free(&s); }
};
struct Workspace {
    ds4_glm5_next_workspace *p;
    explicit Workspace(unsigned m) : p(ds4_glm5_next_workspace_create_capacity_context(m,context)) {
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
    if (il%4!=3) {
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

int main(int argc,char **argv) {
    REQUIRE(argc==1 || (argc==2 && std::strcmp(argv[1],"--smoke")==0));
    const bool smoke=argc==2;
    const char *path=std::getenv("DS4_GLM5_MODEL");
    REQUIRE(path && std::getenv("DS4_RESEARCH_ROOT"));
    setenv("DS4_ROCM_GLM5_BF16_SMALL_M_EXACT","1",1);
    setenv("DS4_GLM5_NEXT_ENABLE_ORDINARY","1",1);
    Glm5TestGGUF g; REQUIRE(g.open_file(path));
    ds4_glm5_next_model_offsets model={}; REQUIRE(glm5_next_bind_real_offsets(g,model));
    REQUIRE(ds4_gpu_init() && ds4_gpu_set_model_fd_for_map(g.fd,g.map) && ds4_gpu_set_model_map(g.map,g.size));
    {
        Tensor slab(65536,true), out(8*4096*4,true), in(8*4096*4,true);
        ds4_tp peer; peer.slab=(unsigned char *)ds4_gpu_tensor_contents(slab);
        uint64_t sequence=0;
        ds4_glm5_next_exec_ctx x={};
        x.model=&model; x.model_map=g.map; x.model_size=g.size; x.tp=&peer;
        x.tp_slab=slab; x.tp_big_out=out; x.tp_big_in=in;
        x.tp_big_out_host=ds4_gpu_tensor_contents(out); x.tp_big_in_host=ds4_gpu_tensor_contents(in);
        x.tp_sequence=&sequence;
        if (smoke) run_case(x,0,4,3,2);
        else {
            for (unsigned rank=0;rank<2;++rank) {
                peer.rank=x.tp_rank=rank;
                for (unsigned il : {0u,3u,44u}) for (unsigned m : {2u,4u,8u})
                    for (unsigned prefix : {0u,3u}) for (unsigned accepted=0;accepted<=m;++accepted)
                        run_case(x,il,m,prefix,accepted);
                for (unsigned prefix : {2047u,2048u,8192u,8193u})
                    for (unsigned accepted=0;accepted<=4;++accepted)
                        run_case(x,3,4,prefix,accepted);
            }
        }
        refusal_and_failure(x);
    }
    std::printf("PASS target-layer verification cases=%u compared_float_values=%llu simulated_peer=echo "
        "network_test=0 full_target_test=0 quality_test=0\n",cases,(unsigned long long)compared_values);
    ds4_gpu_cleanup();
    return 0;
}
