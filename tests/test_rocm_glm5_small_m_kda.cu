// Real-weight differential test of the production head-local KDA boundary.
#include "ds4_glm5_kda.h"
#include "ds4_gpu_mgpu.h"
extern "C" {
#include "ds4_tp.h"
}
#include "glm5_gguf_test.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}
#define REQUIRE(x) do { if (!(x)) { \
    std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); std::exit(1); \
} } while (0)

static ds4_glm5_kda_weight_offsets bind(const Glm5TestGGUF &g, unsigned il) {
    ds4_glm5_kda_weight_offsets w = {};
    auto get = [&](const char *role, std::initializer_list<uint64_t> dims,
                   unsigned type, uint64_t &offset) {
        char name[96];
        std::snprintf(name,sizeof(name),"blk.%u.%s.weight",il,role);
        REQUIRE(g.tensor(name,dims,type,offset));
    };
    get("attn_norm",{4096},0,w.attn_norm);
    get("kda_q",{4096,8192},30,w.q);
    get("kda_k",{4096,8192},30,w.k);
    get("kda_v",{4096,8192},30,w.v);
    get("kda_output",{8192,4096},30,w.output);
    get("kda_q_conv",{4,1,8192},0,w.q_conv);
    get("kda_k_conv",{4,1,8192},0,w.k_conv);
    get("kda_v_conv",{4,1,8192},0,w.v_conv);
    get("kda_f_a",{4096,128},30,w.f_a);
    get("kda_f_b",{128,8192},30,w.f_b);
    get("kda_g_a",{4096,128},30,w.g_a);
    get("kda_g_b",{128,8192},30,w.g_b);
    get("kda_beta",{4096,64},30,w.beta);
    get("kda_o_norm",{128},0,w.o_norm);
    get("kda_dt_bias",{8192},0,w.dt_bias);
    get("kda_a_log",{64},0,w.a_log);
    w.q_type=w.k_type=w.v_type=w.output_type=30;
    w.f_a_type=w.f_b_type=w.g_a_type=w.g_b_type=w.beta_type=30;
    return w;
}

struct HeadState {
    ds4_glm5_kda_slot slot = {};
    ds4_glm5_kda_layer_state local = {};
    explicit HeadState(unsigned rank) {
        const ds4_glm5_layer_kind schedule = {0,true};
        REQUIRE(ds4_glm5_kda_slot_init(&slot,&schedule,1,1,nullptr));
        const uint64_t history = 4096u*3u*4u, recurrent = 32u*128u*128u*4u;
        local.q_history=ds4_gpu_tensor_view(slot.layer[0].q_history,rank*history,history);
        local.k_history=ds4_gpu_tensor_view(slot.layer[0].k_history,rank*history,history);
        local.v_history=ds4_gpu_tensor_view(slot.layer[0].v_history,rank*history,history);
        local.recurrent=ds4_gpu_tensor_view(slot.layer[0].recurrent,rank*recurrent,recurrent);
        local.valid=true;
        local.owner_slot=&slot;
        REQUIRE(local.q_history && local.k_history && local.v_history && local.recurrent);
    }
    ~HeadState() {
        ds4_gpu_tensor_free(local.q_history); ds4_gpu_tensor_free(local.k_history);
        ds4_gpu_tensor_free(local.v_history); ds4_gpu_tensor_free(local.recurrent);
        ds4_glm5_kda_slot_free(&slot);
    }
};

static std::vector<float> read(const ds4_gpu_tensor *t, uint64_t count) {
    std::vector<float> v(count);
    REQUIRE(ds4_gpu_tensor_read(t,0,v.data(),count*4u));
    for (float x : v) REQUIRE(std::isfinite(x));
    return v;
}

struct Stage {
    const char *name;
    ds4_gpu_tensor *ds4_glm5_kda_workspace::*member;
    unsigned width;
};
// f_low and forget are reused: their final contents are g_a and g_b.
static const Stage stages[] = {
    {"norm",&ds4_glm5_kda_workspace::norm,4096},
    {"q",&ds4_glm5_kda_workspace::q,4096},
    {"k",&ds4_glm5_kda_workspace::k,4096},
    {"v",&ds4_glm5_kda_workspace::v,4096},
    {"g_a",&ds4_glm5_kda_workspace::f_low,128},
    {"g_b",&ds4_glm5_kda_workspace::forget,4096},
    {"beta",&ds4_glm5_kda_workspace::beta,32},
};

static uint64_t compare(const std::vector<float> &ref,
                        const std::vector<float> &got, const char *stage,
                        unsigned il, unsigned rank, unsigned m,
                        unsigned prefix, unsigned arm) {
    REQUIRE(ref.size()==got.size());
    uint64_t different=0;
    double max_abs=0;
    for (size_t i=0;i<ref.size();++i) {
        different+=std::memcmp(&ref[i],&got[i],4u)!=0;
        max_abs=std::max(max_abs,std::fabs((double)ref[i]-got[i]));
    }
    std::printf("SMALL_M_KDA stage=%s layer=%u rank=%u m=%u prefix=%u arm=%u values=%zu different=%llu max_abs=%.9g\n",
        stage,il,rank,m,prefix,arm,ref.size(),(unsigned long long)different,max_abs);
    return different;
}

static void time_kda(const Glm5TestGGUF &g,
                     const ds4_glm5_kda_weight_offsets &weights) {
    // Local warm-weight stage budget, not a target-verifier throughput test.
    for (unsigned rank=0;rank<2;++rank) for (unsigned m : {2u,4u,8u}) {
        HeadState state(rank);
        ds4_glm5_kda_workspace scalar = {}, batch = {};
        REQUIRE(ds4_glm5_kda_workspace_init(&scalar,1));
        REQUIRE(ds4_glm5_kda_workspace_init(&batch,m));
        std::vector<float> host((7u+m)*4096u);
        for (size_t i=0;i<host.size();++i)
            host[i]=(float)((int)((i*193u+(i/4096u)*761u)%997u)-498)/1001.3f;
        auto *input=ds4_gpu_tensor_alloc(host.size()*4u);
        auto *output=ds4_gpu_tensor_alloc((uint64_t)m*4096u*4u);
        REQUIRE(input && output && ds4_gpu_tensor_write(input,0,host.data(),host.size()*4u));
        std::vector<ds4_gpu_tensor *> rows;
        for (unsigned t=0;t<7u+m;++t) {
            rows.push_back(ds4_gpu_tensor_view(input,(uint64_t)t*4096u*4u,4096u*4u));
            REQUIRE(rows.back());
        }
        auto *group=ds4_gpu_tensor_view(input,7u*4096u*4u,(uint64_t)m*4096u*4u);
        REQUIRE(group);
        auto run=[&](const ds4_gpu_tensor *x, unsigned count,
                     ds4_glm5_kda_workspace &ws) {
            REQUIRE(ds4_glm5_kda_layer_begin(&state.local,&ws,&weights,g.map,
                g.size,x,output,count,1.0e-5f,rank*32u,32u));
            REQUIRE(ds4_glm5_kda_layer_commit(&state.local,count));
        };
        hipEvent_t start,end;
        REQUIRE(hipEventCreate(&start)==hipSuccess && hipEventCreate(&end)==hipSuccess);
        std::vector<double> samples[3];
        for (unsigned round=0;round<12;++round) for (unsigned j=0;j<3;++j) {
            const unsigned arm=(round+j)%3u;
            REQUIRE(setenv("DS4_ROCM_GLM5_BF16_SMALL_M_EXACT",arm==2?"1":"0",1)==0);
            REQUIRE(ds4_glm5_kda_slot_reset(&state.slot));
            state.local.token_count=0;
            state.local.pending_tokens=0;
            state.local.valid=true;
            for (unsigned t=0;t<7;++t) run(rows[t],1,scalar);
            REQUIRE(hipEventRecord(start,nullptr)==hipSuccess);
            if (arm==0) for (unsigned t=0;t<m;++t) run(rows[7+t],1,scalar);
            else run(group,m,batch);
            REQUIRE(hipEventRecord(end,nullptr)==hipSuccess && hipEventSynchronize(end)==hipSuccess);
            float ms=0;
            REQUIRE(hipEventElapsedTime(&ms,start,end)==hipSuccess && std::isfinite(ms) && ms>0);
            if (round>=3) {
                samples[arm].push_back(ms);
                std::printf("SMALL_M_KDA_SAMPLE rank=%u m=%u arm=%u round=%u ms=%.6f\n",
                    rank,m,arm,round-3,ms);
            }
        }
        for (unsigned arm=0;arm<3;++arm) {
            std::sort(samples[arm].begin(),samples[arm].end());
            std::printf("SMALL_M_KDA_MEDIAN rank=%u m=%u arm=%u ms=%.6f\n",
                rank,m,arm,samples[arm][4]);
        }
        REQUIRE(hipEventDestroy(start)==hipSuccess && hipEventDestroy(end)==hipSuccess);
        ds4_gpu_tensor_free(group);
        for (auto *row : rows) ds4_gpu_tensor_free(row);
        ds4_gpu_tensor_free(output); ds4_gpu_tensor_free(input);
        ds4_glm5_kda_workspace_free(&batch); ds4_glm5_kda_workspace_free(&scalar);
    }
}

int main(int argc, char **argv) {
    const bool diagnostic=argc==2 && std::strcmp(argv[1],"--diagnostic")==0;
    const bool timing=argc==2 && std::strcmp(argv[1],"--timing")==0;
    REQUIRE(argc==1 || diagnostic || timing);
    const char *path=std::getenv("DS4_GLM5_MODEL");
    REQUIRE(path);
    Glm5TestGGUF g;
    REQUIRE(g.open_file(path));
    REQUIRE(g.metadata_f32.at("glm5-next.attention.layer_norm_rms_epsilon")==1.0e-5f);
    // Freeze the ordinary decode projection controls before their first use.
    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_QKV_PREFETCH","64",1)==0);
    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_KDA_SIX_DECODE_MULTIPTR","0",1)==0);
    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_KDA_SIX_PREFILL","0",1)==0);
    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_QKV_DECODE_MULTIPTR","1",1)==0);
    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_WMMA_HILO","1",1)==0);
    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_WMMA_QKV_FUSED","1",1)==0);
    ds4_gpu_config config = {};
    config.n_gpus=1;
    REQUIRE(ds4_gpu_init_multi(&config));
    REQUIRE(ds4_gpu_set_model_fd_for_map(g.fd,g.map));
    REQUIRE(ds4_gpu_set_model_map(g.map,g.size));
    uint64_t candidate_different=0, candidate_values=0;
    for (unsigned il : {0u,1u,44u}) {
        const auto weights=bind(g,il);
        for (unsigned rank=0;rank<2;++rank) for (unsigned m : {2u,4u,8u})
        for (unsigned prefix : {0u,3u,7u}) {
            std::vector<float> host((prefix+m+1u)*4096u);
            for (size_t i=0;i<host.size();++i)
                host[i]=(float)((int)((i*193u+(i/4096u)*761u+il*47u)%997u)-498)/
                    (1001.3f+(float)(i%7u));
            auto *input=ds4_gpu_tensor_alloc(host.size()*4u);
            auto *output=ds4_gpu_tensor_alloc((m+1u)*4096u*4u);
            REQUIRE(input && output && ds4_gpu_tensor_write(input,0,host.data(),host.size()*4u));
            std::vector<std::vector<float>> reference;
            for (unsigned arm=0;arm<3;++arm) {
                REQUIRE(setenv("DS4_ROCM_GLM5_BF16_SMALL_M_EXACT",arm==2?"1":"0",1)==0);
                HeadState state(rank);
                ds4_glm5_kda_workspace scalar = {}, batch = {};
                REQUIRE(ds4_glm5_kda_workspace_init(&scalar,1));
                REQUIRE(ds4_glm5_kda_workspace_init(&batch,m));
                auto run=[&](unsigned first, unsigned count, unsigned dest,
                             ds4_glm5_kda_workspace &ws) {
                    auto *x=ds4_gpu_tensor_view(input,(uint64_t)first*4096u*4u,(uint64_t)count*4096u*4u);
                    auto *y=ds4_gpu_tensor_view(output,(uint64_t)dest*4096u*4u,(uint64_t)count*4096u*4u);
                    REQUIRE(x && y);
                    REQUIRE(ds4_glm5_kda_layer_begin(&state.local,&ws,&weights,
                        g.map,g.size,x,y,count,1.0e-5f,rank*32u,32u));
                    REQUIRE(ds4_glm5_kda_layer_commit(&state.local,count));
                    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(y);
                };
                for (unsigned t=0;t<prefix;++t) run(t,1,0,scalar);
                std::vector<std::vector<float>> captures(7);
                if (arm==0) {
                    for (unsigned t=0;t<m;++t) {
                        run(prefix+t,1,t,scalar);
                        for (unsigned j=0;j<7;++j) {
                            auto row=read(scalar.*stages[j].member,stages[j].width);
                            captures[j].insert(captures[j].end(),row.begin(),row.end());
                        }
                    }
                } else {
                    run(prefix,m,0,batch);
                    for (unsigned j=0;j<7;++j)
                        captures[j]=read(batch.*stages[j].member,(uint64_t)m*stages[j].width);
                }
                captures.push_back(read(output,(uint64_t)m*4096u));
                for (auto *t : {state.local.q_history,state.local.k_history,
                                state.local.v_history,state.local.recurrent})
                    captures.push_back(read(t,ds4_gpu_tensor_bytes(t)/4u));
                // A scalar continuation exposes a damaged recurrence/history.
                run(prefix+m,1,0,scalar);
                captures.push_back(read(output,4096u));
                REQUIRE(state.local.token_count==prefix+m+1u && state.local.pending_tokens==0);
                const char *names[]={"norm","q","k","v","g_a","g_b","beta",
                    "gated","q_history","k_history","v_history","recurrent","next_gated"};
                if (arm==0) reference=captures;
                else for (unsigned j=0;j<captures.size();++j) {
                    const auto diff=compare(reference[j],captures[j],names[j],il,rank,m,prefix,arm);
                    if (arm==2) { candidate_different+=diff; candidate_values+=captures[j].size(); }
                }
                ds4_glm5_kda_workspace_free(&batch);
                ds4_glm5_kda_workspace_free(&scalar);
            }
            ds4_gpu_tensor_free(input); ds4_gpu_tensor_free(output);
        }
    }
    std::printf("SMALL_M_KDA_RESULT diagnostic=%d candidate_values=%llu different=%llu\n",
        diagnostic,(unsigned long long)candidate_values,(unsigned long long)candidate_different);
    REQUIRE(diagnostic || candidate_different==0);
    if (timing) time_kda(g,bind(g,0));
    return 0;
}
