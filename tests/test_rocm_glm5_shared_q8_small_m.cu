// Original Q8_0 shared experts: two scalar projections and K-slice down oracle.
#include "ds4_glm5_next_exec.h"
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
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); std::exit(1); } } while (0)

struct Buffer {
    ds4_gpu_tensor *storage, *view;
    uint64_t count;
    explicit Buffer(uint64_t n) : storage(ds4_gpu_tensor_alloc((n+32)*4)),
        view(storage?ds4_gpu_tensor_view(storage,64,n*4):nullptr), count(n) { REQUIRE(view); }
    ~Buffer() { ds4_gpu_tensor_free(view); ds4_gpu_tensor_free(storage); }
    void poison() { REQUIRE(ds4_gpu_tensor_fill_f32(storage,12345.0f,count+32)); }
    std::vector<float> read() {
        std::vector<float> all(count+32);
        REQUIRE(ds4_gpu_tensor_read(storage,0,all.data(),all.size()*4));
        for (unsigned i=0;i<16;++i) REQUIRE(all[i]==12345.0f && all[count+16+i]==12345.0f);
        for (uint64_t i=16;i<count+16;++i) REQUIRE(std::isfinite(all[i]));
        return std::vector<float>(all.begin()+16,all.end()-16);
    }
};

int main() {
    const char *path=std::getenv("DS4_GLM5_MODEL"); REQUIRE(path);
    REQUIRE(setenv("DS4_GLM5_NEXT_ENABLE_ORDINARY","1",1)==0);
    Glm5TestGGUF g; REQUIRE(g.open_file(path));
    uint64_t offsets[42*3], sizes[42*3];
    for (unsigned il=3;il<45;++il) for (unsigned p=0;p<3;++p) {
        char name[80]; std::snprintf(name,sizeof(name),"blk.%u.ffn_%s_shexp.weight",il,
            p==0?"gate":p==1?"up":"down");
        REQUIRE(g.tensor(name,p==2?std::vector<uint64_t>{2048,4096}:
            std::vector<uint64_t>{4096,2048},8u,offsets[(il-3)*3+p]));
        sizes[(il-3)*3+p]=2048u*4096u/32u*34u;
    }
    REQUIRE(ds4_gpu_init()); ds4_gpu_set_glm_model(true); ds4_gpu_set_q8_cache_suppressed(1);
    REQUIRE(ds4_gpu_set_model_fd_for_map(g.fd,g.map));
    REQUIRE(ds4_gpu_set_model_map_spans(g.map,g.size,offsets,sizes,126,sizes[0]));
    uint64_t compared=0;
    for (unsigned il=3;il<45;++il) for (unsigned rank : {0u,1u}) for (unsigned m : glm5_test_verifier_widths()) {
        Buffer input((uint64_t)m*4096), gate((uint64_t)m*1024), up(gate.count), mid(gate.count), down(input.count);
        Buffer *outputs[]={&gate,&up,&mid,&down};
        const uint64_t gate_offset=offsets[(il-3)*3]+rank*1024u*4352u;
        const uint64_t up_offset=offsets[(il-3)*3+1]+rank*1024u*4352u;
        const uint64_t down_offset=offsets[(il-3)*3+2];
        std::vector<ds4_gpu_tensor *> xv(m),gv(m),uv(m),mv(m),dv(m);
        for (unsigned t=0;t<m;++t) {
            xv[t]=ds4_gpu_tensor_view(input.view,t*4096u*4u,4096u*4u);
            gv[t]=ds4_gpu_tensor_view(gate.view,t*1024u*4u,1024u*4u);
            uv[t]=ds4_gpu_tensor_view(up.view,t*1024u*4u,1024u*4u);
            mv[t]=ds4_gpu_tensor_view(mid.view,t*1024u*4u,1024u*4u);
            dv[t]=ds4_gpu_tensor_view(down.view,t*4096u*4u,4096u*4u);
            REQUIRE(xv[t] && gv[t] && uv[t] && mv[t] && dv[t]);
        }
        auto run=[&](bool batch) {
            if (batch) return ds4_rocm_glm5_shared_q8_small_m(gate.view,up.view,g.map,g.size,
                gate_offset,up_offset,4096,1024,4352,0,input.view,m) &&
                ds4_gpu_swiglu_tensor(mid.view,gate.view,up.view,m*1024,10.0f,1.0f) &&
                ds4_rocm_glm5_shared_q8_small_m(down.view,nullptr,g.map,g.size,
                    down_offset,0,1024,4096,2176,rank*1024,mid.view,m);
            for (unsigned t=0;t<m;++t)
                if (!ds4_gpu_matmul_q8_0_tensor(gv[t],g.map,g.size,gate_offset,4096,1024,xv[t],1) ||
                    !ds4_gpu_matmul_q8_0_tensor(uv[t],g.map,g.size,up_offset,4096,1024,xv[t],1) ||
                    !ds4_gpu_swiglu_tensor(mv[t],gv[t],uv[t],1024,10.0f,1.0f) ||
                    !ds4_gpu_matmul_q8_0_kslice_tensor(dv[t],g.map,g.size,down_offset,2048,rank*1024,1024,4096,mv[t],0))
                    return false;
            return true;
        };
        for (unsigned seed=0;seed<3;++seed) {
            std::vector<float> x(input.count);
            for (uint64_t i=0;i<x.size();++i)
                x[i]=seed==2?((int)((i*193u+(i/4096)*761u+il*31u)%997u)-498)/(1001.3f+(i%7)):
                    seed==1?((i&1)?-1.0f:1.0f)*(i%3?1e-4f:10.0f):0.0f;
            input.poison(); REQUIRE(ds4_gpu_tensor_write(input.view,0,x.data(),x.size()*4));
            for (auto *o:outputs) o->poison();
            REQUIRE(run(false) && ds4_gpu_synchronize());
            std::vector<float> reference[4];
            for (unsigned p=0;p<4;++p) reference[p]=outputs[p]->read();
            for (auto *o:outputs) o->poison();
            REQUIRE(run(true) && ds4_gpu_synchronize());
            for (unsigned p=0;p<4;++p) {
                auto got=outputs[p]->read();
                for (uint64_t i=0;i<got.size();++i) if (std::memcmp(&got[i],&reference[p][i],4)) {
                    std::fprintf(stderr,"DIFF layer=%u rank=%u m=%u seed=%u stage=%u index=%llu scalar=%.9g batch=%.9g\n",
                        il,rank,m,seed,p,(unsigned long long)i,reference[p][i],got[i]); std::exit(1);
                }
                compared+=got.size();
            }
            REQUIRE(input.read()==x);
            std::printf("SHARED_Q8_EXACT layer=%u rank=%u m=%u seed=%u PASS\n",il,rank,m,seed); std::fflush(stdout);
        }
        if (il==3) {
            REQUIRE(unsetenv("DS4_GLM5_NEXT_ENABLE_ORDINARY")==0);
            REQUIRE(!run(true));
            REQUIRE(setenv("DS4_GLM5_NEXT_ENABLE_ORDINARY","1",1)==0);
            auto bad_pair=[&](ds4_gpu_tensor *a,ds4_gpu_tensor *b,uint64_t off,unsigned k,unsigned n,uint64_t stride,unsigned first,unsigned tokens) {
                REQUIRE(!ds4_rocm_glm5_shared_q8_small_m(a,b,g.map,g.size,off,up_offset,k,n,stride,first,input.view,tokens));
            };
            bad_pair(gate.view,up.view,gate_offset,4096,1024,4352,0,1);
            bad_pair(gate.view,up.view,gate_offset,4096,1024,4352,0,3);
            bad_pair(gate.view,up.view,gate_offset,4096,1024,4352,0,5);
            bad_pair(gate.view,up.view,gate_offset,4096,1024,4352,0,7);
            bad_pair(gate.view,up.view,gate_offset,2048,1024,4352,0,m);
            bad_pair(gate.view,up.view,gate_offset,4096,1023,4352,0,m);
            bad_pair(gate.view,up.view,gate_offset,4096,1024,2176,0,m);
            bad_pair(gate.view,up.view,gate_offset,4096,1024,4352,32,m);
            bad_pair(gate.view,gate.view,gate_offset,4096,1024,4352,0,m);
            bad_pair(input.view,up.view,gate_offset,4096,1024,4352,0,m);
            bad_pair(gate.view,up.view,g.size-2,4096,1024,4352,0,m);
            bad_pair(gate.view,up.view,UINT64_MAX-1,4096,1024,4352,0,m);
            bad_pair(gate.view,up.view,gate_offset+1,4096,1024,4352,0,m);
            auto *short_out=ds4_gpu_tensor_view(gate.view,0,gate.count*4-4); REQUIRE(short_out);
            bad_pair(short_out,up.view,gate_offset,4096,1024,4352,0,m);
            ds4_gpu_tensor_free(short_out);
            auto *overlap=ds4_gpu_tensor_view(gate.storage,68,gate.count*4); REQUIRE(overlap);
            bad_pair(gate.view,overlap,gate_offset,4096,1024,4352,0,m);
            ds4_gpu_tensor_free(overlap);
            for (unsigned first : {1u,32u,2048u})
                REQUIRE(!ds4_rocm_glm5_shared_q8_small_m(down.view,nullptr,g.map,g.size,down_offset,0,
                    1024,4096,2176,first,mid.view,m));
            REQUIRE(!ds4_rocm_glm5_shared_q8_small_m(down.view,nullptr,g.map,g.size,down_offset,0,
                1024,4096,1088,rank*1024,mid.view,m));
            REQUIRE(setenv("DS4_ROCM_GLM5_Q8_SHAREDX_PREFETCH","0",1)==0);
            REQUIRE(!run(true));
            REQUIRE(setenv("DS4_ROCM_GLM5_Q8_SHAREDX_PREFETCH","8",1)==0);
            REQUIRE(setenv("DS4_ROCM_GLM5_Q8_DECODE_TILE","3",1)==0);
            REQUIRE(!run(true));
            REQUIRE(setenv("DS4_ROCM_GLM5_Q8_DECODE_TILE","1",1)==0);
            REQUIRE(run(true) && ds4_gpu_synchronize());
            hipEvent_t begin,end; REQUIRE(hipEventCreate(&begin)==hipSuccess && hipEventCreate(&end)==hipSuccess);
            std::vector<float> times[2];
            for (unsigned sample=0;sample<13;++sample) for (unsigned turn=0;turn<2;++turn) {
                const unsigned arm=turn^(sample&1);
                REQUIRE(hipEventRecord(begin)==hipSuccess && run(arm!=0));
                REQUIRE(hipEventRecord(end)==hipSuccess && hipEventSynchronize(end)==hipSuccess);
                float ms=0; REQUIRE(hipEventElapsedTime(&ms,begin,end)==hipSuccess);
                if (sample>=4) {
                    times[arm].push_back(ms);
                    std::printf("SHARED_Q8_SAMPLE rank=%u m=%u sample=%u batch=%u ms=%.6f\n",rank,m,sample-4,arm,ms);
                }
            }
            for (auto &v:times) std::sort(v.begin(),v.end());
            std::printf("SHARED_Q8_MEDIAN rank=%u m=%u scalar_ms=%.6f batch_ms=%.6f\n",rank,m,times[0][4],times[1][4]);
            REQUIRE(hipEventDestroy(begin)==hipSuccess && hipEventDestroy(end)==hipSuccess);
        }
        for (unsigned t=0;t<m;++t) {
            ds4_gpu_tensor_free(xv[t]); ds4_gpu_tensor_free(gv[t]); ds4_gpu_tensor_free(uv[t]);
            ds4_gpu_tensor_free(mv[t]); ds4_gpu_tensor_free(dv[t]);
        }
    }
    std::printf("PASS shared Q8 small-M compared_floats=%llu weights=original\n",(unsigned long long)compared);
    ds4_gpu_cleanup();
}
