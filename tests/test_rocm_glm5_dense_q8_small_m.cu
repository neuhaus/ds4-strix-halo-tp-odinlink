// Real original Q8_0 weights; complete scalar dense FFN arithmetic is the oracle.
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
    // Match the production GLM selector and the full-target fixture.
    REQUIRE(setenv("DS4_GLM5_NEXT_ENABLE_ORDINARY","1",1)==0);
    Glm5TestGGUF g; REQUIRE(g.open_file(path));
    uint64_t offsets[9], sizes[9];
    for (unsigned il=0;il<3;++il) for (unsigned p=0;p<3;++p) {
        char name[80]; std::snprintf(name,sizeof(name),"blk.%u.ffn_%s.weight",il,
            p==0?"gate":p==1?"up":"down");
        REQUIRE(g.tensor(name,p==2?std::vector<uint64_t>{12288,4096}:
            std::vector<uint64_t>{4096,12288},8u,offsets[il*3+p]));
        sizes[il*3+p]=12288u*4096u/32u*34u;
    }
    REQUIRE(ds4_gpu_init()); ds4_gpu_set_glm_model(true); ds4_gpu_set_q8_cache_suppressed(1);
    REQUIRE(ds4_gpu_set_model_fd_for_map(g.fd,g.map));
    REQUIRE(ds4_gpu_set_model_map_spans(g.map,g.size,offsets,sizes,9,sizes[0]));
    uint64_t compared=0;
    for (unsigned il=0;il<3;++il) for (unsigned m : glm5_test_verifier_widths()) {
        Buffer input((uint64_t)m*4096), gate((uint64_t)m*12288), up(gate.count), mid(gate.count), down(input.count);
        Buffer *outputs[]={&gate,&up,&mid,&down};
        std::vector<ds4_gpu_tensor *> xv(m),gv(m),uv(m),mv(m),dv(m);
        for (unsigned t=0;t<m;++t) {
            xv[t]=ds4_gpu_tensor_view(input.view,t*4096u*4u,4096u*4u);
            gv[t]=ds4_gpu_tensor_view(gate.view,t*12288u*4u,12288u*4u);
            uv[t]=ds4_gpu_tensor_view(up.view,t*12288u*4u,12288u*4u);
            mv[t]=ds4_gpu_tensor_view(mid.view,t*12288u*4u,12288u*4u);
            dv[t]=ds4_gpu_tensor_view(down.view,t*4096u*4u,4096u*4u);
            REQUIRE(xv[t] && gv[t] && uv[t] && mv[t] && dv[t]);
        }
        auto run=[&](bool batch) {
            if (batch) return ds4_rocm_glm5_dense_q8_small_m(gate.view,up.view,g.map,g.size,
                offsets[3*il],offsets[3*il+1],4096,12288,input.view,m) &&
                ds4_gpu_swiglu_tensor(mid.view,gate.view,up.view,m*12288,10.0f,1.0f) &&
                ds4_rocm_glm5_dense_q8_small_m(down.view,nullptr,g.map,g.size,
                    offsets[3*il+2],0,12288,4096,mid.view,m);
            for (unsigned t=0;t<m;++t)
                if (!ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(gv[t],uv[t],mv[t],g.map,g.size,
                        offsets[3*il],offsets[3*il+1],4096,12288,xv[t],10.0f) ||
                    !ds4_gpu_matmul_q8_0_tensor(dv[t],g.map,g.size,offsets[3*il+2],12288,4096,mv[t],1))
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
                    std::fprintf(stderr,"DIFF layer=%u m=%u seed=%u stage=%u index=%llu scalar=%.9g batch=%.9g\n",
                        il,m,seed,p,(unsigned long long)i,reference[p][i],got[i]); std::exit(1);
                }
                compared+=got.size();
            }
            REQUIRE(input.read()==x);
            std::printf("DENSE_Q8_EXACT layer=%u m=%u seed=%u PASS\n",il,m,seed); std::fflush(stdout);
        }
        if (il==0) {
            REQUIRE(unsetenv("DS4_GLM5_NEXT_ENABLE_ORDINARY")==0);
            REQUIRE(!ds4_rocm_glm5_dense_q8_small_m(gate.view,up.view,g.map,g.size,offsets[0],offsets[1],4096,12288,input.view,m));
            REQUIRE(setenv("DS4_GLM5_NEXT_ENABLE_ORDINARY","1",1)==0);
            REQUIRE(!ds4_rocm_glm5_dense_q8_small_m(gate.view,up.view,g.map,g.size,offsets[0],offsets[1],4096,12288,input.view,1));
            for (unsigned bad : {3u,5u,7u})
                REQUIRE(!ds4_rocm_glm5_dense_q8_small_m(gate.view,up.view,g.map,g.size,offsets[0],offsets[1],4096,12288,input.view,bad));
            REQUIRE(!ds4_rocm_glm5_dense_q8_small_m(gate.view,gate.view,g.map,g.size,offsets[0],offsets[1],4096,12288,input.view,m));
            auto *short_out=ds4_gpu_tensor_view(gate.view,0,gate.count*4-4); REQUIRE(short_out);
            REQUIRE(!ds4_rocm_glm5_dense_q8_small_m(short_out,up.view,g.map,g.size,offsets[0],offsets[1],4096,12288,input.view,m));
            ds4_gpu_tensor_free(short_out);
            REQUIRE(!ds4_rocm_glm5_dense_q8_small_m(gate.view,up.view,g.map,g.size,g.size-2,offsets[1],4096,12288,input.view,m));
            REQUIRE(setenv("DS4_ROCM_GLM5_Q8_SHAREDX_PREFETCH","0",1)==0);
            REQUIRE(!run(true));
            REQUIRE(setenv("DS4_ROCM_GLM5_Q8_SHAREDX_PREFETCH","8",1)==0);
            REQUIRE(run(true) && ds4_gpu_synchronize());
        }
        if (il==0) {
            hipEvent_t begin,end; REQUIRE(hipEventCreate(&begin)==hipSuccess && hipEventCreate(&end)==hipSuccess);
            std::vector<float> times[2];
            for (unsigned sample=0;sample<13;++sample) for (unsigned turn=0;turn<2;++turn) {
                const unsigned arm=turn^(sample&1);
                REQUIRE(hipEventRecord(begin)==hipSuccess && run(arm!=0));
                REQUIRE(hipEventRecord(end)==hipSuccess && hipEventSynchronize(end)==hipSuccess);
                float ms=0; REQUIRE(hipEventElapsedTime(&ms,begin,end)==hipSuccess);
                if (sample>=4) {
                    times[arm].push_back(ms);
                    std::printf("DENSE_Q8_SAMPLE m=%u sample=%u batch=%u ms=%.6f\n",m,sample-4,arm,ms);
                }
            }
            for (auto &v:times) std::sort(v.begin(),v.end());
            std::printf("DENSE_Q8_MEDIAN m=%u scalar_ms=%.6f batch_ms=%.6f\n",m,times[0][4],times[1][4]);
            REQUIRE(hipEventDestroy(begin)==hipSuccess && hipEventDestroy(end)==hipSuccess);
        }
        for (unsigned t=0;t<m;++t) {
            ds4_gpu_tensor_free(xv[t]); ds4_gpu_tensor_free(gv[t]); ds4_gpu_tensor_free(uv[t]);
            ds4_gpu_tensor_free(mv[t]); ds4_gpu_tensor_free(dv[t]);
        }
    }
    std::printf("PASS dense Q8 small-M compared_floats=%llu weights=original\n",(unsigned long long)compared);
    ds4_gpu_cleanup();
}
