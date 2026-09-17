#include "glm5_gguf_test.hpp"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
extern "C" hipError_t glm5_exact_geometry(float *,const uint16_t *,const float *,unsigned,unsigned,unsigned,unsigned,uint32_t *);
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr,"FAIL line=%d: %s\n",__LINE__,#x); std::exit(1); } } while (0)
int main() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    REQUIRE(model);
    Glm5TestGGUF gguf;
    REQUIRE(gguf.open_file(model));
    size_t total = 0;
    for (unsigned layer : {0u,44u}) for (unsigned layout=0; layout<4; ++layout) {
        const unsigned k=layout == 3 ? 8192u : 4096u, n=layout == 2 ? 8192u : 4096u;
        char name[80];
        std::snprintf(name,sizeof(name),"blk.%u.kda_%s.weight",layer,layout == 3 ? "output" : "q");
        uint64_t offset;
        REQUIRE(gguf.tensor(name,layout == 3 ? std::vector<uint64_t>{8192,4096} :
                            std::vector<uint64_t>{4096,8192},30,offset));
        if (layout == 1) offset += uint64_t(4096)*4096u*2u;
        REQUIRE(offset <= gguf.size && uint64_t(k)*n*2u <= gguf.size-offset);
        uint16_t *w;
        REQUIRE(hipMalloc(&w,size_t(k)*n*2u) == hipSuccess);
        REQUIRE(hipMemcpy(w,gguf.map+offset,size_t(k)*n*2u,hipMemcpyHostToDevice) == hipSuccess);
        std::vector<unsigned> rows{256u,1024u};
        if (layer == 0 && layout == 0) rows.insert(rows.end(),{16u,80u,96u,112u,272u,1008u});
        for (unsigned m : rows) {
            constexpr size_t guard=32;
            constexpr float sentinel=-1234567.0f;
            const size_t count=size_t(m)*n;
            std::vector<float> x(size_t(m)*k), reference(count+2*guard,sentinel), candidate(reference);
            for (size_t i=0; i<count; ++i) candidate[i+guard]=NAN;
            for (size_t i=0; i<x.size(); ++i)
                x[i]=float(std::sin(double(i)*0.017+layer)*0.13+std::cos(double(i)*0.037+layout)*0.19);
            float *dx,*a,*b;
            uint32_t *panel;
            REQUIRE(hipMalloc(&panel,x.size()*4u) == hipSuccess);
            REQUIRE(hipMalloc(&dx,x.size()*4u) == hipSuccess);
            REQUIRE(hipMalloc(&a,reference.size()*4u) == hipSuccess);
            REQUIRE(hipMalloc(&b,candidate.size()*4u) == hipSuccess);
            REQUIRE(hipMemcpy(dx,x.data(),x.size()*4u,hipMemcpyHostToDevice) == hipSuccess);
            REQUIRE(hipMemcpy(a,reference.data(),reference.size()*4u,hipMemcpyHostToDevice) == hipSuccess);
            REQUIRE(hipMemcpy(b,candidate.data(),candidate.size()*4u,hipMemcpyHostToDevice) == hipSuccess);
            auto launch = [&](unsigned mode) {
                REQUIRE(glm5_exact_geometry((mode ? b : a)+guard,w,dx,k,n,m,mode,panel) == hipSuccess);
            };
            launch(0);
            launch(1);
            REQUIRE(hipMemcpy(reference.data(),a,reference.size()*4u,hipMemcpyDeviceToHost) == hipSuccess);
            REQUIRE(hipMemcpy(candidate.data(),b,candidate.size()*4u,hipMemcpyDeviceToHost) == hipSuccess);
            size_t different=0;
            for (size_t i=0; i<count; ++i) {
                REQUIRE(std::isfinite(reference[i+guard]) && std::isfinite(candidate[i+guard]));
                different += std::memcmp(&reference[i+guard],&candidate[i+guard],sizeof(float)) != 0;
            }
            for (size_t i=0; i<guard; ++i)
                REQUIRE(reference[i] == sentinel && candidate[i] == sentinel &&
                        reference[count+guard+i] == sentinel && candidate[count+guard+i] == sentinel);
            std::printf("exact_geometry layer=%u layout=%u M=%u K=%u N=%u values=%zu different=%zu\n",layer,layout,m,k,n,count,different);
            std::fflush(stdout);
            REQUIRE(different == 0);
            total += count;
            REQUIRE(glm5_exact_geometry(b+guard,w,dx,k,n,m-1u,1,panel) == hipErrorInvalidValue);
            REQUIRE(glm5_exact_geometry(b+guard,w,dx,k,n,m,2,panel) == hipErrorInvalidValue);
            REQUIRE(glm5_exact_geometry(b+guard,w,dx,k,n,m,1,nullptr) == hipErrorInvalidValue);
            if (m == 256u || m == 1024u) {
                hipEvent_t begin,end;
                REQUIRE(hipEventCreate(&begin) == hipSuccess);
                REQUIRE(hipEventCreate(&end) == hipSuccess);
                for (unsigned mode : {0u,1u}) launch(mode);
                REQUIRE(hipDeviceSynchronize() == hipSuccess);
                for (unsigned pair=0; pair<3; ++pair) for (unsigned arm=0; arm<2; ++arm) {
                    const unsigned mode=arm^(pair&1u);
                    REQUIRE(hipEventRecord(begin,nullptr) == hipSuccess);
                    for (unsigned repeat=0; repeat<3; ++repeat) launch(mode);
                    REQUIRE(hipEventRecord(end,nullptr) == hipSuccess);
                    REQUIRE(hipEventSynchronize(end) == hipSuccess);
                    float ms=0;
                    REQUIRE(hipEventElapsedTime(&ms,begin,end) == hipSuccess);
                    REQUIRE(std::isfinite(ms) && ms>0);
                    std::printf("geometry_time layer=%u layout=%u M=%u K=%u N=%u pair=%u mode=%u ms=%.6f\n",layer,layout,m,k,n,pair,mode,ms/3);
                }
                REQUIRE(hipEventDestroy(begin) == hipSuccess);
                REQUIRE(hipEventDestroy(end) == hipSuccess);
            }
            REQUIRE(hipFree(b) == hipSuccess);
            REQUIRE(hipFree(a) == hipSuccess);
            REQUIRE(hipFree(dx) == hipSuccess);
            REQUIRE(hipFree(panel) == hipSuccess);
            std::fflush(stdout);
        }
        REQUIRE(hipFree(w) == hipSuccess);
    }
    std::printf("PASS exact geometry values=%zu; no full-model speed claim\n",total);
}
