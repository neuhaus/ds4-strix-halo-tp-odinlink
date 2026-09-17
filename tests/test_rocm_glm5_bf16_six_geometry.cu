#include "glm5_gguf_test.hpp"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
extern "C" hipError_t glm5_six_geometry(float *const *,const uint16_t *const *,
    const float *,unsigned,unsigned,unsigned,unsigned,uint32_t *);
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr,"FAIL line=%d: %s\n",__LINE__,#x); std::exit(1); } } while (0)
int main() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    REQUIRE(model);
    Glm5TestGGUF gguf;
    REQUIRE(gguf.open_file(model));
    const char *names[] = {"q","k","v","f_a","g_a","beta"};
    constexpr unsigned full_rows[] = {8192,8192,8192,128,128,64};
    constexpr unsigned k = 4096u;
    constexpr size_t guard = 32;
    constexpr float sentinel = -1234567.0f;
    size_t total = 0;
    for (unsigned layer : {0u,44u}) for (unsigned layout=0; layout<3; ++layout) {
        const unsigned n = layout < 2u ? 4096u : 8192u;
        const unsigned beta = layout < 2u ? 32u : 64u;
        const unsigned rows[] = {n,n,n,128,128,beta};
        uint16_t *weight[6];
        const uint16_t *w[6];
        for (unsigned i=0; i<6; ++i) {
            char name[80];
            std::snprintf(name,sizeof(name),"blk.%u.kda_%s.weight",layer,names[i]);
            uint64_t offset;
            REQUIRE(gguf.tensor(name,{k,full_rows[i]},30,offset));
            if (layout == 1u && (i<3u || i==5u)) offset += uint64_t(rows[i])*k*2u;
            const size_t bytes = size_t(rows[i])*k*2u;
            REQUIRE(offset <= gguf.size && bytes <= gguf.size-offset);
            REQUIRE(hipMalloc(&weight[i],bytes) == hipSuccess);
            REQUIRE(hipMemcpy(weight[i],gguf.map+offset,bytes,hipMemcpyHostToDevice) == hipSuccess);
            w[i] = weight[i];
        }
        for (unsigned m : {256u,1024u}) {
            std::vector<float> x(size_t(m)*k);
            for (size_t i=0; i<x.size(); ++i)
                x[i] = float(std::sin(double(i)*0.017+layer)*0.13+
                             std::cos(double(i)*0.037+layout)*0.19);
            float *dx, *allocation[2][6], *out[2][6];
            uint32_t *panel;
            REQUIRE(hipMalloc(&dx,x.size()*4u) == hipSuccess);
            REQUIRE(hipMalloc(&panel,x.size()*4u) == hipSuccess);
            REQUIRE(hipMemcpy(dx,x.data(),x.size()*4u,hipMemcpyHostToDevice) == hipSuccess);
            for (unsigned arm=0; arm<2; ++arm) for (unsigned i=0; i<6; ++i) {
                const size_t count = size_t(m)*rows[i];
                std::vector<float> init(count+2*guard,sentinel);
                for (size_t j=0; j<count; ++j) init[j+guard] = NAN;
                REQUIRE(hipMalloc(&allocation[arm][i],init.size()*4u) == hipSuccess);
                out[arm][i] = allocation[arm][i]+guard;
                REQUIRE(hipMemcpy(allocation[arm][i],init.data(),init.size()*4u,hipMemcpyHostToDevice) == hipSuccess);
            }
            auto launch = [&](unsigned mode) {
                REQUIRE(glm5_six_geometry(out[mode],w,dx,n,beta,m,mode,panel) == hipSuccess);
            };
            launch(0);
            launch(1);
            for (unsigned i=0; i<6; ++i) {
                const size_t count = size_t(m)*rows[i];
                std::vector<float> a(count+2*guard), b(a.size());
                REQUIRE(hipMemcpy(a.data(),allocation[0][i],a.size()*4u,hipMemcpyDeviceToHost) == hipSuccess);
                REQUIRE(hipMemcpy(b.data(),allocation[1][i],b.size()*4u,hipMemcpyDeviceToHost) == hipSuccess);
                size_t different = 0;
                for (size_t j=0; j<count; ++j) {
                    REQUIRE(std::isfinite(a[j+guard]) && std::isfinite(b[j+guard]));
                    different += std::memcmp(&a[j+guard],&b[j+guard],4u) != 0;
                }
                for (size_t j=0; j<guard; ++j)
                    REQUIRE(a[j] == sentinel && b[j] == sentinel &&
                            a[count+guard+j] == sentinel && b[count+guard+j] == sentinel);
                std::printf("six_exact layer=%u layout=%u M=%u role=%u values=%zu different=%zu\n",
                            layer,layout,m,i,count,different);
                std::fflush(stdout);
                REQUIRE(different == 0);
                total += count;
            }
            REQUIRE(glm5_six_geometry(out[1],w,dx,n,beta,m-1u,1,panel) == hipErrorInvalidValue);
            REQUIRE(glm5_six_geometry(out[1],w,dx,n,beta,m,2,panel) == hipErrorInvalidValue);
            REQUIRE(glm5_six_geometry(out[1],w,dx,n,beta,m,1,nullptr) == hipErrorInvalidValue);
            hipEvent_t begin,end;
            REQUIRE(hipEventCreate(&begin) == hipSuccess);
            REQUIRE(hipEventCreate(&end) == hipSuccess);
            launch(0);
            launch(1);
            REQUIRE(hipDeviceSynchronize() == hipSuccess);
            for (unsigned pair=0; pair<3; ++pair) for (unsigned arm=0; arm<2; ++arm) {
                const unsigned mode = arm^(pair&1u);
                REQUIRE(hipEventRecord(begin,nullptr) == hipSuccess);
                for (unsigned repeat=0; repeat<3; ++repeat) launch(mode);
                REQUIRE(hipEventRecord(end,nullptr) == hipSuccess);
                REQUIRE(hipEventSynchronize(end) == hipSuccess);
                float ms = 0;
                REQUIRE(hipEventElapsedTime(&ms,begin,end) == hipSuccess);
                REQUIRE(std::isfinite(ms) && ms>0);
                std::printf("six_time layer=%u layout=%u M=%u pair=%u mode=%u ms=%.6f\n",
                            layer,layout,m,pair,mode,ms/3);
            }
            REQUIRE(hipEventDestroy(begin) == hipSuccess);
            REQUIRE(hipEventDestroy(end) == hipSuccess);
            for (unsigned arm=0; arm<2; ++arm) for (unsigned i=0; i<6; ++i)
                REQUIRE(hipFree(allocation[arm][i]) == hipSuccess);
            REQUIRE(hipFree(panel) == hipSuccess);
            REQUIRE(hipFree(dx) == hipSuccess);
            std::fflush(stdout);
        }
        for (unsigned i=0; i<6; ++i) REQUIRE(hipFree(weight[i]) == hipSuccess);
    }
    std::printf("PASS six geometry values=%zu; no full-model speed claim\n",total);
}
