// Real-GGUF generic hi/lo projection oracle against frozen production objects.
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
extern "C" {
#include "ds4_tp.h"
}
#include "glm5_gguf_test.hpp"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}
#define REQUIRE(x) do { if (!(x)) { \
    std::fprintf(stderr,"FAIL line=%d: %s\n",__LINE__,#x); \
    std::exit(1); } } while (0)

static constexpr const char *selector = "DS4_ROCM_GLM5_BF16_WMMA_WIDE_TILE";

int main() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    REQUIRE(model);
    Glm5TestGGUF gguf;
    REQUIRE(gguf.open_file(model));
    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_WMMA_NATIVE","0",1) == 0);
    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_WMMA_COALESCED_WEIGHT","0",1) == 0);
    REQUIRE(unsetenv("DS4_ROCM_DISABLE_BF16_BATCH_TOKTILE") == 0);
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    REQUIRE(ds4_gpu_init_multi(&config));
    REQUIRE(ds4_gpu_set_model_fd_for_map(gguf.fd,gguf.map));
    REQUIRE(ds4_gpu_set_model_map(gguf.map,gguf.size));
    size_t total = 0;
    for (unsigned layer : {0u,44u}) {
        uint64_t q_offset, output_offset;
        char name[80];
        std::snprintf(name,sizeof(name),"blk.%u.kda_q.weight",layer);
        REQUIRE(gguf.tensor(name,{4096,8192},30,q_offset));
        std::snprintf(name,sizeof(name),"blk.%u.kda_output.weight",layer);
        REQUIRE(gguf.tensor(name,{8192,4096},30,output_offset));
        // Both TP output-row slices, full Q, and full KDA output.
        for (unsigned layout=0; layout<4; ++layout)
        for (unsigned rows : {256u,1024u}) {
            const unsigned k = layout == 3 ? 8192u : 4096u;
            const unsigned n = layout == 2 ? 8192u : 4096u;
            const uint64_t offset = layout == 3 ? output_offset :
                q_offset + (layout == 1 ? uint64_t(4096)*4096u*2u : 0u);
            std::vector<float> x(size_t(rows)*k), a(size_t(rows)*n), b(a.size());
            for (size_t i=0; i<x.size(); ++i)
                x[i] = float(std::sin(double(i)*0.017+layer)*0.13 +
                             std::cos(double(i)*0.037+layout)*0.19);
            auto *input = ds4_gpu_tensor_alloc(x.size()*4u);
            auto *ref = ds4_gpu_tensor_alloc(a.size()*4u);
            auto *cand = ds4_gpu_tensor_alloc(a.size()*4u);
            REQUIRE(input && ref && cand);
            REQUIRE(ds4_gpu_tensor_write(input,0,x.data(),x.size()*4u));
            auto call = [&](ds4_gpu_tensor *out, unsigned m) {
                return ds4_gpu_matmul_bf16_wmma_hilo_tensor(
                    out,gguf.map,gguf.size,offset,k,n,input,m);
            };
            REQUIRE(setenv(selector,"0",1) == 0);
            REQUIRE(call(ref,rows) == 1);
            REQUIRE(setenv(selector,"1",1) == 0);
            REQUIRE(ds4_gpu_tensor_fill_f32(cand,NAN,a.size()));
            REQUIRE(call(cand,rows) == 1);
            REQUIRE(ds4_gpu_tensor_read(ref,0,a.data(),a.size()*4u));
            REQUIRE(ds4_gpu_tensor_read(cand,0,b.data(),b.size()*4u));
            size_t different = 0;
            for (size_t i=0; i<a.size(); ++i) {
                REQUIRE(std::isfinite(a[i]) && std::isfinite(b[i]));
                different += std::memcmp(&a[i],&b[i],sizeof(float)) != 0;
            }
            std::printf("wide_tile layer=%u layout=%u rows=%u K=%u N=%u values=%zu different=%zu\n",
                        layer,layout,rows,k,n,a.size(),different);
            REQUIRE(different == 0);
            total += a.size();
            REQUIRE(ds4_gpu_tensor_fill_f32(cand,NAN,a.size()));
            for (const char *invalid : {"invalid","2",""}) {
                REQUIRE(setenv(selector,invalid,1) == 0);
                REQUIRE(call(cand,rows) == 0);
            }
            REQUIRE(setenv(selector,"1",1) == 0);
            for (const char *other : {"DS4_ROCM_GLM5_BF16_WMMA_NATIVE",
                                     "DS4_ROCM_GLM5_BF16_WMMA_COALESCED_WEIGHT"}) {
                REQUIRE(setenv(other,"1",1) == 0);
                REQUIRE(call(cand,rows) == 0);
                REQUIRE(setenv(other,"0",1) == 0);
            }
            REQUIRE(call(cand,1) == -1);
            REQUIRE(call(cand,rows-1u) == -1);
            REQUIRE(ds4_gpu_tensor_read(cand,0,b.data(),b.size()*4u));
            for (float value : b) REQUIRE(std::isnan(value));
            auto *short_out = ds4_gpu_tensor_view(cand,0,a.size()*4u-4u);
            REQUIRE(short_out);
            REQUIRE(call(short_out,rows) == 0);
            ds4_gpu_tensor_free(short_out);
            REQUIRE(ds4_gpu_tensor_read(cand,0,b.data(),b.size()*4u));
            for (float value : b) REQUIRE(std::isnan(value));
            REQUIRE(call(input,rows) == 0);
            REQUIRE(ds4_gpu_tensor_read(input,0,b.data(),16u*sizeof(float)));
            REQUIRE(std::memcmp(b.data(),x.data(),16u*sizeof(float)) == 0);

            if (layer == 0) {
                hipEvent_t begin,end;
                REQUIRE(hipEventCreate(&begin) == hipSuccess);
                REQUIRE(hipEventCreate(&end) == hipSuccess);
                for (const char *mode : {"0","1"}) {
                    REQUIRE(setenv(selector,mode,1) == 0);
                    REQUIRE(call(cand,rows) == 1);
                }
                REQUIRE(ds4_gpu_synchronize());
                for (unsigned pair=0; pair<3; ++pair)
                for (unsigned arm=0; arm<2; ++arm) {
                    const unsigned mode = arm ^ (pair & 1u);
                    REQUIRE(setenv(selector,mode ? "1" : "0",1) == 0);
                    REQUIRE(hipEventRecord(begin,nullptr) == hipSuccess);
                    for (unsigned repeat=0; repeat<3; ++repeat)
                        REQUIRE(call(cand,rows) == 1);
                    REQUIRE(hipEventRecord(end,nullptr) == hipSuccess);
                    REQUIRE(hipEventSynchronize(end) == hipSuccess);
                    float ms = 0;
                    REQUIRE(hipEventElapsedTime(&ms,begin,end) == hipSuccess);
                    REQUIRE(std::isfinite(ms) && ms > 0);
                    std::printf("wide_time layout=%u rows=%u K=%u N=%u pair=%u mode=%u ms=%.6f\n",
                                layout,rows,k,n,pair,mode,ms/3.0f);
                }
                REQUIRE(hipEventDestroy(begin) == hipSuccess);
                REQUIRE(hipEventDestroy(end) == hipSuccess);
            }
            ds4_gpu_tensor_free(input);
            ds4_gpu_tensor_free(ref);
            ds4_gpu_tensor_free(cand);
            std::fflush(stdout);
        }
    }
    REQUIRE(ds4_gpu_synchronize());
    ds4_gpu_cleanup();
    std::printf("PASS generic wide tile exact values=%zu\n",total);
}
