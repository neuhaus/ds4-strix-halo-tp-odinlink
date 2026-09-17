// Compare six-pointer prefill with the projections used by layer_begin:
// hi/lo WMMA QKV, but ordinary F32-activation f_a/g_a/beta reductions.
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
    std::fprintf(stderr, "FAIL line=%d: %s\n", __LINE__, #x); \
    std::exit(1); } } while (0)

static constexpr const char *weight_selector =
    "DS4_ROCM_GLM5_BF16_WMMA_COALESCED_WEIGHT";

template <typename Launch>
static void time_weight_modes(Launch launch, const char *role,
                              unsigned rank, unsigned rows) {
    hipEvent_t begin, end;
    REQUIRE(hipEventCreate(&begin) == hipSuccess);
    REQUIRE(hipEventCreate(&end) == hipSuccess);
    for (const char *mode : {"0", "1"}) {
        REQUIRE(setenv(weight_selector,mode,1) == 0);
        REQUIRE(launch() == 1);
    }
    REQUIRE(ds4_gpu_synchronize());
    for (unsigned pair=0; pair<3; ++pair) for (unsigned arm=0; arm<2; ++arm) {
        const unsigned mode = arm ^ (pair & 1u);
        REQUIRE(setenv(weight_selector,mode ? "1" : "0",1) == 0);
        REQUIRE(hipEventRecord(begin,nullptr) == hipSuccess);
        for (unsigned repeat=0; repeat<3; ++repeat) REQUIRE(launch() == 1);
        REQUIRE(hipEventRecord(end,nullptr) == hipSuccess);
        REQUIRE(hipEventSynchronize(end) == hipSuccess);
        float ms = 0;
        REQUIRE(hipEventElapsedTime(&ms,begin,end) == hipSuccess);
        REQUIRE(std::isfinite(ms) && ms > 0);
        std::printf("weight_load role=%s rank=%u rows=%u pair=%u mode=%u ms=%.6f\n",
                    role,rank,rows,pair,mode,ms/3.0f);
    }
    REQUIRE(hipEventDestroy(begin) == hipSuccess);
    REQUIRE(hipEventDestroy(end) == hipSuccess);
}

int main(int argc, char **argv) {
    REQUIRE(argc <= 3);
    const bool coalesced = argc == 3;
    if (coalesced) REQUIRE(std::strcmp(argv[2],"--coalesced") == 0);
    const char *skinny = argc >= 2 ? argv[1] : "0";
    REQUIRE(std::strcmp(skinny,"0") == 0 || std::strcmp(skinny,"1") == 0);
    const char *model = std::getenv("DS4_GLM5_MODEL");
    REQUIRE(model);
    Glm5TestGGUF gguf;
    REQUIRE(gguf.open_file(model));
    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_KDA_SIX_PREFILL", "1", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_WMMA_NATIVE", "0", 1) == 0);
    REQUIRE(setenv(weight_selector,"0",1) == 0);
    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_SKINNY_EXACT_TOKTILE", skinny, 1) == 0);
    REQUIRE(unsetenv("DS4_ROCM_DISABLE_BF16_BATCH_TOKTILE") == 0);
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    REQUIRE(ds4_gpu_init_multi(&config));
    REQUIRE(ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map));
    REQUIRE(ds4_gpu_set_model_map(gguf.map, gguf.size));
    const char *names[] = {"q", "k", "v", "f_a", "g_a", "beta"};
    const uint32_t full_widths[] = {8192,8192,8192,128,128,64};
    size_t total = 0;
    for (unsigned layer : {0u, 1u, 44u}) {
        uint64_t offsets[6];
        for (unsigned i=0; i<6; ++i) {
            char name[80];
            std::snprintf(name,sizeof(name),"blk.%u.kda_%s.weight",layer,names[i]);
            REQUIRE(gguf.tensor(name,{4096,full_widths[i]},30,offsets[i]));
        }
        // Layouts 0/1 are TP halves; layout 2 is the supported full-head API.
        for (unsigned rank=0; rank<3; ++rank) for (unsigned rows : {256u,1024u}) {
            if (rank == 2 && rows != 256) continue;
            const uint32_t q_width = rank < 2 ? 4096u : 8192u;
            const uint32_t widths[] = {q_width,q_width,q_width,128,128,
                                      rank < 2 ? 32u : 64u};
            uint64_t local[6];
            ds4_gpu_tensor *reference[6], *candidate[6];
            for (unsigned i=0; i<6; ++i) {
                local[i] = offsets[i] + (rank < 2 && (i<3 || i==5) ?
                    uint64_t(rank)*widths[i]*4096u*2u : 0u);
                reference[i] = ds4_gpu_tensor_alloc(uint64_t(rows)*widths[i]*4u);
                candidate[i] = ds4_gpu_tensor_alloc(uint64_t(rows)*widths[i]*4u);
                REQUIRE(reference[i] && candidate[i]);
                REQUIRE(ds4_gpu_tensor_fill_f32(candidate[i],NAN,rows*widths[i]));
            }
            std::vector<float> x(size_t(rows)*4096u);
            for (size_t i=0; i<x.size(); ++i)
                x[i] = float(std::sin(double(i)*0.013+layer)*0.17 +
                             std::cos(double(i)*0.031+rank)*0.11);
            ds4_gpu_tensor *input = ds4_gpu_tensor_alloc(x.size()*4u);
            REQUIRE(input && ds4_gpu_tensor_write(input,0,x.data(),x.size()*4u));
            REQUIRE(setenv(weight_selector,"0",1) == 0);
            if (rank < 2) {
                REQUIRE(ds4_gpu_matmul_bf16_wmma_hilo_qkv_tensor(
                    reference[0],reference[1],reference[2],gguf.map,gguf.size,
                    local[0],local[1],local[2],4096,q_width,input,rows) == 1);
            } else {
                // The incumbent fused-QKV selector supports TP halves only;
                // full heads use the three ordinary hi/lo dispatches.
                for (unsigned i=0; i<3; ++i)
                    REQUIRE(ds4_gpu_matmul_bf16_wmma_hilo_tensor(reference[i],
                        gguf.map,gguf.size,local[i],4096,q_width,input,rows) == 1);
            }
            for (unsigned i=3; i<6; ++i)
                REQUIRE(ds4_gpu_matmul_bf16_tensor(reference[i],gguf.map,
                    gguf.size,local[i],4096,widths[i],input,rows));
            auto launch = [&]() { return ds4_gpu_matmul_bf16_kda_six_multiptr_tensor(
                candidate[0],candidate[1],candidate[2],candidate[3],
                candidate[4],candidate[5],gguf.map,gguf.size,
                local[0],local[1],local[2],local[3],local[4],local[5],
                4096,q_width,128,widths[5],input,rows); };
            REQUIRE(setenv(weight_selector,coalesced ? "1" : "0",1) == 0);
            REQUIRE(launch() == 1);
            // No partial tile may reach the exact kernel or alter its output.
            REQUIRE(ds4_gpu_matmul_bf16_kda_six_multiptr_tensor(
                candidate[0],candidate[1],candidate[2],candidate[3],
                candidate[4],candidate[5],gguf.map,gguf.size,
                local[0],local[1],local[2],local[3],local[4],local[5],
                4096,q_width,128,widths[5],input,rows-1u) == -1);
            if (coalesced) {
                // Invalid selectors must leave the recorded outputs intact.
                REQUIRE(setenv(weight_selector,"invalid",1) == 0);
                REQUIRE(launch() == 0);
                REQUIRE(setenv(weight_selector,"1",1) == 0);
            }
            bool exact = true;
            for (unsigned i=0; i<6; ++i) {
                const size_t count = size_t(rows)*widths[i];
                std::vector<float> a(count), b(count);
                REQUIRE(ds4_gpu_tensor_read(reference[i],0,a.data(),count*4u));
                REQUIRE(ds4_gpu_tensor_read(candidate[i],0,b.data(),count*4u));
                size_t different = 0;
                double max_abs = 0;
                for (size_t j=0; j<count; ++j) {
                    REQUIRE(std::isfinite(a[j]) && std::isfinite(b[j]));
                    different += std::memcmp(&a[j],&b[j],sizeof(float)) != 0;
                    max_abs = std::fmax(max_abs,std::fabs(double(a[j])-b[j]));
                }
                std::printf("layer=%u rank=%u rows=%u projection=%s values=%zu different=%zu max_abs=%.9g\n",
                    layer,rank,rows,names[i],count,different,max_abs);
                exact = exact && different == 0;
                total += count;
            }
            REQUIRE(exact);
            if (coalesced) {
                REQUIRE(ds4_gpu_matmul_bf16_wmma_hilo_tensor(candidate[0],
                    gguf.map,gguf.size,local[0],4096,q_width,input,rows) == 1);
                const size_t count = size_t(rows)*q_width;
                std::vector<float> a(count), b(count);
                REQUIRE(ds4_gpu_tensor_read(reference[0],0,a.data(),count*4u));
                REQUIRE(ds4_gpu_tensor_read(candidate[0],0,b.data(),count*4u));
                for (size_t i=0; i<count; ++i) {
                    REQUIRE(std::isfinite(a[i]) && std::isfinite(b[i]));
                    REQUIRE(std::memcmp(&a[i],&b[i],sizeof(float)) == 0);
                }
                total += count;
                std::printf("generic_weight_load layer=%u rank=%u rows=%u K=4096 N=%u values=%zu different=0\n",
                            layer,rank,rows,q_width,count);
            }
            if (coalesced && layer == 0 && rank < 2)
                time_weight_modes(launch,"six",rank,rows);
            for (unsigned i=0; i<6; ++i) {
                ds4_gpu_tensor_free(reference[i]);
                ds4_gpu_tensor_free(candidate[i]);
            }
            ds4_gpu_tensor_free(input);
            std::fflush(stdout);
            REQUIRE(exact);
        }
    }
    if (coalesced) {
        // Generic prefill hi/lo admits the full K8192/N4096 output shape.
        // Its load schedule is separate from the six-pointer kernel above.
        for (unsigned layer : {0u,44u}) for (unsigned rows : {256u,1024u}) {
            constexpr unsigned rank = 2u; // Full projection, as in the QKV oracle.
            char name[80];
            std::snprintf(name,sizeof(name),"blk.%u.kda_output.weight",layer);
            uint64_t offset;
            REQUIRE(gguf.tensor(name,{8192,4096},30,offset));
            std::vector<float> x(size_t(rows)*8192u), a(size_t(rows)*4096u), b(a.size());
            for (size_t i=0; i<x.size(); ++i)
                x[i] = float(std::sin(double(i)*0.017+layer)*0.13 +
                             std::cos(double(i)*0.037+rank)*0.19);
            auto *input = ds4_gpu_tensor_alloc(x.size()*4u);
            auto *reference = ds4_gpu_tensor_alloc(a.size()*4u);
            auto *candidate = ds4_gpu_tensor_alloc(a.size()*4u);
            REQUIRE(input && reference && candidate);
            REQUIRE(ds4_gpu_tensor_write(input,0,x.data(),x.size()*4u));
            auto call = [&](ds4_gpu_tensor *out) {
                return ds4_gpu_matmul_bf16_wmma_hilo_tensor(out,gguf.map,
                    gguf.size,offset,8192,4096,input,rows);
            };
            REQUIRE(setenv(weight_selector,"0",1) == 0);
            REQUIRE(call(reference) == 1);
            REQUIRE(setenv(weight_selector,"1",1) == 0);
            REQUIRE(call(candidate) == 1);
            REQUIRE(setenv(weight_selector,"invalid",1) == 0);
            REQUIRE(call(candidate) == 0);
            REQUIRE(setenv(weight_selector,"1",1) == 0);
            REQUIRE(setenv("DS4_ROCM_GLM5_BF16_WMMA_NATIVE","1",1) == 0);
            REQUIRE(call(candidate) == 0);
            REQUIRE(setenv("DS4_ROCM_GLM5_BF16_WMMA_NATIVE","0",1) == 0);
            REQUIRE(ds4_gpu_tensor_read(reference,0,a.data(),a.size()*4u));
            REQUIRE(ds4_gpu_tensor_read(candidate,0,b.data(),b.size()*4u));
            size_t different = 0;
            for (size_t i=0; i<a.size(); ++i) {
                REQUIRE(std::isfinite(a[i]) && std::isfinite(b[i]));
                different += std::memcmp(&a[i],&b[i],sizeof(float)) != 0;
            }
            std::printf("output_weight_load layer=%u rank=%u rows=%u values=%zu different=%zu\n",
                        layer,rank,rows,a.size(),different);
            REQUIRE(different == 0);
            total += a.size();
            if (layer == 0) time_weight_modes([&]() { return call(candidate); },"output",rank,rows);
            ds4_gpu_tensor_free(input);
            ds4_gpu_tensor_free(reference);
            ds4_gpu_tensor_free(candidate);
        }
    }
    REQUIRE(ds4_gpu_synchronize());
    ds4_gpu_cleanup();
    std::printf("PASS six-prefill matches production projections skinny=%s values=%zu\n",
                skinny,total);
}
