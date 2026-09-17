// Compare six-pointer prefill with the projections used by layer_begin:
// hi/lo WMMA QKV, but ordinary F32-activation f_a/g_a/beta reductions.
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
extern "C" {
#include "ds4_tp.h"
}
#include "glm5_gguf_test.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}
#define REQUIRE(x) do { if (!(x)) { \
    std::fprintf(stderr, "FAIL line=%d: %s\n", __LINE__, #x); \
    std::exit(1); } } while (0)

int main(int argc, char **argv) {
    REQUIRE(argc <= 2);
    const char *skinny = argc == 2 ? argv[1] : "0";
    REQUIRE(std::strcmp(skinny,"0") == 0 || std::strcmp(skinny,"1") == 0);
    const char *model = std::getenv("DS4_GLM5_MODEL");
    REQUIRE(model);
    Glm5TestGGUF gguf;
    REQUIRE(gguf.open_file(model));
    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_KDA_SIX_PREFILL", "1", 1) == 0);
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
            REQUIRE(ds4_gpu_matmul_bf16_wmma_hilo_qkv_tensor(
                reference[0],reference[1],reference[2],gguf.map,gguf.size,
                local[0],local[1],local[2],4096,q_width,input,rows) == 1);
            for (unsigned i=3; i<6; ++i)
                REQUIRE(ds4_gpu_matmul_bf16_tensor(reference[i],gguf.map,
                    gguf.size,local[i],4096,widths[i],input,rows));
            REQUIRE(ds4_gpu_matmul_bf16_kda_six_multiptr_tensor(
                candidate[0],candidate[1],candidate[2],candidate[3],
                candidate[4],candidate[5],gguf.map,gguf.size,
                local[0],local[1],local[2],local[3],local[4],local[5],
                4096,q_width,128,widths[5],input,rows) == 1);
            // No partial tile may reach the exact kernel or alter its output.
            REQUIRE(ds4_gpu_matmul_bf16_kda_six_multiptr_tensor(
                candidate[0],candidate[1],candidate[2],candidate[3],
                candidate[4],candidate[5],gguf.map,gguf.size,
                local[0],local[1],local[2],local[3],local[4],local[5],
                4096,q_width,128,widths[5],input,rows-1u) == -1);
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
                ds4_gpu_tensor_free(reference[i]);
                ds4_gpu_tensor_free(candidate[i]);
            }
            ds4_gpu_tensor_free(input);
            std::fflush(stdout);
            REQUIRE(exact);
        }
    }
    REQUIRE(ds4_gpu_synchronize());
    ds4_gpu_cleanup();
    std::printf("PASS six-prefill matches production projections skinny=%s values=%zu\n",
                skinny,total);
}
