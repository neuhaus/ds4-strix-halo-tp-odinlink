// Exercise the public entry against frozen production objects and real GGUF.
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
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr,"FAIL line=%d: %s\n",__LINE__,#x); std::exit(1); } } while (0)
static constexpr const char *selector = "DS4_ROCM_GLM5_BF16_WMMA_EXACT_M96";
int main() {
    Glm5TestGGUF gguf;
    REQUIRE(std::getenv("DS4_GLM5_MODEL"));
    REQUIRE(gguf.open_file(std::getenv("DS4_GLM5_MODEL")));
    const char *competitors[] = {"DS4_ROCM_GLM5_BF16_WMMA_NATIVE",
        "DS4_ROCM_GLM5_BF16_WMMA_COALESCED_WEIGHT",
        "DS4_ROCM_GLM5_BF16_WMMA_WIDE_TILE", "DS4_ROCM_GLM5_BF16_LT_HILO"};
    for (const char *name : competitors) REQUIRE(setenv(name,"0",1) == 0);
    REQUIRE(unsetenv("DS4_ROCM_DISABLE_BF16_BATCH_TOKTILE") == 0);
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    REQUIRE(ds4_gpu_init_multi(&config));
    REQUIRE(ds4_gpu_set_model_fd_for_map(gguf.fd,gguf.map));
    REQUIRE(ds4_gpu_set_model_map(gguf.map,gguf.size));
    size_t total = 0;
    constexpr size_t guard = 32;
    constexpr float canary = 12345.25f;
    for (unsigned layer : {0u,44u}) for (unsigned layout=0; layout<4; ++layout)
    for (unsigned m : {256u,1024u}) {
        const unsigned k = layout == 3 ? 8192u : 4096u;
        const unsigned n = layout == 2 ? 8192u : 4096u;
        char name[80];
        std::snprintf(name,sizeof(name),"blk.%u.kda_%s.weight",layer,layout == 3 ? "output" : "q");
        uint64_t offset;
        REQUIRE(gguf.tensor(name,layout == 3 ? std::vector<uint64_t>{8192,4096} :
            std::vector<uint64_t>{4096,8192},30,offset));
        if (layout == 1) offset += uint64_t(4096)*4096u*2u;
        std::vector<float> x(size_t(m)*k), a(size_t(m)*n), b(a.size()),
                           guarded(a.size()+2*guard,canary);
        for (size_t i=0; i<x.size(); ++i)
            x[i] = float(std::sin(double(i)*0.017+layer)*0.13+
                         std::cos(double(i)*0.037+layout)*0.19);
        auto *input = ds4_gpu_tensor_alloc(x.size()*4u);
        auto *ref = ds4_gpu_tensor_alloc(a.size()*4u);
        auto *storage = ds4_gpu_tensor_alloc(guarded.size()*4u);
        REQUIRE(input && ref && storage);
        auto *cand = ds4_gpu_tensor_view(storage,guard*4u,a.size()*4u);
        REQUIRE(cand);
        REQUIRE(ds4_gpu_tensor_write(input,0,x.data(),x.size()*4u));
        REQUIRE(ds4_gpu_tensor_write(storage,0,guarded.data(),guarded.size()*4u));
        auto call = [&](ds4_gpu_tensor *out, const ds4_gpu_tensor *in,
                        uint64_t rows, uint64_t off) {
            return ds4_gpu_matmul_bf16_wmma_hilo_tensor(out,gguf.map,gguf.size,off,k,n,in,rows);
        };
        REQUIRE(setenv(selector,"0",1) == 0);
        REQUIRE(call(ref,input,m,offset) == 1);
        REQUIRE(setenv(selector,"1",1) == 0);
        REQUIRE(ds4_gpu_tensor_fill_f32(cand,NAN,a.size()));
        REQUIRE(call(cand,input,m,offset) == 1);
        REQUIRE(ds4_gpu_tensor_read(ref,0,a.data(),a.size()*4u));
        REQUIRE(ds4_gpu_tensor_read(cand,0,b.data(),b.size()*4u));
        for (size_t i=0; i<a.size(); ++i)
            REQUIRE(std::isfinite(a[i]) && std::isfinite(b[i]));
        REQUIRE(std::memcmp(a.data(),b.data(),a.size()*4u) == 0);
        REQUIRE(ds4_gpu_tensor_read(storage,0,guarded.data(),guarded.size()*4u));
        for (size_t i=0; i<guard; ++i)
            REQUIRE(guarded[i] == canary && guarded[guard+a.size()+i] == canary);
        total += a.size();
        REQUIRE(ds4_gpu_tensor_fill_f32(cand,NAN,a.size()));
        for (const char *invalid : {"", "2", "invalid"}) {
            REQUIRE(setenv(selector,invalid,1) == 0);
            REQUIRE(call(cand,input,m,offset) == 0);
        }
        REQUIRE(setenv(selector,"1",1) == 0);
        for (const char *other : competitors) {
            REQUIRE(setenv(other,"1",1) == 0);
            REQUIRE(call(cand,input,m,offset) == 0);
            REQUIRE(setenv(other,"0",1) == 0);
        }
        REQUIRE(call(cand,input,1,offset) == -1);
        REQUIRE(call(cand,input,m-1,offset) == -1);
        REQUIRE(call(cand,input,m,gguf.size) == 0);
        if (m == 1024u) {
            REQUIRE(call(cand,input,m,offset+2u) == 0);
            auto *unaligned = ds4_gpu_tensor_view(storage,4u,a.size()*4u);
            REQUIRE(unaligned && call(unaligned,input,m,offset) == 0);
            ds4_gpu_tensor_free(unaligned);
        }
        auto *short_out = ds4_gpu_tensor_view(cand,0,a.size()*4u-4u);
        auto *short_in = ds4_gpu_tensor_view(input,0,x.size()*4u-4u);
        REQUIRE(short_out && call(short_out,input,m,offset) == 0);
        REQUIRE(short_in && call(cand,short_in,m,offset) == 0);
        ds4_gpu_tensor_free(short_out);
        ds4_gpu_tensor_free(short_in);
        REQUIRE(call(input,input,m,offset) == 0);
        REQUIRE(call(nullptr,input,m,offset) == 0);
        REQUIRE(ds4_gpu_tensor_read(cand,0,b.data(),b.size()*4u));
        for (float value : b) REQUIRE(std::isnan(value));
        // An absent selector restores the incumbent even after scratch reuse.
        REQUIRE(unsetenv(selector) == 0);
        REQUIRE(call(cand,input,m,offset) == 1);
        REQUIRE(ds4_gpu_tensor_read(cand,0,b.data(),b.size()*4u));
        REQUIRE(std::memcmp(a.data(),b.data(),a.size()*4u) == 0);
        ds4_gpu_tensor_free(cand);
        ds4_gpu_tensor_free(storage);
        ds4_gpu_tensor_free(ref);
        ds4_gpu_tensor_free(input);
        std::printf("exact_integration layer=%u layout=%u M=%u K=%u N=%u values=%zu different=0\n",
                    layer,layout,m,k,n,a.size());
        std::fflush(stdout);
    }
    REQUIRE(ds4_gpu_synchronize());
    ds4_gpu_cleanup();
    std::printf("PASS exact integration finite bitwise values=%zu; canaries/refusal/default-off passed\n",total);
}
