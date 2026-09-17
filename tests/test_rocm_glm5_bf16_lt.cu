// Production-object native-weight Lt integration diagnostic, not a quality gate.
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
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr,"FAIL line=%d: %s\n",__LINE__,#x); std::exit(1); } } while (0)
static constexpr const char *selector = "DS4_ROCM_GLM5_BF16_LT_HILO";
static float bf16(uint16_t value) {
    const uint32_t bits = uint32_t(value)<<16u;
    float result;
    std::memcpy(&result,&bits,sizeof(result));
    return result;
}
int main() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    REQUIRE(model);
    Glm5TestGGUF gguf;
    REQUIRE(gguf.open_file(model));
    for (const char *name : {"DS4_ROCM_GLM5_BF16_WMMA_NATIVE",
                            "DS4_ROCM_GLM5_BF16_WMMA_COALESCED_WEIGHT",
                            "DS4_ROCM_GLM5_BF16_WMMA_WIDE_TILE"})
        REQUIRE(setenv(name,"0",1) == 0);
    REQUIRE(unsetenv("DS4_ROCM_DISABLE_BF16_BATCH_TOKTILE") == 0);
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    REQUIRE(ds4_gpu_init_multi(&config));
    REQUIRE(ds4_gpu_set_model_fd_for_map(gguf.fd,gguf.map));
    REQUIRE(ds4_gpu_set_model_map(gguf.map,gguf.size));
    size_t total = 0;
    for (unsigned layer : {0u,44u}) for (unsigned layout=0; layout<3; ++layout)
    for (unsigned m : {256u,1024u}) {
        const unsigned k = layout == 2 ? 8192u : 4096u, n = 4096u;
        char name[80];
        std::snprintf(name,sizeof(name),"blk.%u.kda_%s.weight",layer,layout == 2 ? "output" : "q");
        uint64_t offset;
        REQUIRE(gguf.tensor(name,layout == 2 ? std::vector<uint64_t>{8192,4096} :
                            std::vector<uint64_t>{4096,8192},30,offset));
        if (layout == 1) offset += uint64_t(4096)*4096u*2u;
        const auto *w = reinterpret_cast<const uint16_t *>(gguf.map+offset);
        std::vector<float> x(size_t(m)*k), a(size_t(m)*n), b(a.size());
        for (size_t i=0; i<x.size(); ++i)
            x[i] = float(std::sin(double(i)*0.017+layer)*0.13+
                         std::cos(double(i)*0.037+layout)*0.19);
        auto *input = ds4_gpu_tensor_alloc(x.size()*4u);
        auto *ref = ds4_gpu_tensor_alloc(a.size()*4u);
        auto *cand = ds4_gpu_tensor_alloc(a.size()*4u);
        REQUIRE(input && ref && cand);
        REQUIRE(ds4_gpu_tensor_write(input,0,x.data(),x.size()*4u));
        auto call = [&](ds4_gpu_tensor *out, uint64_t rows, uint64_t off) {
            return ds4_gpu_matmul_bf16_wmma_hilo_tensor(out,gguf.map,gguf.size,off,k,n,input,rows);
        };
        REQUIRE(setenv(selector,"0",1) == 0);
        REQUIRE(call(ref,m,offset) == 1);
        REQUIRE(setenv(selector,"1",1) == 0);
        REQUIRE(ds4_gpu_tensor_fill_f32(cand,NAN,a.size()));
        REQUIRE(call(cand,m,offset) == 1);
        REQUIRE(ds4_gpu_tensor_read(ref,0,a.data(),a.size()*4u));
        REQUIRE(ds4_gpu_tensor_read(cand,0,b.data(),b.size()*4u));
        double error2=0,norm2=0,max_abs=0,ref_oracle2=0,cand_oracle2=0;
        size_t different=0;
        for (size_t i=0; i<a.size(); ++i) {
            REQUIRE(std::isfinite(a[i]) && std::isfinite(b[i]));
            const double error = double(b[i])-a[i];
            error2 += error*error;
            norm2 += double(a[i])*a[i];
            max_abs = std::fmax(max_abs,std::fabs(error));
            different += std::memcmp(&a[i],&b[i],sizeof(float)) != 0;
        }
        for (unsigned s=0; s<16; ++s) {
            const unsigned t=(s*17u)%m, row=(s*257u)%n;
            double dot=0;
            for (unsigned c=0; c<k; ++c)
                dot += double(x[size_t(t)*k+c])*bf16(w[size_t(row)*k+c]);
            const double da = double(a[size_t(t)*n+row])-dot;
            const double db = double(b[size_t(t)*n+row])-dot;
            ref_oracle2 += da*da;
            cand_oracle2 += db*db;
        }
        std::printf("lt_integration layer=%u layout=%u M=%u K=%u N=%u values=%zu different=%zu max_abs=%.9g nrmse=%.9g reference_oracle_rms=%.9g candidate_oracle_rms=%.9g\n",
            layer,layout,m,k,n,a.size(),different,max_abs,std::sqrt(error2/std::fmax(norm2,1e-30)),
            std::sqrt(ref_oracle2/16),std::sqrt(cand_oracle2/16));
        total += a.size();
        // Shape/selector failures must leave the caller's output untouched.
        REQUIRE(ds4_gpu_tensor_fill_f32(cand,NAN,a.size()));
        for (const char *invalid : {"", "2", "invalid"}) {
            REQUIRE(setenv(selector,invalid,1) == 0);
            REQUIRE(call(cand,m,offset) == 0);
        }
        REQUIRE(setenv(selector,"1",1) == 0);
        for (const char *other : {"DS4_ROCM_GLM5_BF16_WMMA_NATIVE",
                                 "DS4_ROCM_GLM5_BF16_WMMA_COALESCED_WEIGHT",
                                 "DS4_ROCM_GLM5_BF16_WMMA_WIDE_TILE"}) {
            REQUIRE(setenv(other,"1",1) == 0);
            REQUIRE(call(cand,m,offset) == 0);
            REQUIRE(setenv(other,"0",1) == 0);
        }
        REQUIRE(call(cand,1,offset) == -1);
        REQUIRE(call(cand,m-1,offset) == -1);
        REQUIRE(call(cand,m,gguf.size) == 0);
        REQUIRE(call(cand,m,offset+2u) == 0);
        auto *short_out = ds4_gpu_tensor_view(cand,0,a.size()*4u-4u);
        REQUIRE(short_out && call(short_out,m,offset) == 0);
        ds4_gpu_tensor_free(short_out);
        REQUIRE(call(input,m,offset) == 0);
        REQUIRE(ds4_gpu_tensor_read(cand,0,b.data(),b.size()*4u));
        for (float value : b) REQUIRE(std::isnan(value));
        // Switching off after plan reuse must reproduce the original path.
        REQUIRE(setenv(selector,"0",1) == 0);
        REQUIRE(call(cand,m,offset) == 1);
        REQUIRE(ds4_gpu_tensor_read(cand,0,b.data(),b.size()*4u));
        REQUIRE(std::memcmp(a.data(),b.data(),a.size()*4u) == 0);
        ds4_gpu_tensor_free(cand);
        ds4_gpu_tensor_free(ref);
        ds4_gpu_tensor_free(input);
        std::fflush(stdout);
    }
    REQUIRE(ds4_gpu_synchronize());
    ds4_gpu_cleanup();
    std::printf("PASS Lt integration finite values=%zu; numerical metrics are not quality approval\n",total);
}
