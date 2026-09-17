// Diagnostic only: source-header hi/lo versus native-weight vendor GEMM.
#include "glm5_gguf_test.hpp"
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" hipError_t glm5_test_hilo_prepare(uint16_t *,uint16_t *,const float *,uint64_t);
extern "C" hipError_t glm5_test_hilo_reference(float *,const uint16_t *,const float *,unsigned,unsigned,unsigned);
#define REQUIRE(x) do { if (!(x)) { \
    std::fprintf(stderr,"FAIL line=%d: %s\n",__LINE__,#x); std::exit(1); \
} } while (0)

static float bf16_value(uint16_t v) {
    const uint32_t bits = uint32_t(v)<<16u;
    float value;
    std::memcpy(&value,&bits,sizeof(value));
    return value;
}

int main() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    REQUIRE(model);
    Glm5TestGGUF gguf;
    REQUIRE(gguf.open_file(model));
    hipblasHandle_t handle;
    REQUIRE(hipblasCreate(&handle) == HIPBLAS_STATUS_SUCCESS);
    std::puts("diagnostic=library-hilo lane=B weights=original single_weight_buffer=1");
    for (unsigned layer : {0u,44u}) for (unsigned layout=0; layout<3; ++layout) {
        const unsigned k = layout == 2 ? 8192u : 4096u, n = 4096u;
        char name[80];
        std::snprintf(name,sizeof(name),"blk.%u.kda_%s.weight",layer,
                      layout == 2 ? "output" : "q");
        uint64_t offset;
        if (layout == 2) REQUIRE(gguf.tensor(name,{8192,4096},30,offset));
        else REQUIRE(gguf.tensor(name,{4096,8192},30,offset));
        if (layout == 1) offset += uint64_t(4096)*4096u*2u;
        const auto *host_w = reinterpret_cast<const uint16_t *>(
            static_cast<const char *>(gguf.map)+offset);
        uint16_t *w;
        const size_t weight_bytes = size_t(k)*n*2u;
        REQUIRE(hipMalloc(&w,weight_bytes) == hipSuccess);
        REQUIRE(hipMemcpy(w,host_w,weight_bytes,hipMemcpyHostToDevice) == hipSuccess);
        for (unsigned m : {256u,1024u}) {
            std::vector<float> x(size_t(m)*k), ref(size_t(m)*n), result(ref.size());
            for (size_t i=0; i<x.size(); ++i)
                x[i] = float(std::sin(double(i)*0.017+layer)*0.13 +
                             std::cos(double(i)*0.037+layout)*0.19);
            float *dx,*out;
            uint16_t *hi,*lo;
            REQUIRE(hipMalloc(&dx,x.size()*4u) == hipSuccess);
            REQUIRE(hipMalloc(&out,ref.size()*4u) == hipSuccess);
            REQUIRE(hipMalloc(&hi,x.size()*2u) == hipSuccess);
            REQUIRE(hipMalloc(&lo,x.size()*2u) == hipSuccess);
            REQUIRE(hipMemcpy(dx,x.data(),x.size()*4u,hipMemcpyHostToDevice) == hipSuccess);
            auto launch = [&](unsigned mode) {
                if (!mode) {
                    REQUIRE(glm5_test_hilo_reference(out,w,dx,k,n,m) == hipSuccess);
                    return;
                }
                REQUIRE(glm5_test_hilo_prepare(hi,lo,dx,x.size()) == hipSuccess);
                const float alpha=1, zero=0, one=1;
                auto gemm = [&](const uint16_t *input, const float *beta) {
                    const auto status = hipblasGemmEx(handle,HIPBLAS_OP_T,HIPBLAS_OP_N,
                        n,m,k,&alpha,w,HIP_R_16BF,k,input,HIP_R_16BF,k,
                        beta,out,HIP_R_32F,n,HIPBLAS_COMPUTE_32F,HIPBLAS_GEMM_DEFAULT);
                    if (status != HIPBLAS_STATUS_SUCCESS)
                        std::fprintf(stderr,"library_status=%d layer=%u layout=%u M=%u\n",
                                     int(status),layer,layout,m);
                    REQUIRE(status == HIPBLAS_STATUS_SUCCESS);
                };
                gemm(hi,&zero);
                if (mode == 2) gemm(lo,&one);
            };
            launch(0);
            REQUIRE(hipMemcpy(ref.data(),out,ref.size()*4u,hipMemcpyDeviceToHost) == hipSuccess);
            for (float value:ref) REQUIRE(std::isfinite(value));
            double oracle[16], ref_error2=0;
            for (unsigned s=0; s<16; ++s) {
                const unsigned t=(s*17u)%m, row=(s*257u)%n;
                double sum=0;
                for (unsigned c=0; c<k; ++c)
                    sum += double(x[size_t(t)*k+c])*bf16_value(host_w[size_t(row)*k+c]);
                oracle[s]=sum;
                const double error=double(ref[size_t(t)*n+row])-sum;
                ref_error2 += error*error;
            }
            for (unsigned mode : {1u,2u}) {
                launch(mode);
                REQUIRE(hipMemcpy(result.data(),out,result.size()*4u,hipMemcpyDeviceToHost) == hipSuccess);
                double error2=0,norm2=0,max_abs=0,oracle_error2=0;
                size_t different=0;
                for (size_t i=0; i<result.size(); ++i) {
                    REQUIRE(std::isfinite(result[i]));
                    const double error=double(result[i])-ref[i];
                    error2 += error*error;
                    norm2 += double(ref[i])*ref[i];
                    max_abs=std::fmax(max_abs,std::fabs(error));
                    different += std::memcmp(&ref[i],&result[i],sizeof(float)) != 0;
                }
                for (unsigned s=0; s<16; ++s) {
                    const unsigned t=(s*17u)%m, row=(s*257u)%n;
                    const double error=double(result[size_t(t)*n+row])-oracle[s];
                    oracle_error2 += error*error;
                }
                std::printf("library_error layer=%u layout=%u M=%u K=%u N=%u mode=%u values=%zu different=%zu max_abs=%.9g nrmse=%.9g reference_oracle_rms=%.9g candidate_oracle_rms=%.9g\n",
                    layer,layout,m,k,n,mode,ref.size(),different,max_abs,
                    std::sqrt(error2/std::fmax(norm2,1e-30)),
                    std::sqrt(ref_error2/16),std::sqrt(oracle_error2/16));
            }
            hipEvent_t begin,end;
            REQUIRE(hipEventCreate(&begin) == hipSuccess);
            REQUIRE(hipEventCreate(&end) == hipSuccess);
            for (unsigned mode : {0u,1u,2u}) launch(mode);
            REQUIRE(hipDeviceSynchronize() == hipSuccess);
            for (unsigned mode : {1u,2u}) for (unsigned pair=0; pair<3; ++pair)
            for (unsigned arm=0; arm<2; ++arm) {
                const unsigned run_mode = (arm^(pair&1u)) ? mode : 0u;
                REQUIRE(hipEventRecord(begin,nullptr) == hipSuccess);
                for (unsigned repeat=0; repeat<3; ++repeat) launch(run_mode);
                REQUIRE(hipEventRecord(end,nullptr) == hipSuccess);
                REQUIRE(hipEventSynchronize(end) == hipSuccess);
                float ms=0;
                REQUIRE(hipEventElapsedTime(&ms,begin,end) == hipSuccess);
                REQUIRE(std::isfinite(ms) && ms>0);
                std::printf("library_time layer=%u layout=%u M=%u K=%u N=%u contrast=%u pair=%u mode=%u ms=%.6f activation_scratch_bytes=%zu weight_bytes=%zu\n",
                    layer,layout,m,k,n,mode,pair,run_mode,ms/3.0f,x.size()*4u,weight_bytes);
            }
            REQUIRE(hipEventDestroy(begin) == hipSuccess);
            REQUIRE(hipEventDestroy(end) == hipSuccess);
            REQUIRE(hipFree(lo) == hipSuccess);
            REQUIRE(hipFree(hi) == hipSuccess);
            REQUIRE(hipFree(out) == hipSuccess);
            REQUIRE(hipFree(dx) == hipSuccess);
            std::fflush(stdout);
        }
        REQUIRE(hipFree(w) == hipSuccess);
    }
    REQUIRE(hipblasDestroy(handle) == HIPBLAS_STATUS_SUCCESS);
    std::puts("COMPLETE library diagnostic; numerical equivalence and model quality not established");
}
