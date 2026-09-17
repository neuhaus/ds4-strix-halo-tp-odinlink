// Diagnostic only: source-header hi/lo versus native-weight vendor GEMM.
#include "glm5_gguf_test.hpp"
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
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

int main(int argc, char **argv) {
    REQUIRE(argc == 1 || ((argc == 2 || argc == 3) && std::strcmp(argv[1],"--lt") == 0));
    const bool use_lt = argc >= 2;
    int requested = -1; // Without an index retain first-successful selection.
    if (argc == 3) {
        REQUIRE(std::strlen(argv[2]) == 1 && argv[2][0] >= '0' && argv[2][0] <= '7');
        requested = argv[2][0]-'0';
    }
    const char *model = std::getenv("DS4_GLM5_MODEL");
    REQUIRE(model);
    Glm5TestGGUF gguf;
    REQUIRE(gguf.open_file(model));
    hipblasHandle_t handle;
    REQUIRE(hipblasCreate(&handle) == HIPBLAS_STATUS_SUCCESS);
    hipblasLtHandle_t lt_handle = nullptr;
    if (use_lt) REQUIRE(hipblasLtCreate(&lt_handle) == HIPBLAS_STATUS_SUCCESS);
    std::printf("diagnostic=library-hilo lane=B weights=original single_weight_buffer=1 backend=%s\n",
                use_lt ? "hipblasLt" : "hipblas");
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
            gguf.map+offset);
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
            hipblasLtMatmulDesc_t desc = nullptr;
            hipblasLtMatrixLayout_t ad = nullptr, bd = nullptr, cd = nullptr;
            hipblasLtMatmulHeuristicResult_t algorithm = {};
            void *workspace = nullptr;
            if (use_lt) {
                REQUIRE(hipblasLtMatmulDescCreate(&desc,HIPBLAS_COMPUTE_32F,HIP_R_32F) == HIPBLAS_STATUS_SUCCESS);
                const hipblasOperation_t trans_a=HIPBLAS_OP_T, trans_b=HIPBLAS_OP_N;
                REQUIRE(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSA,&trans_a,sizeof(trans_a)) == HIPBLAS_STATUS_SUCCESS);
                REQUIRE(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSB,&trans_b,sizeof(trans_b)) == HIPBLAS_STATUS_SUCCESS);
                REQUIRE(hipblasLtMatrixLayoutCreate(&ad,HIP_R_16BF,k,n,k) == HIPBLAS_STATUS_SUCCESS);
                REQUIRE(hipblasLtMatrixLayoutCreate(&bd,HIP_R_16BF,k,m,k) == HIPBLAS_STATUS_SUCCESS);
                REQUIRE(hipblasLtMatrixLayoutCreate(&cd,HIP_R_32F,n,m,n) == HIPBLAS_STATUS_SUCCESS);
                hipblasLtMatmulPreference_t pref;
                REQUIRE(hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS);
                const size_t max_workspace = 16u*1024u*1024u;
                REQUIRE(hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&max_workspace,sizeof(max_workspace)) == HIPBLAS_STATUS_SUCCESS);
                hipblasLtMatmulHeuristicResult_t heuristics[8];
                int returned=0, chosen=-1;
                const auto status = hipblasLtMatmulAlgoGetHeuristic(lt_handle,desc,ad,bd,cd,cd,pref,8,heuristics,&returned);
                REQUIRE(hipblasLtMatmulPreferenceDestroy(pref) == HIPBLAS_STATUS_SUCCESS);
                REQUIRE(returned >= 0 && returned <= 8);
                if (status == HIPBLAS_STATUS_SUCCESS)
                    for (int i=0; i<returned; ++i) {
                        if (requested >= 0 && i != requested) continue;
                        if (heuristics[i].state == HIPBLAS_STATUS_SUCCESS &&
                            heuristics[i].workspaceSize <= max_workspace) { chosen=i; break; }
                    }
                std::printf("lt_plan layer=%u layout=%u M=%u K=%u N=%u status=%d returned=%d requested=%d chosen=%d\n",
                            layer,layout,m,k,n,int(status),returned,requested,chosen);
                std::fflush(stdout);
                REQUIRE(chosen >= 0);
                algorithm=heuristics[chosen];
                if (algorithm.workspaceSize)
                    REQUIRE(hipMalloc(&workspace,algorithm.workspaceSize) == hipSuccess);
                std::printf("lt_workspace_bytes=%zu\n",algorithm.workspaceSize);
            }
            auto launch = [&](unsigned mode) {
                if (!mode) {
                    REQUIRE(glm5_test_hilo_reference(out,w,dx,k,n,m) == hipSuccess);
                    return;
                }
                REQUIRE(glm5_test_hilo_prepare(hi,lo,dx,x.size()) == hipSuccess);
                const float alpha=1, zero=0, one=1;
                auto gemm = [&](const uint16_t *input, const float *beta) {
                    const auto status = use_lt ? hipblasLtMatmul(lt_handle,desc,&alpha,
                        w,ad,input,bd,beta,out,cd,out,cd,&algorithm.algo,
                        workspace,algorithm.workspaceSize,nullptr) :
                        hipblasGemmEx(handle,HIPBLAS_OP_T,HIPBLAS_OP_N,
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
            if (workspace) REQUIRE(hipFree(workspace) == hipSuccess);
            if (cd) REQUIRE(hipblasLtMatrixLayoutDestroy(cd) == HIPBLAS_STATUS_SUCCESS);
            if (bd) REQUIRE(hipblasLtMatrixLayoutDestroy(bd) == HIPBLAS_STATUS_SUCCESS);
            if (ad) REQUIRE(hipblasLtMatrixLayoutDestroy(ad) == HIPBLAS_STATUS_SUCCESS);
            if (desc) REQUIRE(hipblasLtMatmulDescDestroy(desc) == HIPBLAS_STATUS_SUCCESS);
            REQUIRE(hipFree(lo) == hipSuccess);
            REQUIRE(hipFree(hi) == hipSuccess);
            REQUIRE(hipFree(out) == hipSuccess);
            REQUIRE(hipFree(dx) == hipSuccess);
            std::fflush(stdout);
        }
        REQUIRE(hipFree(w) == hipSuccess);
    }
    REQUIRE(hipblasDestroy(handle) == HIPBLAS_STATUS_SUCCESS);
    if (lt_handle) REQUIRE(hipblasLtDestroy(lt_handle) == HIPBLAS_STATUS_SUCCESS);
    std::puts("COMPLETE library diagnostic; numerical equivalence and model quality not established");
}
