// Production fast-math flags here; host checks live in a precise translation unit.
#include <hip/hip_runtime.h>
#include <cstdint>
#include "../rocm/ds4_rocm_bf16_toktile.cuh"

extern "C" hipError_t glm5_exact_geometry(float *out, const uint16_t *w,
        const float *x, unsigned k, unsigned n, unsigned m, unsigned mode,
        uint32_t *panel) {
    if (!out || !w || !x || !k || !n || k%32u || n%32u || !m || m%16u || mode>1u ||
        (mode == 1u && !panel))
        return hipErrorInvalidValue;
    if (mode == 0)
        matmul_bf16_f32_wmma_hilo_m256_kernel<2u><<<dim3(n/32u,(m+255u)/256u),512>>>(out,w,x,k,n,m);
    else {
        const uint64_t count=uint64_t(k)*m;
        ds4_bf16_hilo_prepare_kernel<<<(count+255u)/256u,256>>>(panel,x,count);
        const auto status=hipGetLastError();
        if (status != hipSuccess) return status;
        matmul_bf16_f32_wmma_hilo_m96n32k32_kernel<true,0u,true><<<dim3(n/32u,(m+95u)/96u),128>>>(out,w,x,k,n,m,panel);
    }
    return hipGetLastError();
}
