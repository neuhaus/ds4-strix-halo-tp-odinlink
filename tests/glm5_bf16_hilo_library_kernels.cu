// Compile with production flags; keep host numerical checks in a separate TU.
#include <hip/hip_runtime.h>
#include <cstdint>
#include "../rocm/ds4_rocm_bf16_toktile.cuh"

__global__ static void prepare_hilo_library(uint16_t *hi, uint16_t *lo,
                                           const float *x, uint64_t count) {
    const uint64_t i = uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i >= count) return;
    const float value = x[i];
    const uint16_t high = ds4_bf16_rne_bits(value);
    hi[i] = high;
    lo[i] = ds4_bf16_rne_bits(value-__uint_as_float(uint32_t(high)<<16u));
}

extern "C" hipError_t glm5_test_hilo_prepare(uint16_t *hi, uint16_t *lo,
                                           const float *x, uint64_t count) {
    prepare_hilo_library<<<(count+255u)/256u,256>>>(hi,lo,x,count);
    return hipGetLastError();
}

extern "C" hipError_t glm5_test_hilo_reference(float *out, const uint16_t *w,
                                             const float *x, unsigned k,
                                             unsigned n, unsigned m) {
    matmul_bf16_f32_wmma_hilo_m256_kernel<2u><<<dim3(n/32u,m/256u),512>>>(
        out,w,x,k,n,m);
    return hipGetLastError();
}
