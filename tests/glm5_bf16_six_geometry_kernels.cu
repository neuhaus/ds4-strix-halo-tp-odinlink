// Compile with production fast-math; the checking host TU is precise.
#include <hip/hip_runtime.h>
#include <cstdint>
#include "../rocm/ds4_rocm_bf16_toktile.cuh"

extern "C" hipError_t glm5_six_geometry(float *const *out,
        const uint16_t *const *w, const float *x, unsigned n, unsigned beta,
        unsigned m, unsigned mode, uint32_t *panel) {
    if (!out || !w || !x || (n != 4096u && n != 8192u) ||
        beta != (n == 4096u ? 32u : 64u) ||
        (m != 256u && m != 1024u) || mode > 1u || (mode && !panel))
        return hipErrorInvalidValue;
    for (unsigned i=0; i<6; ++i)
        if (!out[i] || !w[i]) return hipErrorInvalidValue;
    constexpr unsigned k = 4096u, low = 128u;
    const unsigned skinny_blocks = low+beta/2u;
    if (mode == 0u) {
        matmul_bf16_f32_wmma_hilo_kda_six_multiptr_kernel<><<<
            dim3(3u*n/32u+skinny_blocks,m/256u),512>>>(
            out[0],out[1],out[2],out[3],out[4],out[5],
            w[0],w[1],w[2],w[3],w[4],w[5],x,k,n,low,beta,m);
    } else {
        const uint64_t count = uint64_t(k)*m;
        ds4_bf16_hilo_prepare_kernel<<<(count+255u)/256u,256>>>(panel,x,count);
        auto status = hipGetLastError();
        if (status != hipSuccess) return status;
        matmul_bf16_f32_wmma_hilo_m96n32k32_kernel<true,0u,true,true><<<
            dim3(3u*n/32u,(m+95u)/96u),128>>>(
            out[0],w[0],x,k,n,m,panel,out[1],out[2],w[1],w[2]);
        status = hipGetLastError();
        if (status != hipSuccess) return status;
        matmul_bf16_f32_wmma_hilo_kda_six_multiptr_kernel<false,true><<<
            dim3(skinny_blocks,m/256u),512>>>(
            out[0],out[1],out[2],out[3],out[4],out[5],
            w[0],w[1],w[2],w[3],w[4],w[5],x,k,n,low,beta,m);
    }
    return hipGetLastError();
}
