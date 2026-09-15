#include "ds4_glm5_kda.h"
#include "ds4_gpu_mgpu.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

extern "C" int ds4_rocm_glm5_qkv_dispatch_test(
    ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *,
    const ds4_glm5_kda_device_args *, uint64_t, uint64_t,
    const ds4_gpu_tensor *);

// Exercise the real adapter selector without executing a kernel. Empty model
// bounds force every eligible call to fail before touching these addresses.
int main(int argc, char **argv) {
    if (argc != 2 || (argv[1][0] != '0' && argv[1][0] != '1') || argv[1][1])
        return 2;
    const bool prefill = argv[1][0] == '1';
    unsetenv("DS4_ROCM_GLM5_BF16_QKV_ACTIVATION_PANEL");
    unsetenv("DS4_ROCM_GLM5_BF16_KDA_SIX_MULTIPTR");
    unsetenv("DS4_ROCM_GLM5_BF16_KDA_SIX_PREFILL");
    setenv("DS4_ROCM_GLM5_BF16_QKV_DECODE_MULTIPTR", "1", 1);
    setenv("DS4_ROCM_GLM5_BF16_WMMA_QKV_FUSED", argv[1], 1);
    setenv("DS4_ROCM_GLM5_BF16_WMMA_HILO", "1", 1);
    setenv("DS4_ROCM_GLM5_BF16_QKV_SHARED_A_PREFILL", "0", 1);
    unsetenv("DS4_ROCM_DISABLE_BF16_BATCH_TOKTILE");
    ds4_gpu_tensor x = {}, q = {}, k = {}, v = {};
    ds4_gpu_tensor *tensors[] = {&x, &q, &k, &v};
    for (unsigned i = 0; i < 4; ++i) {
        tensors[i]->ptr = reinterpret_cast<void *>(
            UINT64_C(0x100000000) * (i + 1));
        tensors[i]->bytes = UINT64_C(256) * 4096 * sizeof(float);
    }
    ds4_glm5_kda_weight_offsets weights = {};
    weights.q_type = weights.k_type = weights.v_type = 30;
    ds4_glm5_kda_device_args args = {};
    args.weights = &weights;
    args.model_map = reinterpret_cast<const void *>(uintptr_t(1));
    args.model_size = 0;
    for (unsigned rank = 0; rank < 2; ++rank) {
        for (unsigned tokens : {1u, 256u}) {
            args.n_tokens = tokens;
            const int result = ds4_rocm_glm5_qkv_dispatch_test(
                &q, &k, &v, &args, rank * 4096u, 4096u, &x);
            const int expected = tokens == 1 || prefill ? 0 : -1;
            if (result != expected) {
                std::fprintf(stderr,
                    "FAIL requested QKV route rank=%u tokens=%u returned=%d "
                    "expected=%d (0=weight-bounds rejection)\n",
                    rank, tokens, result, expected);
                return 1;
            }
        }
    }
    args.n_tokens = 256;
    weights.k_type = 8;
    if (ds4_rocm_glm5_qkv_dispatch_test(
            &q, &k, &v, &args, 0, 4096, &x) != -1) return 1;
    weights.k_type = 30;
    if (ds4_rocm_glm5_qkv_dispatch_test(
            &q, &k, &v, &args, 0, 8192, &x) != -1) return 1;
    ds4_glm5_kda_workspace workspace = {};
    ds4_gpu_tensor panel = {};
    panel.ptr = reinterpret_cast<void *>(UINT64_C(0x500000000));
    panel.bytes = UINT64_C(256) * 4096u * sizeof(float);
    workspace.qkv_activation_panel = &panel;
    args.workspace = &workspace;
    for (const char *value : {"0", "1", "garbage"}) {
        setenv("DS4_ROCM_GLM5_BF16_QKV_ACTIVATION_PANEL", value, 1);
        for (unsigned tokens : {1u, 256u, 512u}) {
            args.n_tokens = tokens;
            // Garbage is deliberately a hard configuration failure for every
            // route. The explicit panel request requires fused WMMA at M256.
            const int expected = value[0] == 'g' || tokens == 1u || prefill ||
                (value[0] == '1' && tokens == 256u) ? 0 : -1;
            if (ds4_rocm_glm5_qkv_dispatch_test(
                    &q, &k, &v, &args, 0, 4096, &x) != expected) return 1;
        }
    }
    setenv("DS4_ROCM_GLM5_BF16_QKV_ACTIVATION_PANEL", "1", 1);
    args.n_tokens = 256;
    workspace.qkv_activation_panel = nullptr;
    if (ds4_rocm_glm5_qkv_dispatch_test(
            &q, &k, &v, &args, 0, 4096, &x) != 0) return 1;
    workspace.qkv_activation_panel = &panel;
    setenv("DS4_ROCM_GLM5_BF16_WMMA_HILO", "0", 1);
    if (ds4_rocm_glm5_qkv_dispatch_test(
            &q, &k, &v, &args, 0, 4096, &x) != 0) return 1;
    setenv("DS4_ROCM_GLM5_BF16_WMMA_HILO", "1", 1);
    setenv("DS4_ROCM_GLM5_BF16_QKV_SHARED_A_PREFILL", "1", 1);
    if (ds4_rocm_glm5_qkv_dispatch_test(
            &q, &k, &v, &args, 0, 4096, &x) != 0) return 1;
    std::puts("PASS independent QKV decode/prefill dispatch, both rank offsets; "
              "mixed quant, full-head shape and panel configuration guards");
    return 0;
}
