// Real-GGUF prefill panel probe. No DS4 runtime or weight-copy cache.
// hipcc --offload-arch=gfx1151 -mno-wavefrontsize64 -O3 -fno-fast-math
//   -ffp-contract=off -I. scripts/glm5_bf16_panel_bench.cu -o <artifact>/probe
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "tests/glm5_gguf_test.hpp"
#include "rocm/ds4_rocm_bf16_toktile.cuh"
#include "scripts/glm5_bf16_panel_kernels.cuh"

static void check(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", what); std::exit(1); }
}
static void hip_check(hipError_t status, const char *what) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "FAIL %s: %s\n", what, hipGetErrorString(status));
        std::exit(1);
    }
}

// Separate file mappings avoid overlapping registered edge pages. Each GPU
// pointer refers to the original GGUF bytes, including for the second rank.
struct WeightView {
    void *mapping = nullptr;
    size_t bytes = 0;
    const uint16_t *device = nullptr;
    void bind(int fd, uint64_t offset, uint64_t count) {
        const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
        const uint64_t base = offset / page * page;
        bytes = (size_t)((offset - base + count * 2u + page - 1u) / page * page);
        mapping = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, (off_t)base);
        check(mapping != MAP_FAILED, "map weight slice");
        hip_check(hipHostRegister(mapping, bytes, hipHostRegisterMapped),
                  "register weight slice");
        void *gpu = nullptr;
        hip_check(hipHostGetDevicePointer(&gpu, mapping, 0), "weight device pointer");
        device = (const uint16_t *)((const char *)gpu + offset - base);
    }
    ~WeightView() {
        if (mapping && mapping != MAP_FAILED) {
            hip_check(hipHostUnregister(mapping), "unregister weights");
            check(munmap(mapping, bytes) == 0, "unmap weights");
        }
    }
};

int main() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    check(model && *model, "DS4_GLM5_MODEL required");
    Glm5TestGGUF gguf;
    check(gguf.open_file(model), "open real GGUF");
    hipDeviceProp_t props{};
    hip_check(hipGetDeviceProperties(&props, 0), "device properties");
    check(std::strstr(props.gcnArchName, "gfx1151") != nullptr &&
          props.warpSize == 32, "gfx1151 wave32 required");
    constexpr uint32_t M = 256, K = 4096, N = 4096;
    constexpr size_t outputs = (size_t)3 * M * N;
    float *x = nullptr, *out[6] = {};
    hip_check(hipMalloc(&x, (size_t)M * K * sizeof(float)), "allocate input");
    for (auto &p : out)
        hip_check(hipMalloc(&p, outputs * sizeof(float)), "allocate output");
    std::vector<float> input((size_t)M * K);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = 0.23f * std::cos((double)(i % 8191u) * 0.009) -
                   0.08f * std::sin((double)i * 0.004);
    hip_check(hipMemcpy(x, input.data(), input.size() * sizeof(float),
                        hipMemcpyHostToDevice), "upload input");
    hipEvent_t begin, end;
    hip_check(hipEventCreate(&begin), "begin event");
    hip_check(hipEventCreate(&end), "end event");
    const char *names[] = {"kda_q.weight", "kda_k.weight", "kda_v.weight"};
    const char *arms[] = {"sequential", "column-b", "production-fused",
                          "transpose-b", "transpose-b-k32", "row-b-k32"};
    std::puts("workload=real-GGUF-QKV M=256 K=4096 N=4096 weights=mapped inputs=synthetic");
    for (unsigned arm = 0; arm < 6; ++arm)
        std::printf("arm=%u name=%s\n", arm, arms[arm]);
    for (uint32_t layer : {0u, 4u, 20u, 44u}) {
        for (uint32_t rank : {0u, 1u}) {
            WeightView weights[3];
            for (uint32_t p = 0; p < 3; ++p) {
                uint64_t offset = 0;
                check(gguf.tensor("blk." + std::to_string(layer) + "." + names[p],
                                  {K, 2u * N}, 30u, offset), "BF16 QKV geometry");
                weights[p].bind(gguf.fd, offset + (uint64_t)rank * K * N * 2u,
                                (uint64_t)K * N);
            }
            auto launch = [&](unsigned arm) {
                if (arm == 2u) {
                    matmul_bf16_f32_wmma_hilo_qkv_multiptr_kernel<<<
                        dim3(3u * (N / 32u), 1), 512>>>(
                        out[arm], out[arm] + M * N, out[arm] + 2u * M * N,
                        weights[0].device, weights[1].device, weights[2].device,
                        x, K, N, M);
                } else {
                    for (unsigned p = 0; p < 3; ++p) {
                        if (arm == 0u)
                            matmul_bf16_f32_wmma_hilo_m256_kernel<2u><<<
                                dim3(N / 32u, 1), 512>>>(
                                out[arm] + p * M * N, weights[p].device, x, K, N, M);
                        else if (arm == 1u)
                            ds4_bf16_panel_probe_kernel<2u, true><<<
                                dim3(N / 32u, 1), 512>>>(
                                out[arm] + p * M * N, weights[p].device, x, K, N, M);
                        else if (arm == 3u)
                            ds4_bf16_panel_probe_kernel<2u, true, true><<<
                                dim3(N / 32u, 1), 512>>>(
                                out[arm] + p * M * N, weights[p].device, x, K, N, M);
                        else if (arm == 4u)
                            ds4_bf16_panel_probe_kernel<2u, true, true, 32u><<<
                                dim3(N / 32u, 1), 512>>>(
                                out[arm] + p * M * N, weights[p].device, x, K, N, M);
                        else
                            ds4_bf16_panel_probe_kernel<2u, false, false, 32u><<<
                                dim3(N / 32u, 1), 512>>>(
                                out[arm] + p * M * N, weights[p].device, x, K, N, M);
                    }
                }
                hip_check(hipGetLastError(), "panel launch");
            };
            for (unsigned arm = 0; arm < 6; ++arm) launch(arm);
            hip_check(hipDeviceSynchronize(), "warm and compare");
            std::vector<float> reference(outputs), candidate(outputs);
            hip_check(hipMemcpy(reference.data(), out[0], outputs * sizeof(float),
                                hipMemcpyDeviceToHost), "read reference");
            for (float v : reference) check(std::isfinite(v), "finite reference");
            for (unsigned arm = 1; arm < 6; ++arm) {
                hip_check(hipMemcpy(candidate.data(), out[arm], outputs * sizeof(float),
                                    hipMemcpyDeviceToHost), "read candidate");
                size_t mismatches = 0;
                float max_abs = 0;
                for (size_t i = 0; i < outputs; ++i) {
                    check(std::isfinite(candidate[i]), "finite candidate");
                    mismatches += std::memcmp(&reference[i], &candidate[i], sizeof(float)) != 0;
                    max_abs = std::max(max_abs, std::fabs(reference[i] - candidate[i]));
                }
                std::printf("compare layer=%u rank=%u arm=%u mismatches=%zu max_abs=%.9g\n",
                            layer, rank, arm, mismatches, max_abs);
                check(mismatches == 0, "bit-exact hi/lo output");
            }
            std::vector<float> samples[6];
            for (unsigned sample = 0; sample < 5; ++sample) {
                for (unsigned order = 0; order < 6; ++order) {
                    const unsigned arm = sample % 2 ? 5u - order : order;
                    hip_check(hipEventRecord(begin), "record begin");
                    for (unsigned repeat = 0; repeat < 3; ++repeat) launch(arm);
                    hip_check(hipEventRecord(end), "record end");
                    hip_check(hipEventSynchronize(end), "wait timing");
                    float ms = 0;
                    hip_check(hipEventElapsedTime(&ms, begin, end), "elapsed time");
                    samples[arm].push_back(ms / 3.0f);
                }
            }
            for (unsigned arm = 0; arm < 6; ++arm) {
                std::sort(samples[arm].begin(), samples[arm].end());
                std::printf("timing layer=%u rank=%u arm=%u median_ms=%.6f min_ms=%.6f max_ms=%.6f\n",
                            layer, rank, arm, samples[arm][2], samples[arm][0], samples[arm][4]);
            }
            std::fflush(stdout);
        }
    }
    hip_check(hipEventDestroy(begin), "destroy begin");
    hip_check(hipEventDestroy(end), "destroy end");
    for (auto p : out) hip_check(hipFree(p), "free output");
    hip_check(hipFree(x), "free input");
    std::puts("PASS real-GGUF coalesced BF16 panel probe");
}
