// Original-GGUF leaf comparison with production residency and full M1 oracle.
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "tests/glm5_gguf_test.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

static void check(bool ok, const char *label) {
    if (!ok) { fprintf(stderr, "FAIL %s\n", label); exit(1); }
}
static void hc(hipError_t e) { check(e == hipSuccess, hipGetErrorString(e)); }
static uint64_t hash(const std::vector<float>& v) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < v.size()*4; ++i)
        h = (h ^ ((const unsigned char *)v.data())[i])*UINT64_C(1099511628211);
    return h;
}
__global__ static void cold_read(const volatile unsigned *a, unsigned *sink) {
    unsigned v = 0;
    for (unsigned i = blockIdx.x*blockDim.x+threadIdx.x; i < 16777216u;
         i += gridDim.x*blockDim.x) v += a[i];
    if (threadIdx.x == 0) sink[blockIdx.x] = v;
}
int main(int argc, char **argv) {
    check(argc == 3, "usage: test family rank");
    const std::string family = argv[1];
    check(std::string(argv[2]) == "0" || std::string(argv[2]) == "1", "rank");
    const unsigned rank = atoi(argv[2]);
    uint32_t K = 0, fullK = 0, N = 0, fullN = 0;
    bool pair = false, slice = false;
    std::string name, name1;
    if (family == "qb") {
        K = fullK = 1536; N = fullN = 16384; name = "blk.3.attn_q_b.weight";
    } else if (family == "gate" || family == "up") {
        K = fullK = 4096; N = 1024; fullN = 2048;
        name = family == "gate" ? "blk.3.ffn_gate_shexp.weight" :
                                  "blk.3.ffn_up_shexp.weight";
    } else if (family == "mla") {
        K = 8192; fullK = 16384; N = fullN = 4096;
        name = "blk.3.attn_output.weight"; slice = true;
    } else if (family == "down") {
        K = 1024; fullK = 2048; N = fullN = 4096;
        name = "blk.3.ffn_down_shexp.weight"; slice = true;
    } else if (family == "pair") {
        K = fullK = 4096; N = fullN = 12288; pair = true;
        name = "blk.0.ffn_gate.weight"; name1 = "blk.0.ffn_up.weight";
    } else check(false, "family qb/gate/up/mla/down/pair");
    const char *model = getenv("DS4_GLM5_MODEL");
    check(model && *model, "model path");
    Glm5TestGGUF gguf;
    check(gguf.open_file(model), "GGUF open");
    uint64_t offsets[2] = {}, sizes[2] = {};
    check(gguf.tensor(name, {fullK, fullN}, 8, offsets[0]), "Q8 tensor");
    const uint64_t stride = (uint64_t)(fullK/32)*34;
    if (fullN != N) offsets[0] += rank*N*stride;
    sizes[0] = N*stride;
    if (pair) {
        check(gguf.tensor(name1, {fullK, fullN}, 8, offsets[1]), "Q8 pair");
        sizes[1] = N*stride;
    }
    check(ds4_gpu_init(), "GPU init");
    check(ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map), "model fd");
    check(ds4_gpu_set_model_map_spans(gguf.map, gguf.size, offsets, sizes,
                                      pair ? 2 : 1, sizes[0]), "device spans");
    unsigned *scratch = nullptr, *sink = nullptr;
    hc(hipMalloc(&scratch, 64u*1024u*1024u));
    hc(hipMalloc(&sink, 1024u*4u));
    hc(hipMemset(scratch, 1, 64u*1024u*1024u));
    auto *x = ds4_gpu_tensor_alloc((uint64_t)256*K*4);
    ds4_gpu_tensor *storage[2] = {ds4_gpu_tensor_alloc(((uint64_t)256*N+32)*4),
        pair ? ds4_gpu_tensor_alloc(((uint64_t)256*N+32)*4) : nullptr};
    check(x && storage[0] && (!pair || storage[1]), "activation buffers");
    hipEvent_t begin, end;
    hc(hipEventCreate(&begin)); hc(hipEventCreate(&end));
    printf("family=%s rank=%u K=%u fullK=%u N=%u stride=%llu placement=device-spans "
           "eviction_bytes=67108864\n", family.c_str(), rank, K, fullK, N,
           (unsigned long long)stride);
    for (unsigned M : {1u, 256u}) {
        ds4_gpu_tensor *out[2] = {ds4_gpu_tensor_view(storage[0], 64, (uint64_t)M*N*4),
            pair ? ds4_gpu_tensor_view(storage[1], 64, (uint64_t)M*N*4) : nullptr};
        check(out[0] && (!pair || out[1]), "output views");
        for (unsigned seed = 0; seed < (M == 1 ? 3u : 1u); ++seed) {
            std::vector<float> input((size_t)M*K);
            for (size_t i = 0; i < input.size(); ++i) {
                input[i] = seed == 0 ? 0.23f*cos(i*0.009) - 0.08f*sin(i*0.004) :
                    seed == 1 ? ((int)((i*193u+761u)%997u)-498)/499.0f :
                    ((i%2) ? -1.0f : 1.0f)*(i%3 ? 1e-4f : 10.0f);
            }
            check(ds4_gpu_tensor_write(x, 0, input.data(), input.size()*4), "input");
            std::vector<double> oracle[2];
            if (M == 1) for (unsigned p = 0; p < (pair ? 2u : 1u); ++p) {
                oracle[p].resize(N);
                for (unsigned n = 0; n < N; ++n) {
                    const uint8_t *w = gguf.map+offsets[p]+n*stride+
                        (slice ? rank*(K/32)*34u : 0u);
                    double dot = 0;
                    for (unsigned k = 0; k < K; ++k) {
                        uint16_t bits;
                        memcpy(&bits, w+(k/32)*34u, 2);
                        const float scale = __half2float(__ushort_as_half(bits));
                        const float weight = scale*(int8_t)w[(k/32)*34u+2u+k%32u];
                        dot += (double)weight*input[k];
                    }
                    oracle[p][n] = dot;
                }
            }
            std::vector<float> base[2];
            for (unsigned mode : {0u, 1u, 2u, 3u}) {
                // Mode3 exercises the opt-in with the GLM gate disabled.
                check(setenv("DS4_GLM5_NEXT_ENABLE_ORDINARY", mode == 3 ? "0" : "1", 1) == 0,
                      "GLM gate");
                check(setenv("DS4_ROCM_GLM5_Q8_DECODE_TILE",
                      mode == 0 ? "0" : mode == 2 ? "2" : "1", 1) == 0, "mode");
                auto launch = [&]() {
                    int ok;
                    if (pair) ok = ds4_gpu_matmul_q8_0_pair_tensor(out[0], out[1],
                        gguf.map, gguf.size, offsets[0], offsets[1], K, N, N, x, M);
                    else if (slice) ok = ds4_gpu_matmul_q8_0_kslice_rows_tensor(out[0],
                        gguf.map, gguf.size, offsets[0], fullK, N, rank*K, K, x, M);
                    else ok = ds4_gpu_matmul_q8_0_tensor(out[0], gguf.map, gguf.size,
                                                       offsets[0], K, N, x, M);
                    check(ok, "real projection launch");
                };
                std::vector<float> guarded((size_t)M*N+32, 1234567.0f);
                for (unsigned p = 0; p < (pair ? 2u : 1u); ++p)
                    check(ds4_gpu_tensor_write(storage[p], 0, guarded.data(), guarded.size()*4),
                          "canaries");
                launch(); check(ds4_gpu_synchronize(), "warm synchronize");
                if (M == 1 && seed == 0 && mode != 3) {
                    std::vector<float> times;
                    for (unsigned r = 0; r < 7; ++r) {
                        cold_read<<<1024, 256>>>(scratch, sink);
                        hc(hipGetLastError()); hc(hipEventRecord(begin));
                        launch(); hc(hipEventRecord(end)); hc(hipEventSynchronize(end));
                        float ms = 0; hc(hipEventElapsedTime(&ms, begin, end));
                        times.push_back(ms);
                    }
                    std::sort(times.begin(), times.end());
                    printf("timing M=1 mode=%u median_ms=%.6f min_ms=%.6f max_ms=%.6f "
                           "logical_GBs=%.3f\n", mode, times[3], times[0], times[6],
                           (double)N*(K/32)*34*(pair ? 2 : 1)/(times[3]*1e6));
                }
                for (unsigned p = 0; p < (pair ? 2u : 1u); ++p) {
                    check(ds4_gpu_tensor_read(storage[p], 0, guarded.data(), guarded.size()*4),
                          "full output and canaries");
                    for (unsigned i = 0; i < 16; ++i)
                        check(guarded[i] == 1234567.0f && guarded[16+(size_t)M*N+i] == 1234567.0f,
                              "output sentinel");
                    std::vector<float> got(guarded.begin()+16, guarded.end()-16);
                    for (float f : got) check(std::isfinite(f), "finite output");
                    if (mode == 0) base[p] = got;
                    // The repaired tile deliberately preserves the established
                    // production arithmetic. A lost whitelist entry must still
                    // be caught by the engagement logs and timing inventory.
                    if (M == 1 && mode == 1 && !pair) {
                        double max_diff = 0.0;
                        size_t diff_count = 0;
                        for (size_t i = 0; i < got.size(); ++i) {
                            max_diff = std::max(max_diff,
                                                fabs((double)got[i] - base[p][i]));
                            if (got[i] != base[p][i]) ++diff_count;
                        }
                        if (diff_count)
                            fprintf(stderr, "tile mismatch part=%u count=%zu max_abs=%.9g\n",
                                    p, diff_count, max_diff);
                        check(diff_count == 0, "repaired tile must preserve production output");
                    }
                    if (M == 256 || mode == 3)
                        check(memcmp(got.data(), base[p].data(), got.size()*4) == 0,
                              "M256 / non-GLM dispatch unchanged");
                    printf("output M=%u seed=%u mode=%u part=%u fnv64=%016llx\n",
                           M, seed, mode, p, (unsigned long long)hash(got));
                    if (M == 1) {
                        double e2 = 0, ref2 = 0, maxabs = 0, maxref = 0;
                        for (unsigned n = 0; n < N; ++n) {
                            const double e = got[n]-oracle[p][n];
                            e2 += e*e; ref2 += oracle[p][n]*oracle[p][n];
                            maxabs = std::max(maxabs, fabs(e));
                            maxref = std::max(maxref, fabs(oracle[p][n]));
                        }
                        printf("oracle count=%u max_abs=%.9g normalized_max=%.9g nmse=%.9g\n",
                               N, maxabs, maxabs/std::max(1.0, maxref), e2/ref2);
                        check(ref2 > 0 && maxabs <= 2e-6*std::max(1.0, maxref) &&
                              e2/ref2 <= 1e-11, "full canonical leaf envelope");
                    }
                }
                fflush(stdout);
            }
        }
        ds4_gpu_tensor_free(out[0]); if (out[1]) ds4_gpu_tensor_free(out[1]);
    }
    hc(hipEventDestroy(begin)); hc(hipEventDestroy(end));
    hc(hipFree(scratch)); hc(hipFree(sink));
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(storage[0]);
    if (storage[1]) ds4_gpu_tensor_free(storage[1]);
    ds4_gpu_cleanup(); puts("PASS original-Q8 full-oracle/canary/dispatch diagnostics");
}
