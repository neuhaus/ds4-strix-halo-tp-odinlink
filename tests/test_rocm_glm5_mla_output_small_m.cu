// Original MLA output Q8_0: preserve full source stride and scalar bit pattern.
#include "ds4_glm5_next_exec.h"
#include "ds4_gpu_mgpu.h"
extern "C" {
#include "ds4_tp.h"
}
#include "glm5_gguf_test.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}
#define REQUIRE(x) do { if (!(x)) { \
    std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); \
} } while (0)

struct Guarded {
    ds4_gpu_tensor *storage, *view;
    uint64_t count;
    explicit Guarded(uint64_t n) : storage(ds4_gpu_tensor_alloc((n + 32u) * 4u)),
        view(storage ? ds4_gpu_tensor_view(storage, 64u, n * 4u) : nullptr), count(n) {
        REQUIRE(view);
    }
    ~Guarded() { ds4_gpu_tensor_free(view); ds4_gpu_tensor_free(storage); }
    void poison() { REQUIRE(ds4_gpu_tensor_fill_f32(storage, 12345.0f, count + 32u)); }
    std::vector<float> read() {
        std::vector<float> all(count + 32u);
        REQUIRE(ds4_gpu_tensor_read(storage, 0u, all.data(), all.size() * 4u));
        for (unsigned i = 0u; i < 16u; ++i)
            REQUIRE(all[i] == 12345.0f && all[count + 16u + i] == 12345.0f);
        for (uint64_t i = 16u; i < count + 16u; ++i) REQUIRE(std::isfinite(all[i]));
        return {all.begin() + 16, all.end() - 16};
    }
};

static std::vector<float> inputs(unsigned m, unsigned layer, unsigned seed) {
    std::vector<float> result((uint64_t)m * 8192u);
    for (uint64_t i = 0; i < result.size(); ++i) {
        const unsigned token = i / 8192u, column = i % 8192u;
        const float fraction = ((int)((column * 193u + token * 761u + layer * 31u) % 997u) - 498) /
            (1001.3f + column % 7u);
        if (seed == 1u)
            result[i] = (i & 1u ? -1.0f : 1.0f) * (i % 3u ? 1e-4f : 10.0f);
        else if (seed == 2u || (seed >= 3u && seed < 11u && column / 1024u == seed - 3u) ||
                 (seed >= 11u && token == seed - 11u))
            result[i] = fraction;
    }
    return result;
}

int main(int argc, char **argv) {
    REQUIRE(argc == 2 && (!std::strcmp(argv[1], "0") || !std::strcmp(argv[1], "1")));
    const unsigned rank = argv[1][0] - '0';
    const char *path = std::getenv("DS4_GLM5_MODEL"); REQUIRE(path);
    REQUIRE(std::getenv("DS4_ROCM_GLM5_Q8_DECODE_TILE") &&
        !std::strcmp(std::getenv("DS4_ROCM_GLM5_Q8_DECODE_TILE"), "1"));
    REQUIRE(std::getenv("DS4_ROCM_GLM5_Q8_SHAREDX_PREFETCH") &&
        !std::strcmp(std::getenv("DS4_ROCM_GLM5_Q8_SHAREDX_PREFETCH"), "8"));
    REQUIRE(setenv("DS4_GLM5_NEXT_ENABLE_ORDINARY", "1", 1) == 0);
    Glm5TestGGUF g; REQUIRE(g.open_file(path));
    uint64_t offsets[11], sizes[11];
    for (unsigned index = 0u; index < 11u; ++index) {
        char name[80]; std::snprintf(name, sizeof(name), "blk.%u.attn_output.weight", 3u + index * 4u);
        REQUIRE(g.tensor(name, {16384u, 4096u}, 8u, offsets[index]));
        sizes[index] = 4096u * 17408u;
    }
    REQUIRE(ds4_gpu_init());
    ds4_gpu_set_glm_model(true); ds4_gpu_set_q8_cache_suppressed(1);
    REQUIRE(ds4_gpu_set_model_fd_for_map(g.fd, g.map));
    REQUIRE(ds4_gpu_set_model_map_spans(g.map, g.size, offsets, sizes, 11u, sizes[0]));
    uint64_t compared = 0u;
    unsigned cases = 0u, refusals = 0u;
    for (unsigned index = 0u; index < 11u; ++index) for (unsigned m : {2u, 4u, 6u}) {
        const unsigned layer = 3u + index * 4u;
        Guarded x((uint64_t)m * 8192u), out((uint64_t)m * 4096u);
        std::vector<ds4_gpu_tensor *> xv(m), ov(m);
        for (unsigned t = 0u; t < m; ++t) {
            xv[t] = ds4_gpu_tensor_view(x.view, (uint64_t)t * 8192u * 4u, 8192u * 4u);
            ov[t] = ds4_gpu_tensor_view(out.view, (uint64_t)t * 4096u * 4u, 4096u * 4u);
            REQUIRE(xv[t] && ov[t]);
        }
        auto run = [&](bool batch) {
            if (batch) return ds4_rocm_glm5_mla_output_q8_small_m(out.view, g.map, g.size,
                offsets[index], 16384u, rank * 8192u, 8192u, 4096u, 17408u, x.view, m);
            for (unsigned t = 0u; t < m; ++t)
                if (!ds4_gpu_matmul_q8_0_kslice_tensor(ov[t], g.map, g.size, offsets[index],
                    16384u, rank * 8192u, 8192u, 4096u, xv[t], 0u)) return 0;
            return 1;
        };
        for (unsigned seed = 0u; seed < 11u + m; ++seed) {
            auto host = inputs(m, layer, seed);
            x.poison(); REQUIRE(ds4_gpu_tensor_write(x.view, 0u, host.data(), host.size() * 4u));
            out.poison(); REQUIRE(run(false) && ds4_gpu_synchronize());
            const auto reference = out.read();
            REQUIRE(std::none_of(reference.begin(), reference.end(), [](float v) { return v == 12345.0f; }));
            REQUIRE(!seed || std::any_of(reference.begin(), reference.end(), [](float v) { return v != 0.0f; }));
            out.poison(); REQUIRE(run(true) && ds4_gpu_synchronize());
            const auto got = out.read(), kept = x.read();
            REQUIRE(!std::memcmp(host.data(), kept.data(), host.size() * 4u));
            for (uint64_t i = 0u; i < got.size(); ++i) {
                if (std::memcmp(&got[i], &reference[i], 4u)) {
                    std::fprintf(stderr, "DIFF layer=%u rank=%u m=%u seed=%u index=%llu scalar=%.9g batch=%.9g\n",
                        layer, rank, m, seed, (unsigned long long)i, reference[i], got[i]);
                    return 1;
                }
            }
            compared += got.size(); ++cases;
        }
        std::printf("MLA_OUTPUT_EXACT layer=%u rank=%u m=%u seeds=%u PASS\n", layer, rank, m, 11u + m);
        std::fflush(stdout);
        if (index == 0u) {
            auto refuse = [&](ds4_gpu_tensor *dst, const ds4_gpu_tensor *src,
                              uint64_t off, unsigned full, unsigned first,
                              unsigned k, unsigned n, uint64_t stride, unsigned tokens) {
                out.poison();
                REQUIRE(!ds4_rocm_glm5_mla_output_q8_small_m_supported(dst, g.map, g.size,
                    off, full, first, k, n, stride, src, tokens));
                REQUIRE(!ds4_rocm_glm5_mla_output_q8_small_m(dst, g.map, g.size,
                    off, full, first, k, n, stride, src, tokens));
                REQUIRE(ds4_gpu_synchronize());
                for (float v : out.read()) REQUIRE(v == 12345.0f);
                ++refusals;
            };
            auto reject_mode = [&]() {
                refuse(out.view, x.view, offsets[index], 16384u, rank * 8192u, 8192u, 4096u, 17408u, m);
            };
            for (unsigned tokens : {0u, 1u, 3u, 5u, 7u, 8u, UINT32_MAX})
                refuse(out.view, x.view, offsets[index], 16384u, rank * 8192u, 8192u, 4096u, 17408u, tokens);
            for (unsigned first : {1u, 32u, 4096u, 16384u, UINT32_MAX})
                refuse(out.view, x.view, offsets[index], 16384u, first, 8192u, 4096u, 17408u, m);
            for (uint64_t off : {offsets[index] + 1u, (g.size - 2u) & ~UINT64_C(1), UINT64_MAX - 1u})
                refuse(out.view, x.view, off, 16384u, rank * 8192u, 8192u, 4096u, 17408u, m);
            refuse(out.view, x.view, offsets[index], 8192u, rank * 8192u, 8192u, 4096u, 17408u, m);
            refuse(out.view, x.view, offsets[index], 16384u, rank * 8192u, 4096u, 4096u, 17408u, m);
            refuse(out.view, x.view, offsets[index], 16384u, rank * 8192u, 8192u, 4095u, 17408u, m);
            refuse(out.view, x.view, offsets[index], 16384u, rank * 8192u, 8192u, 4096u, 8704u, m);
            auto *short_x = ds4_gpu_tensor_view(x.view, 0u, x.count * 4u - 4u);
            auto *short_out = ds4_gpu_tensor_view(out.view, 0u, out.count * 4u - 4u);
            auto *overlap = ds4_gpu_tensor_view(x.storage, 68u, out.count * 4u);
            REQUIRE(short_x && short_out && overlap);
            refuse(out.view, short_x, offsets[index], 16384u, rank * 8192u, 8192u, 4096u, 17408u, m);
            refuse(short_out, x.view, offsets[index], 16384u, rank * 8192u, 8192u, 4096u, 17408u, m);
            refuse(x.view, x.view, offsets[index], 16384u, rank * 8192u, 8192u, 4096u, 17408u, m);
            refuse(overlap, x.view, offsets[index], 16384u, rank * 8192u, 8192u, 4096u, 17408u, m);
            ds4_gpu_tensor_free(overlap); ds4_gpu_tensor_free(short_out); ds4_gpu_tensor_free(short_x);
            const auto kept = x.read(), host = inputs(m, layer, 10u + m);
            REQUIRE(!std::memcmp(kept.data(), host.data(), host.size() * 4u));
            Guarded touching(out.count + x.count);
            touching.poison();
            auto *reverse_x = ds4_gpu_tensor_view(touching.view, 4u, x.count * 4u);
            REQUIRE(reverse_x);
            refuse(touching.view, reverse_x, offsets[index], 16384u, rank * 8192u, 8192u, 4096u, 17408u, m);
            for (float v : touching.read()) REQUIRE(v == 12345.0f);
            ds4_gpu_tensor_free(reverse_x);
            // An unrelated host mapping must refuse, not lazily allocate weights.
            REQUIRE(!ds4_rocm_glm5_mla_output_q8_small_m_supported(out.view, g.map + 1u, g.size - 1u,
                offsets[index], 16384u, rank * 8192u, 8192u, 4096u, 17408u, x.view, m));
            REQUIRE(!ds4_rocm_glm5_mla_output_q8_small_m(out.view, g.map + 1u, g.size - 1u,
                offsets[index], 16384u, rank * 8192u, 8192u, 4096u, 17408u, x.view, m));
            ++refusals;
            for (const char *mode : {"0", "2", "3", "4", "5", "6", "invalid"}) {
                REQUIRE(setenv("DS4_ROCM_GLM5_Q8_DECODE_TILE", mode, 1) == 0); reject_mode();
            }
            REQUIRE(setenv("DS4_ROCM_GLM5_Q8_DECODE_TILE", "1", 1) == 0);
            REQUIRE(unsetenv("DS4_GLM5_NEXT_ENABLE_ORDINARY") == 0); reject_mode();
            REQUIRE(setenv("DS4_GLM5_NEXT_ENABLE_ORDINARY", "1", 1) == 0);
            REQUIRE(setenv("DS4_ROCM_GLM5_Q8_SHAREDX_PREFETCH", "0", 1) == 0); reject_mode();
            REQUIRE(setenv("DS4_ROCM_GLM5_Q8_SHAREDX_PREFETCH", "8", 1) == 0);
            ds4_gpu_set_quality(true); reject_mode(); ds4_gpu_set_quality(false);
            REQUIRE(setenv("DS4_ROCM_GLM5_Q8_SHAREDX_NONTEMPORAL", "invalid", 1) == 0); reject_mode();
            REQUIRE(setenv("DS4_ROCM_GLM5_Q8_SHAREDX_NONTEMPORAL", "1", 1) == 0);
            REQUIRE(setenv("DS4_ROCM_GLM5_Q8_SHAREDX_ROWS_PER_BLOCK", "7", 1) == 0); reject_mode();
            REQUIRE(setenv("DS4_ROCM_GLM5_Q8_SHAREDX_ROWS_PER_BLOCK", "32", 1) == 0);

            const auto timed_input = inputs(m, layer, 2u);
            REQUIRE(ds4_gpu_tensor_write(x.view, 0u, timed_input.data(), timed_input.size() * 4u));
            REQUIRE(run(false) && ds4_gpu_synchronize());
            const auto adjacent_reference = out.read();
            auto *adjacent_out = ds4_gpu_tensor_view(touching.view, 0u, out.count * 4u);
            auto *adjacent_x = ds4_gpu_tensor_view(touching.view, out.count * 4u, x.count * 4u);
            REQUIRE(adjacent_out && adjacent_x &&
                ds4_gpu_tensor_write(adjacent_x, 0u, timed_input.data(), x.count * 4u));
            REQUIRE(ds4_rocm_glm5_mla_output_q8_small_m_supported(adjacent_out, g.map, g.size,
                offsets[index], 16384u, rank * 8192u, 8192u, 4096u, 17408u, adjacent_x, m));
            REQUIRE(ds4_rocm_glm5_mla_output_q8_small_m(adjacent_out, g.map, g.size,
                offsets[index], 16384u, rank * 8192u, 8192u, 4096u, 17408u, adjacent_x, m) && ds4_gpu_synchronize());
            const auto adjacent = touching.read();
            REQUIRE(!std::memcmp(adjacent.data(), adjacent_reference.data(), out.count * 4u));
            REQUIRE(!std::memcmp(adjacent.data() + out.count, timed_input.data(), x.count * 4u));
            ds4_gpu_tensor_free(adjacent_out); ds4_gpu_tensor_free(adjacent_x);
            ++cases; compared += out.count;
            REQUIRE(run(true) && ds4_gpu_synchronize());
            hipEvent_t begin, end;
            REQUIRE(hipEventCreate(&begin) == hipSuccess && hipEventCreate(&end) == hipSuccess);
            std::vector<double> gpu[2], wall[2];
            for (unsigned sample = 0u; sample < 13u; ++sample) for (unsigned turn = 0u; turn < 2u; ++turn) {
                const unsigned arm = turn ^ (sample & 1u);
                const auto started = std::chrono::steady_clock::now();
                REQUIRE(hipEventRecord(begin) == hipSuccess && run(arm != 0u));
                REQUIRE(hipEventRecord(end) == hipSuccess && hipEventSynchronize(end) == hipSuccess);
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
                float device_ms = 0.0f; REQUIRE(hipEventElapsedTime(&device_ms, begin, end) == hipSuccess);
                if (sample >= 4u) {
                    gpu[arm].push_back(device_ms); wall[arm].push_back(ms);
                    std::printf("MLA_OUTPUT_SAMPLE rank=%u m=%u sample=%u batch=%u gpu_ms=%.6f wall_ms=%.6f\n",
                        rank, m, sample - 4u, arm, device_ms, ms);
                }
            }
            for (auto &v : gpu) std::sort(v.begin(), v.end());
            for (auto &v : wall) std::sort(v.begin(), v.end());
            std::printf("MLA_OUTPUT_MEDIAN rank=%u m=%u scalar_gpu_ms=%.6f batch_gpu_ms=%.6f scalar_wall_ms=%.6f batch_wall_ms=%.6f\n",
                rank, m, gpu[0][4], gpu[1][4], wall[0][4], wall[1][4]);
            REQUIRE(hipEventDestroy(begin) == hipSuccess && hipEventDestroy(end) == hipSuccess);
        }
        for (unsigned t = 0u; t < m; ++t) { ds4_gpu_tensor_free(xv[t]); ds4_gpu_tensor_free(ov[t]); }
    }
    REQUIRE(cases == 498u && compared == 8519680u && refusals == 111u);
    std::printf("PASS MLA output rank=%u cases=%u compared_floats=%llu refusals=%u weights=original widths=2/4/6 tile=1 prefetch=8\n",
        rank, cases, (unsigned long long)compared, refusals);
    ds4_gpu_cleanup();
}
