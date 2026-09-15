/* Real-GGUF Q8 shape probe.
 *
 * This is deliberately a bounded, cache-free-at-the-harness level probe: it
 * passes mapped weight pointers to the existing backend entry points and
 * never creates a concatenated projection tensor.  It reports the two input
 * regimes used by the GLM-5 graph (decode M=1 and prefill M=256), so an
 * aggregate projection stream cannot hide a shape-specific dispatch loss.
 */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "tests/glm5_gguf_test.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(expr, message) do {                                         \
    if (!(expr)) {                                                        \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__); \
        return 1;                                                         \
    }                                                                     \
} while (0)

namespace {

enum class Kind { Matmul, Pair, PairFused, Kslice, KsliceStrided };

struct Shape {
    const char *name;
    const char *tensor0;
    const char *tensor1;
    std::vector<uint64_t> dims;
    Kind kind;
    uint64_t in_start;
    uint64_t in_count;
    uint64_t out_start;
    uint64_t out_count;
    std::vector<uint64_t> offsets0;
    std::vector<uint64_t> offsets1;
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2u];
}

uint64_t fnv64(const std::vector<float> &values) {
    uint64_t hash = UINT64_C(1469598103934665603);
    const auto *bytes = reinterpret_cast<const unsigned char *>(values.data());
    for (size_t i = 0; i < values.size() * sizeof(float); ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

bool bind_layer(const Glm5TestGGUF &gguf, Shape &shape, uint32_t layer) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    uint64_t a = 0u;
    if (!gguf.tensor(prefix + shape.tensor0, shape.dims, 8u, a)) return false;
    shape.offsets0.push_back(a);
    if (shape.kind == Kind::Pair || shape.kind == Kind::PairFused) {
        uint64_t b = 0u;
        if (!gguf.tensor(prefix + shape.tensor1, shape.dims, 8u, b)) {
            shape.offsets0.pop_back();
            return false;
        }
        shape.offsets1.push_back(b);
    }
    return true;
}

uint64_t input_width(const Shape &shape) {
    return shape.kind == Kind::Kslice || shape.kind == Kind::KsliceStrided
        ? shape.in_count : shape.dims[0];
}

uint64_t output_width(const Shape &shape) {
    if (shape.kind == Kind::Pair || shape.kind == Kind::PairFused)
        return shape.out_count;
    return shape.kind == Kind::Kslice || shape.kind == Kind::KsliceStrided
        ? shape.out_count : shape.dims[1];
}

uint64_t storage_input_width(const Shape &shape) {
    return shape.kind == Kind::KsliceStrided ? shape.dims[0] : input_width(shape);
}

}  // namespace

int main() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    CHECK(model && model[0], "DS4_GLM5_MODEL is required");
    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF");

    std::vector<Shape> shapes = {
        {"q_b", "attn_q_b.weight", nullptr, {1536u, 16384u},
         Kind::Matmul, 0u, 0u, 0u, 0u},
        {"kv_a", "attn_kv_a_mqa.weight", nullptr, {4096u, 512u},
         Kind::Matmul, 0u, 0u, 0u, 0u},
        {"shared_gate_up", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
         {4096u, 2048u}, Kind::Pair, 0u, 0u, 0u, 1024u},
        {"shared_gate_up_fused", "ffn_gate_shexp.weight",
         "ffn_up_shexp.weight", {4096u, 2048u}, Kind::PairFused,
         0u, 0u, 0u, 1024u},
        {"shared_down_khalf", "ffn_down_shexp.weight", nullptr,
         {2048u, 4096u}, Kind::Kslice, 0u, 1024u, 0u, 4096u},
        {"mla_output_khalf", "attn_output.weight", nullptr,
         {16384u, 4096u}, Kind::KsliceStrided, 0u, 8192u, 0u, 4096u},
    };
    for (Shape &shape : shapes) {
        for (uint32_t layer = 0u; layer < 45u; ++layer)
            (void)bind_layer(gguf, shape, layer);
        CHECK(!shape.offsets0.empty(), "bind at least one real Q8 layer");
        if (shape.kind == Kind::Pair || shape.kind == Kind::PairFused)
            CHECK(shape.offsets0.size() == shape.offsets1.size(),
                  "bind paired Q8 layers");
        std::printf("shape,%s,layers,%zu,in,%llu,out,%llu\n", shape.name,
                    shape.offsets0.size(),
                    (unsigned long long)input_width(shape),
                    (unsigned long long)output_width(shape));
    }

    ds4_gpu_config config = {};
    config.n_gpus = 1u;
    config.device_indices[0] = 0u;
    CHECK(ds4_gpu_init_multi(&config) &&
          ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map) &&
          ds4_gpu_set_model_map(gguf.map, gguf.size),
          "initialize GPU and register model map");

    uint64_t max_in = 0u, max_out = 0u;
    for (const Shape &shape : shapes) {
        max_in = std::max(max_in, storage_input_width(shape));
        max_out = std::max(max_out, output_width(shape));
    }
    constexpr uint32_t kMaxRows = 256u;
    ds4_gpu_tensor *input = ds4_gpu_tensor_alloc(
        (uint64_t)kMaxRows * max_in * sizeof(float));
    ds4_gpu_tensor *output0 = ds4_gpu_tensor_alloc(
        (uint64_t)kMaxRows * max_out * sizeof(float));
    ds4_gpu_tensor *output1 = ds4_gpu_tensor_alloc(
        (uint64_t)kMaxRows * max_out * sizeof(float));
    ds4_gpu_tensor *output2 = ds4_gpu_tensor_alloc(
        (uint64_t)kMaxRows * max_out * sizeof(float));
    CHECK(input && output0 && output1 && output2,
          "allocate bounded probe buffers");
    std::vector<float> host((size_t)kMaxRows * max_in);
    for (size_t i = 0u; i < host.size(); ++i) {
        const int value = (int)((i * 73u + (i >> 4u) * 29u) % 1021u) - 510;
        host[i] = (float)value / 1024.0f;
    }
    CHECK(ds4_gpu_tensor_write(input, 0u, host.data(),
                               host.size() * sizeof(float)),
          "upload deterministic probe input");

    const auto run = [&](const Shape &shape, size_t layer, uint32_t rows,
                         bool production_stride) -> bool {
        const uint64_t in = input_width(shape);
        const uint64_t out = output_width(shape);
        ds4_gpu_tensor *xview = ds4_gpu_tensor_view(
            input, 0u, (uint64_t)rows *
            (shape.kind == Kind::KsliceStrided && production_stride ?
             shape.dims[0] : in) * sizeof(float));
        ds4_gpu_tensor *o0view = ds4_gpu_tensor_view(
            output0, 0u, (uint64_t)rows * out * sizeof(float));
        ds4_gpu_tensor *o1view = ds4_gpu_tensor_view(
            output1, 0u, (uint64_t)rows * out * sizeof(float));
        ds4_gpu_tensor *o2view = ds4_gpu_tensor_view(
            output2, 0u, (uint64_t)rows * out * sizeof(float));
        if (!xview || !o0view || !o1view || !o2view) {
            ds4_gpu_tensor_free(xview);
            ds4_gpu_tensor_free(o0view);
            ds4_gpu_tensor_free(o1view);
            ds4_gpu_tensor_free(o2view);
            return false;
        }
        bool ok = false;
        if (shape.kind == Kind::Matmul) {
            ok = ds4_gpu_matmul_q8_0_tensor(
                o0view, gguf.map, gguf.size, shape.offsets0[layer],
                shape.dims[0], shape.dims[1], xview, rows) != 0;
        } else if (shape.kind == Kind::Pair || shape.kind == Kind::PairFused) {
            const uint64_t row_bytes = (shape.dims[0] / 32u) * 34u;
            const uint64_t row_offset =
                (uint64_t)shape.out_start * row_bytes;
            if (shape.kind == Kind::PairFused) {
                ok = ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
                    o0view, o1view, o2view, gguf.map, gguf.size,
                    shape.offsets0[layer] + row_offset,
                    shape.offsets1[layer] + row_offset,
                    shape.dims[0], shape.out_count, xview, rows, 10.0f) != 0;
            } else {
                ok = ds4_gpu_matmul_q8_0_pair_tensor(
                    o0view, o1view, gguf.map, gguf.size,
                    shape.offsets0[layer] + row_offset,
                    shape.offsets1[layer] + row_offset,
                    shape.dims[0], shape.out_count, shape.out_count,
                    xview, rows) != 0;
                if (ok) {
                    ok = ds4_gpu_swiglu_tensor(
                        o2view, o0view, o1view, (uint32_t)(rows * out),
                        10.0f, 1.0f) != 0;
                }
            }
        } else if (shape.kind == Kind::KsliceStrided && production_stride && rows > 1u) {
            ok = ds4_rocm_q8_kslice_f32_rows_strided(
                       o0view, gguf.map, gguf.size, shape.offsets0[layer],
                       shape.dims[0], shape.out_count, shape.in_start,
                       shape.in_count, xview, shape.in_start, rows,
                       shape.dims[0]) > 0;
        } else {
            ok = ds4_gpu_matmul_q8_0_kslice_rows_tensor(
                   o0view, gguf.map, gguf.size, shape.offsets0[layer],
                   shape.dims[0], shape.out_count, shape.in_start,
                   shape.in_count, xview, rows) != 0;
        }
        ds4_gpu_tensor_free(xview);
        ds4_gpu_tensor_free(o0view);
        ds4_gpu_tensor_free(o1view);
        ds4_gpu_tensor_free(o2view);
        return ok;
    };

    const uint32_t rows_list[] = {1u, 256u};
    constexpr uint32_t kSamples = 5u;
    constexpr uint32_t kRepeats = 2u;
    for (Shape &shape : shapes) {
        for (uint32_t rows : rows_list) {
            const bool stride_variant =
                shape.kind == Kind::KsliceStrided && rows > 1u;
            for (size_t i = 0u; i < shape.offsets0.size(); ++i) {
                CHECK(run(shape, i, rows, stride_variant) &&
                      ds4_gpu_synchronize(), "warmup shape launch");
            }
            std::vector<float> first((size_t)rows * output_width(shape));
            ds4_gpu_tensor *hash_tensor =
                (shape.kind == Kind::Pair || shape.kind == Kind::PairFused) ?
                output2 : output0;
            CHECK(ds4_gpu_tensor_read(hash_tensor, 0u, first.data(),
                                      first.size() * sizeof(float)),
                  "read shape output");
            const uint64_t first_hash = fnv64(first);
            for (size_t i = 0u; i < shape.offsets0.size(); ++i)
                CHECK(run(shape, i, rows, stride_variant),
                      "repeat shape launch");
            CHECK(ds4_gpu_synchronize(), "synchronize repeat shape launch");
            std::vector<float> second((size_t)rows * output_width(shape));
            CHECK(ds4_gpu_tensor_read(hash_tensor, 0u, second.data(),
                                      second.size() * sizeof(float)),
                  "read repeated shape output");
            CHECK(std::memcmp(first.data(), second.data(),
                              first.size() * sizeof(float)) == 0,
                  "shape output repeatability");

            std::vector<double> samples;
            for (uint32_t sample = 0u; sample < kSamples; ++sample) {
                CHECK(ds4_gpu_synchronize(), "synchronize before shape timing");
                const auto begin = std::chrono::steady_clock::now();
                for (uint32_t repeat = 0u; repeat < kRepeats; ++repeat)
                    for (size_t i = 0u; i < shape.offsets0.size(); ++i)
                        CHECK(run(shape, i, rows, stride_variant),
                              "timed shape launch");
                CHECK(ds4_gpu_synchronize(), "synchronize after shape timing");
                const auto end = std::chrono::steady_clock::now();
                samples.push_back(std::chrono::duration<double, std::milli>(
                                      end - begin).count() /
                                  (double)kRepeats);
            }
            const double ms = median(samples);
            const double layer_ms = ms / (double)shape.offsets0.size();
            const double tps = (double)rows * 1000.0 / layer_ms;
            std::printf("result,shape=%s,rows=%u,layers=%zu,ms_stream=%.6f,"
                        "ms_layer=%.6f,effective_tps=%.3f,fnv=%016llx,"
                        "strided=%d\n",
                        shape.name, rows, shape.offsets0.size(), ms, layer_ms,
                        tps, (unsigned long long)first_hash,
                        stride_variant ? 1 : 0);
        }
    }

    ds4_gpu_tensor_free(output1);
    ds4_gpu_tensor_free(output0);
    ds4_gpu_tensor_free(output2);
    ds4_gpu_tensor_free(input);
    ds4_gpu_cleanup();
    return 0;
}
