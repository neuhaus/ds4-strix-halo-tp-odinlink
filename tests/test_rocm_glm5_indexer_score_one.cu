// Differential public-API test. Link the production ds4_rocm.o (fast math),
// not a precise rebuild: contraction is part of the incumbent's behavior.
#include "ds4_gpu.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", what); std::exit(2); }
}
struct Tensor {
    ds4_gpu_tensor *p;
    explicit Tensor(size_t bytes) : p(ds4_gpu_tensor_alloc(bytes)) {
        check(p != nullptr, "allocate");
    }
    ~Tensor() { ds4_gpu_tensor_free(p); }
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;
};
static uint32_t rng = 0x5a319e2u;
static float sample() {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (float)((int32_t)(rng & 65535u) - 32768) / 32768.0f;
}

static bool run(uint32_t rows, uint32_t heads, uint32_t dim,
                bool half_cache, int pattern, bool ordinary) {
    rng = 0x5a319e2u;
    std::vector<float> q((size_t)heads * dim), w(heads), keys((size_t)rows * dim);
    for (auto &x : q) x = sample();
    for (auto &x : w) x = 0.5f + sample() * 0.25f;
    for (auto &x : keys) x = sample();
    if (pattern == 1) { // Large opposing products expose association errors.
        for (size_t i = 0; i < q.size(); ++i)
            q[i] = (i % 128u < 32u ? 4096.0f :
                   (i % 128u >= 64u && i % 128u < 96u ? -4096.0f : 0.001f));
        for (uint32_t r = 0; r < rows; ++r)
            for (uint32_t d = 0; d < dim; ++d)
                keys[(size_t)r * dim + d] = 1.0f + (float)(r % 7) * 0.03125f;
    } else if (pattern == 2) { // Exact and near ties in the selected pool set.
        for (uint32_t r = 1; r < rows; ++r) {
            std::copy_n(keys.data(), dim, keys.data() + (size_t)r * dim);
            if (r % 3 == 0)
                keys[(size_t)r * dim] = std::nextafter(keys[0], 1.0f);
        }
    } else if (pattern == 3) {
        for (size_t i = 0; i < q.size(); ++i) q[i] = i & 1u ? -0.0f : 0.0f;
    } else if (pattern == 4) {
        // Learned head weights can be negative; exercise their cancellation.
        for (auto &x : w) x = sample() * 4.0f;
    }
    std::vector<__half> half_keys(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) half_keys[i] = __float2half(keys[i]);
    Tensor dq(q.size()*4), dw(w.size()*4), dk(keys.size()*(half_cache ? 2 : 4));
    check(ds4_gpu_tensor_write(dq.p, 0, q.data(), q.size()*4) &&
          ds4_gpu_tensor_write(dw.p, 0, w.data(), w.size()*4) &&
          ds4_gpu_tensor_write(dk.p, 0, half_cache ? (const void*)half_keys.data() :
                               (const void*)keys.data(), keys.size()*(half_cache ? 2 : 4)), "inputs");
    constexpr uint32_t guard = 32;
    constexpr float canary = -1234567.0f;
    Tensor storage((rows + 2*guard)*4);
    ds4_gpu_tensor *out = ds4_gpu_tensor_view(storage.p, guard*4, rows*4);
    check(out != nullptr, "guarded output view");
    std::vector<uint32_t> valid(rows, 1u);
    if (pattern == 4)
        for (uint32_t r = 0; r < rows; r += 17u) valid[r] = 0u;
    Tensor dvalid(rows*4);
    check(ds4_gpu_tensor_write(dvalid.p, 0, valid.data(), rows*4), "pool validity");
    const uint32_t valid_count = std::count(valid.begin(), valid.end(), 1u);
    const uint32_t topk = std::min(valid_count, 512u);
    Tensor top(topk*4);
    std::vector<float> result[2];
    std::vector<uint32_t> ids[2];
    float times[2] = {};
    check(setenv("DS4_GLM5_NEXT_ENABLE_ORDINARY", ordinary ? "1" : "0", 1) == 0, "ordinary env");
    for (int mode = 0; mode < 2; ++mode) {
        check(setenv("DS4_ROCM_GLM5_INDEXER_SCORE_WARP32", mode ? "1" : "0", 1) == 0, "mode env");
        std::vector<float> guarded(rows + 2*guard, canary);
        check(ds4_gpu_tensor_write(storage.p, 0, guarded.data(), guarded.size()*4), "canaries");
        auto launch = [&]() {
            check(ds4_gpu_glm_indexer_score_one_tensor(out, dq.p, dw.p, dk.p,
                  rows, heads, dim, 0.015625f, half_cache), "score public API");
        };
        launch();
        check(ds4_gpu_synchronize(), "score synchronize");
        check(ds4_gpu_tensor_read(storage.p, 0, guarded.data(), guarded.size()*4), "score read");
        for (uint32_t g = 0; g < guard; ++g)
            check(guarded[g] == canary && guarded[guard+rows+g] == canary, "output bounds");
        result[mode].assign(guarded.begin()+guard, guarded.begin()+guard+rows);
        for (float x : result[mode]) check(std::isfinite(x), "finite scores");
        // Exercise the decode score -> invalid-pool mask -> top-k seam.
        check(ds4_gpu_glm5_mask_pool_scores_tensor(out, dvalid.p, rows), "mask score API");
        check(ds4_gpu_indexer_topk_tensor(top.p, out, rows, 1, topk) && ds4_gpu_synchronize(), "top-k API");
        ids[mode].resize(topk);
        check(ds4_gpu_tensor_read(top.p, 0, ids[mode].data(), topk*4), "top-k read");
        for (uint32_t id : ids[mode]) check(id < rows && valid[id], "selected pool is valid");
        hipEvent_t begin, end;
        check(hipEventCreate(&begin) == hipSuccess && hipEventCreate(&end) == hipSuccess, "events");
        launch();
        check(hipEventRecord(begin, 0) == hipSuccess, "event begin");
        for (int it = 0; it < 5; ++it) launch();
        check(hipEventRecord(end, 0) == hipSuccess && hipEventSynchronize(end) == hipSuccess &&
              hipEventElapsedTime(&times[mode], begin, end) == hipSuccess, "event time");
        check(hipEventDestroy(begin) == hipSuccess && hipEventDestroy(end) == hipSuccess, "event destroy");
    }
    size_t changed = 0;
    float max_error = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        changed += std::memcmp(&result[0][r], &result[1][r], sizeof(float)) != 0;
        max_error = std::max(max_error, std::fabs(result[0][r] - result[1][r]));
    }
    const bool same_ids = ids[0] == ids[1];
    std::printf("rows=%u heads=%u dim=%u f16=%d pattern=%d ordinary=%d changed=%zu max_abs=%.9g topk_equal=%d scalar_us=%.3f candidate_us=%.3f\n",
                rows, heads, dim, half_cache, pattern, ordinary, changed, max_error, same_ids,
                times[0]*200, times[1]*200);
    ds4_gpu_tensor_free(out);
    return changed == 0 && same_ids;
}

int main(int argc, char **) {
    check(ds4_gpu_init(), "GPU init");
    bool ok = run(512, 32, 128, false, 0, true);
    if (argc == 1) {
        for (uint32_t rows : {1024u, 2048u, 8192u, 8193u})
            ok = run(rows, 32, 128, false, 0, true) && ok;
        for (int pattern : {1, 2, 3, 4}) ok = run(513, 32, 128, false, pattern, true) && ok;
        ok = run(513, 32, 128, true, 0, true) && ok;
        ok = run(513, 31, 128, false, 0, true) && ok;
        ok = run(513, 32, 96, false, 0, true) && ok;
        ok = run(513, 32, 128, false, 0, false) && ok;
    }
    ds4_gpu_cleanup();
    std::puts(ok ? "PASS exact public-API score, bounds and top-k" : "FAIL score bit equality or top-k");
    return ok ? 0 : 1;
}
