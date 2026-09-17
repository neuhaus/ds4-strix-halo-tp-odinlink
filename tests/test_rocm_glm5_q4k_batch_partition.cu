// Compare the real packed-half production entry with independent <=M256
// calls, including the final partial group. Rare routes change hot/cold
// occupancy if an outer batch is accidentally treated as one group.
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
extern "C" {
#include "ds4_tp.h"
}
#include "glm5_gguf_test.hpp"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}
#define REQUIRE(x) do { if (!(x)) { \
    std::fprintf(stderr, "FAIL line=%d: %s\n", __LINE__, #x); \
    std::exit(1); } } while (0)

int main(int argc, char **argv) {
    constexpr uint32_t width = 4096, mid_width = 1024;
    uint32_t rows = 1024;
    REQUIRE(argc <= 3);
    const bool cold_lds5 = argc == 3;
    if (cold_lds5) REQUIRE(std::strcmp(argv[2], "--cold-lds5") == 0);
    if (argc >= 2) {
        char *end = nullptr;
        const unsigned long count = std::strtoul(argv[1], &end, 10);
        REQUIRE(end != argv[1] && *end == '\0' && count > 0 && count <= 1024);
        rows = uint32_t(count);
    }
    constexpr uint32_t experts = 288, used = 8, tile = 256;
    constexpr uint64_t gate_row = 16u * 144u, down_row = 8u * 144u;
    const char *model = std::getenv("DS4_GLM5_MODEL");
    REQUIRE(model);
    Glm5TestGGUF gguf;
    REQUIRE(gguf.open_file(model));
    uint64_t go, uo, down_offset;
    REQUIRE(gguf.tensor("blk.3.ffn_gate_exps.weight", {4096,2048,288},12,go));
    REQUIRE(gguf.tensor("blk.3.ffn_up_exps.weight", {4096,2048,288},12,uo));
    REQUIRE(gguf.tensor("blk.3.ffn_down_exps.weight", {2048,4096,288},12,down_offset));
    REQUIRE(setenv("DS4_ROCM_Q4K_KSHARD_RESEARCH", "1", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_Q4K_WMMA", "1", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_DISABLE_Q4K_WMMA", "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_Q4K_WMMA_MIN_COUNT", "6", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_Q4K_WMMA_PAIR_GATE_UP", "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_Q4K_WMMA_FUSE_MID", "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_Q4K_COLD_TILE4", "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_COLD_LDS5", "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_TP_PREFILL_SKIP_UNOWNED", "1", 1) == 0);
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    REQUIRE(ds4_gpu_init_multi(&config));
    REQUIRE(ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map));
    REQUIRE(ds4_gpu_set_model_map(gguf.map, gguf.size));
    ds4_gpu_set_tp_runtime_features(0, DS4_TP_FEATURE_Q4K_WMMA |
                                      DS4_TP_FEATURE_Q4K_KSHARD);

    // out, gate, up, mid, down, selected, routing weights, input.
    const uint64_t strides[] = {width*4u, used*mid_width*4u,
        used*mid_width*4u, used*mid_width*4u, used*width*4u,
        used*4u, used*4u, width*4u};
    ds4_gpu_tensor *t[8];
    for (unsigned i=0; i<8; ++i) {
        t[i] = ds4_gpu_tensor_alloc(rows*strides[i]);
        REQUIRE(t[i]);
    }
    std::vector<float> x(rows*width), weights(rows*used);
    std::vector<int32_t> selected(rows*used);
    for (size_t i=0; i<x.size(); ++i)
        x[i] = float(std::sin(double(i)*0.013)*0.17 +
                     std::cos(double(i)*0.031)*0.11);
    for (unsigned r=0; r<rows; ++r) for (unsigned s=0; s<used; ++s) {
        selected[r*used+s] = s<7 ? int(s) : 7+int(r%128);
        if (cold_lds5 && s == 7u) {
            // Each M256 domain includes experts used 1,2,...,7 times,
            // exercising every cold count and both sides of threshold6.
            unsigned within = (r % tile) % 28u, group = 0u;
            while (within >= group + 1u) within -= ++group;
            selected[r*used+s] = int(7u + ((r % tile) / 28u) * 7u + group);
        }
        weights[r*used+s] = float(1+(r+s)%7)/16.0f;
    }
    REQUIRE(ds4_gpu_tensor_write(t[7],0,x.data(),rows*strides[7]));
    REQUIRE(ds4_gpu_tensor_write(t[5],0,selected.data(),rows*strides[5]));
    REQUIRE(ds4_gpu_tensor_write(t[6],0,weights.data(),rows*strides[6]));
    std::vector<float> reference(rows*width), actual(rows*width);
    for (unsigned rank=0; rank<2; ++rank) {
        for (uint64_t offset : {go, uo}) {
            REQUIRE(ds4_gpu_q4k_packed_slice_declare(gguf.map,gguf.size,
                offset,experts,2048,gate_row,rank*mid_width,mid_width,
                0,gate_row,DS4_GPU_Q4K_PACKED_ROW_RANGE));
            REQUIRE(ds4_gpu_q4k_packed_slice_load(gguf.map,offset,
                rank*mid_width,mid_width,0,gate_row));
        }
        REQUIRE(ds4_gpu_q4k_packed_slice_declare(gguf.map,gguf.size,
            down_offset,experts,width,down_row,0,width,rank*down_row/2,
            down_row/2,DS4_GPU_Q4K_PACKED_K_RANGE));
        REQUIRE(ds4_gpu_q4k_packed_slice_load(gguf.map,down_offset,
            0,width,rank*down_row/2,down_row/2));
        auto call = [&](ds4_gpu_tensor **v, unsigned count) {
            bool half_mid = true;
            const int ok = ds4_gpu_routed_moe_batch_packed_q4k_tensor(
                v[0],v[1],v[2],v[3],v[4],gguf.map,gguf.size,go,uo,
                down_offset,experts,gate_row,down_row,rank*mid_width,
                mid_width,rank*down_row/2,down_row/2,v[5],v[6],used,
                10.0f,v[7],3,count,&half_mid);
            return ok && !half_mid;
        };
        REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_COLD_LDS5", "0", 1) == 0);
        REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_PARTITION", "0", 1) == 0);
        for (unsigned first=0; first<rows; first+=tile) {
            const unsigned count = rows-first < tile ? rows-first : tile;
            ds4_gpu_tensor *v[8];
            for (unsigned i=0; i<8; ++i) {
                v[i] = ds4_gpu_tensor_view(t[i],first*strides[i],count*strides[i]);
                REQUIRE(v[i]);
            }
            REQUIRE(call(v,count));
            for (auto *view:v) ds4_gpu_tensor_free(view);
        }
        REQUIRE(ds4_gpu_tensor_read(t[0],0,reference.data(),rows*strides[0]));
        REQUIRE(ds4_gpu_tensor_fill_f32(t[0],NAN,rows*width));
        REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_PARTITION", "256", 1) == 0);
        REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_COLD_LDS5", cold_lds5 ? "1" : "0", 1) == 0);
        REQUIRE(call(t,rows));
        REQUIRE(ds4_gpu_tensor_read(t[0],0,actual.data(),rows*strides[0]));
        size_t different=0;
        double max_abs=0;
        for (size_t i=0; i<actual.size(); ++i) {
            REQUIRE(std::isfinite(reference[i]) && std::isfinite(actual[i]));
            different += std::memcmp(&reference[i],&actual[i],sizeof(float)) != 0;
            max_abs = std::fmax(max_abs,std::fabs(double(reference[i])-actual[i]));
        }
        std::printf("rank=%u rows=%u values=%zu different=%zu max_abs=%.9g\n",
                    rank,rows,actual.size(),different,max_abs);
        std::fflush(stdout);
        REQUIRE(different == 0);
        if (cold_lds5) {
            // Warm both full-MoE arms before timing. These are GPU-event
            // microbenchmarks with synthetic routes, not model throughput.
            hipEvent_t begin, end;
            REQUIRE(hipEventCreate(&begin) == hipSuccess);
            REQUIRE(hipEventCreate(&end) == hipSuccess);
            for (const char *mode : {"0", "1"}) {
                REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_COLD_LDS5", mode, 1) == 0);
                REQUIRE(call(t,rows));
            }
            REQUIRE(ds4_gpu_synchronize());
            for (unsigned pair = 0; pair < 3u; ++pair) {
                for (unsigned arm = 0; arm < 2u; ++arm) {
                    const unsigned mode = arm ^ (pair & 1u);
                    REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_COLD_LDS5", mode ? "1" : "0", 1) == 0);
                    REQUIRE(hipEventRecord(begin, nullptr) == hipSuccess);
                    for (unsigned repeat = 0; repeat < 5u; ++repeat)
                        REQUIRE(call(t,rows));
                    REQUIRE(hipEventRecord(end, nullptr) == hipSuccess);
                    REQUIRE(hipEventSynchronize(end) == hipSuccess);
                    float ms = 0.0f;
                    REQUIRE(hipEventElapsedTime(&ms, begin, end) == hipSuccess);
                    REQUIRE(std::isfinite(ms) && ms > 0.0f);
                    std::printf("microbench rank=%u rows=%u pair=%u lds5=%u moe_ms=%.6f\n",
                                rank, rows, pair, mode, ms / 5.0f);
                }
            }
            REQUIRE(hipEventDestroy(begin) == hipSuccess);
            REQUIRE(hipEventDestroy(end) == hipSuccess);
        }
        // Invalid outer capacity must be rejected before writing earlier tiles.
        REQUIRE(ds4_gpu_tensor_fill_f32(t[0],NAN,rows*width));
        ds4_gpu_tensor *short_out = ds4_gpu_tensor_view(
            t[0],0,(rows-1u)*strides[0]);
        REQUIRE(short_out);
        ds4_gpu_tensor *invalid[8];
        std::memcpy(invalid,t,sizeof(t));
        invalid[0] = short_out;
        REQUIRE(!call(invalid,rows));
        ds4_gpu_tensor_free(short_out);
        REQUIRE(ds4_gpu_tensor_read(t[0],0,actual.data(),rows*strides[0]));
        for (float value:actual) REQUIRE(std::isnan(value));
        if (cold_lds5) {
            REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_COLD_LDS5", "invalid", 1) == 0);
            REQUIRE(!call(t,rows));
            REQUIRE(ds4_gpu_tensor_read(t[0],0,actual.data(),rows*strides[0]));
            for (float value:actual) REQUIRE(std::isnan(value));
            REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_COLD_LDS5", "0", 1) == 0);
        }
        REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_PARTITION", "invalid", 1) == 0);
        REQUIRE(!call(t,rows));
        REQUIRE(ds4_gpu_synchronize());
        ds4_gpu_q4k_packed_slice_release_all();
    }
    for (auto *tensor:t) ds4_gpu_tensor_free(tensor);
    ds4_gpu_cleanup();
    std::puts("PASS packed Q4_K batch equals independent <=M256 groups on both halves");
}
