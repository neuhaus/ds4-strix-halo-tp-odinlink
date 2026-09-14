// Included by the same-GGUF shard test to reuse its original-byte half
// gathering, tensor ownership and production routed-MoE entry point.
extern "C" int ds4_gpu_q4k_direct_control_expect(uint32_t);
extern "C" uint32_t ds4_gpu_q4k_direct_control_last_stages(void);

bool run_pair_probe() {
    const char *mode = std::getenv("DS4_GLM5_Q4K_PAIR_PROBE");
    CHECK(mode && (!std::strcmp(mode, "0") || !std::strcmp(mode, "1") ||
                   !std::strcmp(mode, "2")), "pair probe mode must be 0/1/2");
    const bool paired = mode[0] != '0', fused = mode[0] == '2';
    const uint32_t stages = 3u | (paired ? 8u : 0u) | (fused ? 4u : 0u);
    const char *output_path = std::getenv("DS4_GLM5_Q4K_PAIR_OUTPUT");
    CHECK(output_path && !std::ifstream(output_path).good(),
          "new pair comparison output path required");
    std::ofstream dump(output_path, std::ios::binary);
    CHECK(dump.good(), "open pair comparison output");
    CHECK(setenv("DS4_ROCM_Q4K_KSHARD_RESEARCH", "1", 1) == 0 &&
          setenv("DS4_ROCM_Q4K_WMMA", "1", 1) == 0 &&
          setenv("DS4_ROCM_DISABLE_Q4K_WMMA", "0", 1) == 0 &&
          setenv("DS4_ROCM_Q4K_WMMA_MIN_COUNT", "6", 1) == 0 &&
          setenv("DS4_ROCM_Q4K_WMMA_PAIR_GATE_UP", paired ? "1" : "0", 1) == 0 &&
          setenv("DS4_ROCM_Q4K_WMMA_FUSE_MID", fused ? "1" : "0", 1) == 0 &&
          setenv("DS4_ROCM_Q4K_WMMA_LAYER_LOG", "0", 1) == 0 &&
          setenv("DS4_ROCM_TP_PREFILL_SKIP_UNOWNED", "1", 1) == 0 &&
          unsetenv("DS4_ROCM_TP_SKIP_UNOWNED") == 0,
          "bind isolated pair settings before initialization");
    CHECK(ds4_gpu_q4k_direct_control_expect(stages) &&
          !ds4_gpu_q4k_direct_control_expect(2u) &&
          !ds4_gpu_q4k_direct_control_expect(7u) &&
          !ds4_gpu_q4k_direct_control_expect(16u),
          "exact direct-stage contract rejects invalid masks");
    const char *model = std::getenv("DS4_GLM5_MODEL");
    Glm5TestGGUF gguf;
    CHECK(model && gguf.open_file(model), "open original GGUF");
    uint64_t go = 0, uo = 0, dwo = 0;
    CHECK(gguf.tensor("blk.3.ffn_gate_exps.weight",
                      {kInput, kFullMid, kTotalExperts}, 12u, go) &&
          gguf.tensor("blk.3.ffn_up_exps.weight",
                      {kInput, kFullMid, kTotalExperts}, 12u, uo) &&
          gguf.tensor("blk.3.ffn_down_exps.weight",
                      {kFullMid, kOutput, kTotalExperts}, 12u, dwo),
          "original layer3 Q4_K tensors");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    RuntimeGuard runtime;
    CHECK(ds4_gpu_init_multi(&config), "initialize pair probe GPU");
    runtime.active = true;
    constexpr uint32_t tokens = 256, pairs = tokens * kUsed, experts = 22;
    const uint64_t gr = (kInput / kQk) * kQ4Block;
    const uint64_t dr = (kHalfMid / kQk) * kQ4Block;
    const uint64_t ge = kHalfMid * gr, de = kOutput * dr;
    const uint64_t mid_bytes = (uint64_t)pairs * kHalfMid * sizeof(float);
    const uint64_t down_bytes = (uint64_t)pairs * kOutput * sizeof(float);
    const uint64_t out_bytes = (uint64_t)tokens * kOutput * sizeof(float);
    std::vector<uint32_t> ids;
    for (uint32_t e = 0; e < experts; ++e) ids.push_back(e * 17u % 288u);
    std::vector<float> input((size_t)tokens * kInput), weights(pairs);
    std::vector<int32_t> selected(pairs);
    struct Events {
        hipEvent_t start = nullptr, stop = nullptr;
        ~Events() {
            if (stop) (void)hipEventDestroy(stop);
            if (start) (void)hipEventDestroy(start);
        }
    } events;
    CHECK(hipEventCreate(&events.start) == hipSuccess &&
          hipEventCreate(&events.stop) == hipSuccess, "pair timing events");

    for (uint32_t half = 0; half < 2u; ++half) {
        DeviceWeights device;
        std::vector<uint8_t> packed;
        const auto upload_half = [&](uint64_t offset, bool down, void **dst) {
            const bool ok = down
                ? gather_down_half_direct(gguf, offset, ids, de * 2u, dr * 2u,
                                           de, dr, half, packed)
                : gather_gate_half_direct(gguf, offset, ids, ge * 2u, ge,
                                           half, packed);
            return ok && hipMalloc(dst, packed.size()) == hipSuccess &&
                hipMemcpy(*dst, packed.data(), packed.size(),
                          hipMemcpyHostToDevice) == hipSuccess;
        };
        CHECK(upload_half(go, false, &device.gate_half[half]) &&
              upload_half(uo, false, &device.up_half[half]) &&
              upload_half(dwo, true, &device.down_half[half]),
              "load only original-byte expert half windows");
        ComponentTensors t;
        CHECK(alloc_tensor(t.selected, pairs * sizeof(int32_t)) &&
              alloc_tensor(t.weights, pairs * sizeof(float)) &&
              alloc_tensor(t.input, input.size() * sizeof(float)) &&
              alloc_tensor(t.gate, mid_bytes) && alloc_tensor(t.up, mid_bytes) &&
              alloc_tensor(t.mid, mid_bytes) && alloc_tensor(t.down, down_bytes) &&
              alloc_tensor(t.out_full, out_bytes), "allocate pair scratch");
        const auto call = [&]() {
            return ds4_gpu_routed_moe_batch_q4k_direct_control(
                &t.out_full, &t.gate, &t.up, &t.mid, &t.down,
                device.gate_half[half], device.up_half[half],
                device.down_half[half], ge, gr, de, dr,
                &t.selected, &t.weights, experts, kUsed, kClamp, &t.input,
                3u, tokens, kHalfMid);
        };
        for (uint32_t pattern = 0; pattern < 3u; ++pattern) {
            for (size_t i = 0; i < input.size(); ++i)
                input[i] = (float)((int)((i * 17u + pattern * 31u) % 257u) - 128)
                           / (pattern == 2u ? 32.0f : 256.0f);
            if (pattern == 1u) {
                // Eight distinct experts per token; changing token route sets.
                for (uint32_t p = 0; p < pairs; ++p)
                    selected[p] = (int32_t)(((p / kUsed) * 5u +
                                             (p % kUsed) * 5u) % 21u);
            } else {
                // Deliberate duplicate-slot stress plus exact hot/cold/tail
                // boundary counts. A bijection changes their pair positions.
                const uint32_t boundary[] = {1,2,3,4,5,6,7,15,16,17,31,32,33};
                uint32_t p = 0;
                for (uint32_t e = 0; e < 13u; ++e)
                    for (uint32_t j = 0; j < boundary[e]; ++j, ++p)
                        selected[(p * 613u + pattern * 37u) % pairs] = (int32_t)e;
                for (; p < pairs; ++p)
                    selected[(p * 613u + pattern * 37u) % pairs] =
                        13 + (int32_t)(p % 8u);
            }
            uint32_t counts[experts] = {};
            for (uint32_t p = 0; p < pairs; ++p) {
                ++counts[selected[p]];
                weights[p] = selected[p] >= 13 && p % 31u == 0u ? 0.0f
                    : (float)(1u + (p + pattern) % 7u) / 16.0f;
            }
            CHECK(ds4_gpu_tensor_write(&t.input, 0, input.data(),
                                       input.size() * sizeof(float)) &&
                  ds4_gpu_tensor_write(&t.selected, 0, selected.data(),
                                       selected.size() * sizeof(int32_t)) &&
                  ds4_gpu_tensor_write(&t.weights, 0, weights.data(),
                                       weights.size() * sizeof(float)),
                  "refresh pair inputs and routes");
            CHECK(hipMemset(t.mid.ptr, 0xff, mid_bytes) == hipSuccess &&
                  hipMemset(t.down.ptr, 0xff, down_bytes) == hipSuccess &&
                  hipMemset(t.out_full.ptr, 0xff, out_bytes) == hipSuccess &&
                  call() && ds4_gpu_synchronize() &&
                  ds4_gpu_q4k_direct_control_last_stages() == stages,
                  "execute exact requested stages and replace sentinels");
            for (const auto &item : {
                     std::make_pair(&t.mid, mid_bytes),
                     std::make_pair(&t.down, down_bytes),
                     std::make_pair(&t.out_full, out_bytes)}) {
                std::vector<float> values((size_t)item.second / sizeof(float));
                CHECK(ds4_gpu_tensor_read(item.first, 0, values.data(), item.second),
                      "read complete pair comparison tensor");
                for (float value : values) CHECK(std::isfinite(value),
                                                "all pair outputs written and finite");
                dump.write((const char *)values.data(), item.second);
                CHECK(dump.good(), "write complete comparison bytes");
            }
            std::vector<float> samples;
            for (uint32_t sample = 0; sample < 9u; ++sample) {
                CHECK(hipEventRecord(events.start) == hipSuccess && call() &&
                      hipEventRecord(events.stop) == hipSuccess &&
                      hipEventSynchronize(events.stop) == hipSuccess,
                      "time complete routed MoE");
                float ms = 0.0f;
                CHECK(hipEventElapsedTime(&ms, events.start, events.stop) == hipSuccess,
                      "read routed MoE time");
                samples.push_back(ms);
            }
            std::sort(samples.begin(), samples.end());
            std::printf("pair-probe mode=%s half=%u pattern=%u stages=0x%x "
                        "M=256 K=4096 N=1024 median_ms=%.6f min_ms=%.6f "
                        "max_ms=%.6f routes_fnv=%016llx counts=",
                        mode, half, pattern, stages, samples[4], samples[0], samples[8],
                        (unsigned long long)fnv1a64(selected.data(), pairs * sizeof(int32_t)));
            for (uint32_t e = 0; e < experts; ++e) std::printf("%u,", counts[e]);
            std::printf("\n");
        }
    }
    dump.close();
    CHECK(dump.good(), "flush comparison dump");
    std::printf("PASS original-GGUF Q4 pair probe mode=%s stages=0x%x\n", mode, stages);
    return true;
}
