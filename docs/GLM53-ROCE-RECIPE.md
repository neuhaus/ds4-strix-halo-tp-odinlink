# GLM-5.3 Flash Q4_K RoCE v2 recipe

The README's ROCm 10 single-run measurement uses original Antirez GLM-5.3 Flash Q4_K,
two gfx1151 devices, ordinary greedy decode, 4,096 prompt tokens, 300 generated
tokens, and prefill batch 256. It uses the following explicit settings; it is
not a measurement of unspecified launcher defaults. The model bytes are
unchanged, with no concatenated weight copy or persistent expanded-weight cache.

Build the ROCm executables using the pinned toolchain documented in the README
and deploy the matching `ds4` executable to the peer. Configure the peer SSH
address, coordinator address, and Mellanox devices for the local installation.
Use the same cross-disciplinary prompt for matched measurements.

```sh
DS4_BENCH_RDMA_PROFILE=roce-v2 \
DS4_RDMA_GID_INDEX=3 \
DS4_BENCH_FRONTIER=4096 DS4_BENCH_FRONTIER_MAX=4096 \
DS4_BENCH_CONTEXT=4608 DS4_BENCH_TOKENS=300 \
DS4_BENCH_PREFILL_CHUNK=2048 \
DS4_BENCH_CANDIDATE=0 \
DS4_BENCH_PROMPT_FILE="$DS4_RESEARCH_ROOT/bench-prompts/cross-discipline-v1.md" \
./run-tp-ds4-bench.sh glm53-q4-roce-reproduce "$GLM53_Q4_MODEL" \
  DS4_TP_RDMA_LOGITS=1 DS4_GLM5_NEXT_PREFILL_BATCH=256 \
  DS4_GLM5_SPARSE_BATCH_BRIDGE=1 DS4_GLM5_SPARSE_BATCH_OUTPUT=1 \
  DS4_GLM5_SPARSE_BATCH_PRELUDE=1 DS4_GLM5_SPARSE_BATCH_VALUE=1 \
  DS4_ROCM_GLM5_SPARSE_ATTN_HEAD_SHARED=1 DS4_ROCM_GLM5_SPARSE_ATTN_F16_GEMM=1 \
  DS4_ROCM_GLM5_NOPE_ATTN_SHARED_PV=1 DS4_ROCM_GLM5_INDEXER_SCORE_BATCH=1 \
  DS4_GLM5_BATCH_POOL_STAGE=1 DS4_GLM5_KDA_TP=1 DS4_GLM5_SMALL_GATE=1 \
  DS4_GLM5_KDA_OUTPUT_ROWSLICE=1 DS4_ROCM_GLM5_Q4K_WMMA=1 \
  DS4_ROCM_GLM5_Q4K_KSHARD=1 DS4_GLM5_ALLOW_Q4_BATCH_MLA_OUTPUT=1 \
  DS4_ROCM_TP_PREFILL_SKIP_UNOWNED=1 \
  DS4_ROCM_GLM5_BF16_QKV_PREFETCH=64 DS4_ROCM_GLM5_BF16_QKV_NONTEMPORAL=1 \
  DS4_ROCM_GLM5_BF16_OUTPUT_PREFETCH=64 DS4_ROCM_GLM5_BF16_OUTPUT_NONTEMPORAL=1 \
  DS4_ROCM_GLM5_BF16_OUTPUT_ROWS_PER_BLOCK=8 DS4_ROCM_GLM5_QK_LOW_LDS_EXACT=1 \
  DS4_GLM_CAUSAL_ATTN_HEAD_SHARED=1 DS4_GLM5_SMALL_GATE_DIRECT_SEND=1 \
  DS4_ROCM_GLM5_BF16_ROWTILE2X16=1 DS4_ROCM_GLM5_Q8_SHAREDX_PREFETCH=8 \
  DS4_ROCM_GLM5_Q8_SHAREDX_NONTEMPORAL=1 DS4_ROCM_GLM5_Q8_SHAREDX_ROWS_PER_BLOCK=32 \
  DS4_ROCM_GLM5_BF16_WMMA_HILO=1 DS4_ROCM_GLM5_BF16_WMMA_QKV_FUSED=1 \
  DS4_ROCM_GLM5_BF16_QKV_DECODE_MULTIPTR=1 DS4_ROCM_SHARED_GU_WMMA_BATCH=0 \
  DS4_ROCM_SHARED_GU_WMMA_BATCH_TILE=256 DS4_ROCM_GLM5_MLA_OUTPUT_WMMA=0 \
  DS4_ROCM_GLM_CAUSAL_ATTN_HEAD_SHARED=1 DS4_ROCM_GLM5_Q8_DECODE_TILE=1 \
  DS4_METAL_MEMORY_REPORT=1 DS4_GLM5_NATIVE_DRAFT=0 DS4_TP_BULK_RECV_READY=0
```

This is an ordinary diagnostic invocation, not a formal performance candidate.
The measured source is `074a7ba024fb2049f6d0586a3d68704294e9e892`, using the
frozen `builds/glm53-074a7ba-rocm10` artifact: **103.42 prefill / 10.14 decode
t/s** at 4K and **103.43 / 10.27 t/s** at 8K. The 8K check uses the separate
security prompt and context 9,216, rather than extending the 4K prompt.
All 300 output tokens match the retained ordinary reference at each frontier.
Both processes' loaded SDK libraries and executable identities were captured.

Exact invocations, model/prompt hashes, effective settings, runtime captures
and quality results are retained in
`$DS4_RESEARCH_ROOT/candidates/glm53-mtp-20260919/`. The user waived repeated
timing for this SDK migration; these rates remain individual observations.
The pinned binaries, live SDK mappings, 4K/8K screens and deployment preflight
are checked. The user approved a [one-time quality exception](ROCM10-MIGRATION.md)
for the failed DeepSeek Q2/Q4 and GLM Q2 checks. Their results remain failures;
the original quality anchor and thresholds are unchanged.
Older ROCm 7.14 aggregate evidence remains in
`$DS4_RESEARCH_ROOT/candidates/glm53-flash-roce-v2-successor-71e6a24/`.
