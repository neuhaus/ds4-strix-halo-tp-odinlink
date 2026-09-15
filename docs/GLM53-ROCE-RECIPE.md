# GLM-5.3 Flash Q4_K RoCE v2 recipe

The README's nine-run measurement uses original Antirez GLM-5.3 Flash Q4_K,
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
DS4_BENCH_CANDIDATE=1 DS4_BENCH_LANE=B \
DS4_BENCH_BASELINE_ID=sha256:ea88ad8e50333637a4487f447acf124c8202405feef8b7b6703a0eda28a39e59 \
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
  DS4_METAL_MEMORY_REPORT=1
```

Lane B above enables the launcher's semantic checks before timing. The baseline
ID identifies the frozen control used for the reported comparison. The exact
invocations, prompt hash, effective settings on both ranks, executable hashes,
all nine paired values, quality comparisons, and regression screens are in
`$DS4_RESEARCH_ROOT/candidates/glm53-flash-roce-v2-successor-71e6a24`.
The tested source is `71e6a2475952c0dd49677a8dcc42661854047199`; documentation
changes do not imply that its frozen binaries were rebuilt.

Promotion retains these frozen runs under a user-approved correction to the
gate's comparison of redundant launcher metadata. Both effective rank
environments are unchanged. The exception, reviews, and full validation replay
are recorded in the same dossier; numerical and performance limits are unchanged.
