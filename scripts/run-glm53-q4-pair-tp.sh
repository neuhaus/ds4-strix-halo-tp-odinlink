#!/usr/bin/env bash
# Matched 4096+300 ordinary inference on the unchanged GGUF. Research only.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo/scripts/ds4-research-root.sh"
ds4_resolve_research_roots "$repo"
revision=${1:?usage: run-glm53-q4-pair-tp.sh COMMIT7 0|1|2 REP [SHARED_Q8_PAIR] [SHARED_Q8_ROWS] [SHARED_Q8_SERIAL]}
mode=${2:?missing pair mode}
repeat=${3:?missing repetition}
shared_q8_pair=${4:-0}
shared_q8_rows=${5:-32}
shared_q8_serial=${6:-0}
[[ $revision =~ ^[0-9a-f]{7}$ && $mode =~ ^[012]$ &&
   $repeat =~ ^[1-9][0-9]*$ && $shared_q8_pair =~ ^[01]$ &&
   $shared_q8_rows =~ ^(8|16|32)$ && $shared_q8_serial =~ ^[01]$ ]]
artifact=$DS4_RESEARCH_ROOT/builds/glm53-full-$revision
dossier=$DS4_RESEARCH_ROOT/candidates/glm53-flash-roce-v2-20260912
record=$dossier/bench-runs/q8-recovery-4096-mode3-c5aa432-r3.manifest
[[ $(sha256sum "$record" | cut -d' ' -f1) == 1bd1994422b81e972c768f474cd784bcfe24342785c02bf90ebb81a00ac11fb5 ]]
[[ $(cut -c1-7 "$artifact/BUILD-SOURCE") == "$revision" ]]
(cd "$artifact" && sha256sum --check --status BUILD-SHA256SUMS)
export DS4_BENCH_CONFIG=/home/wkljohn/Desktop/cc/ds4-glm5-next-tp2/bench.env.local
export DS4_BENCH_REPO=$artifact DS4_PEER_REPO=$DS4_PEER_RESEARCH_ROOT/builds/glm53-full-$revision
export DS4_PEER_MGMT=wkljohn@192.168.99.2
export DS4_BENCH_RDMA_PROFILE=roce-v2 DS4_RDMA_GID_INDEX=3
export DS4_COORDINATOR_ADDR=192.168.99.1
export DS4_LOCAL_RDMA_DEVICE=mlx5_0 DS4_PEER_RDMA_DEVICE=mlx5_1
export DS4_BENCH_FRONTIER=4096 DS4_BENCH_FRONTIER_MAX=4096
export DS4_BENCH_STEP_INCR=2048 DS4_BENCH_STEP_MUL=1
export DS4_BENCH_TOKENS=300 DS4_BENCH_CONTEXT=8192
export DS4_BENCH_PREFILL_CHUNK=2048 DS4_BENCH_TP_TIMEOUT_SEC=600
export DS4_BENCH_PROMPT_FILE=$DS4_RESEARCH_ROOT/bench-prompts/cross-discipline-v1.md
export DS4_BENCH_OUT=$dossier/bench-runs
# Lane A is incremental to the recovered Q8 mode3 research stack. Its
# outstanding Lane-B teacher validation is not cleared by this comparison.
export DS4_BENCH_LANE=A DS4_BENCH_CANDIDATE=0
export DS4_BENCH_EXPECT_FNV64=9012bd4d7c5ce422 DS4_BENCH_BASELINE_ID=
export DS4_BENCH_QUALITY=0 DS4_BENCH_DSPARK=0 DS4_BENCH_ROCPROF=0
[[ $(sha256sum "$DS4_BENCH_CONFIG" | cut -d' ' -f1) == d1fe73588e7ee4c6377533422dde42a497badd20e6ad87caa9bc214cc5448ca0 ]]
[[ $(sha256sum "$DS4_BENCH_PROMPT_FILE" | cut -d' ' -f1) == 24d19432acab4d4cd2971d938b3c013fcfad1010ed701218bc7bdc1b630ecfef ]]
extra=()
while IFS= read -r line; do
    if [[ $line == extra_env=* ]]; then read -r -a extra <<< "${line#extra_env=}"; fi
done < "$record"
(( ${#extra[@]} > 0 ))
declare -A seen=()
for kv in "${extra[@]}"; do
    [[ $kv =~ ^DS4_[A-Z0-9_]+=[0-9]+$ ]]
    key=${kv%%=*}
    [[ ! ${seen[$key]+present} ]]
    seen[$key]=1
done
[[ ! ${seen[DS4_ROCM_GLM5_BF16_QKV_ACTIVATION_PANEL]+present} &&
   ! ${seen[DS4_ROCM_GLM5_INDEXER_SCORE_WARP32]+present} ]]
pair=0
fused=0
(( mode >= 1 )) && pair=1
(( mode == 2 )) && fused=1
for key in DS4_ROCM_Q4K_WMMA_PAIR_GATE_UP DS4_ROCM_Q4K_WMMA_FUSE_MID \
           DS4_ROCM_Q4K_WMMA_MIN_COUNT DS4_ROCM_Q4K_WMMA_LAYER_LOG \
           DS4_ROCM_GLM5_SHARED_Q8_PAIR_DECODE \
           DS4_ROCM_GLM5_SHARED_Q8_PAIR_ROWS \
           DS4_ROCM_GLM5_SHARED_Q8_PAIR_SERIAL; do
    [[ ! ${seen[$key]+present} ]]
done
tag=q4-pair-4096-mode${mode}-shared${shared_q8_pair}-rows${shared_q8_rows}-serial${shared_q8_serial}-${revision}-r${repeat}
[[ ! -e $DS4_BENCH_OUT/$tag.manifest && ! -e $DS4_BENCH_OUT/$tag.launcher.log ]]
cd "$artifact"
bash ./run-tp-ds4-bench.sh "$tag" \
  /home/wkljohn/Desktop/cc/models/antirez-glm-5.3-flash-gguf/GLM-5.3-Flash-Q4_K.gguf \
  "${extra[@]}" DS4_ROCM_GLM5_INDEXER_SCORE_WARP32=0 \
  DS4_ROCM_GLM5_BF16_QKV_ACTIVATION_PANEL=0 \
  DS4_ROCM_Q4K_WMMA_PAIR_GATE_UP="$pair" DS4_ROCM_Q4K_WMMA_FUSE_MID="$fused" \
  DS4_ROCM_Q4K_WMMA_MIN_COUNT=6 DS4_ROCM_Q4K_WMMA_LAYER_LOG=0 \
  DS4_ROCM_GLM5_SHARED_Q8_PAIR_DECODE="$shared_q8_pair" \
  DS4_ROCM_GLM5_SHARED_Q8_PAIR_ROWS="$shared_q8_rows" \
  DS4_ROCM_GLM5_SHARED_Q8_PAIR_SERIAL="$shared_q8_serial" \
  > "$DS4_BENCH_OUT/$tag.launcher.log" 2>&1
cat "$DS4_BENCH_OUT/$tag.csv"
"$repo/scripts/check-ds4-bench-result.sh" "$DS4_BENCH_OUT/$tag.csv" \
  "$DS4_BENCH_OUT/coordinator-$tag.log" "$DS4_BENCH_OUT/worker-$tag.log" \
  9012bd4d7c5ce422 300 0 roce-v2 mlx5_0 3
for rank_log in "$DS4_BENCH_OUT/coordinator-$tag.log" "$DS4_BENCH_OUT/worker-$tag.log"; do
    [[ -f "$rank_log" ]]
    grep -q 'rdma GID index 3 (RoCE v2)' "$rank_log"
    grep -q "pair_active=$pair" "$rank_log"
    if (( fused )); then
        grep -q 'Q4_K WMMA fused-mid active' "$rank_log"
        grep -q 'fused_active=1' "$rank_log"
    else
        ! grep -q 'Q4_K WMMA fused-mid active' "$rank_log"
        grep -q 'fused_active=0' "$rank_log"
    fi
    if (( shared_q8_pair )); then
        grep -q 'GLM5 shared gate/up Q8 decode pair engaged' "$rank_log"
        grep -q 'GLM5 Q8 pair shared-X prefetch engaged' "$rank_log"
    else
        ! grep -q 'GLM5 shared gate/up Q8 decode pair engaged' "$rank_log"
        ! grep -q 'GLM5 Q8 pair shared-X prefetch engaged' "$rank_log"
    fi
    if (( shared_q8_serial )); then
        grep -q 'GLM5 Q8 pair serial shared-X engaged' "$rank_log"
    else
        ! grep -q 'GLM5 Q8 pair serial shared-X engaged' "$rank_log"
    fi
done
