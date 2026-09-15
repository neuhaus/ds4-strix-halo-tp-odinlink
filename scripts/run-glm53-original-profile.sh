#!/usr/bin/env bash
# Diagnostic attribution of the recovered, unchanged-GGUF configuration.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo/scripts/ds4-research-root.sh"
ds4_resolve_research_roots "$repo"
export DS4_RESEARCH_ROOT DS4_PEER_RESEARCH_ROOT
phase=${1:?usage: run-glm53-original-profile.sh decode|prefill REP}
repeat=${2:?missing repetition number}
[[ $phase =~ ^(decode|prefill)$ && $repeat =~ ^[1-9][0-9]*$ ]]
artifact=$DS4_RESEARCH_ROOT/builds/glm53-full-3f1fe7d
dossier=$DS4_RESEARCH_ROOT/candidates/glm53-flash-roce-v2-20260912
record=$dossier/bench-runs/q8-recovery-4096-mode3-c5aa432-r3.manifest
[[ $(sha256sum "$record" | cut -d' ' -f1) == 1bd1994422b81e972c768f474cd784bcfe24342785c02bf90ebb81a00ac11fb5 ]]
[[ $(sha256sum "$artifact/ds4" | cut -d' ' -f1) == 4415be11fe44ea3914c18409b4eb029dab34bf3a79b7a08a0e34e37d750c6403 ]]
[[ $(sha256sum "$artifact/ds4-bench-tp" | cut -d' ' -f1) == 9b615fa17e88337e77e1eabaf2a460952c449ebec1fa2a9c13f63ee1e7957cda ]]
[[ $(sha256sum "$artifact/run-tp-ds4-bench.sh" | cut -d' ' -f1) == eabe664ece4bc064450557760faa5cc48807c67b3a48e1cd0ade66ae70f9f884 ]]
export DS4_BENCH_CONFIG=/home/wkljohn/Desktop/cc/ds4-glm5-next-tp2/bench.env.local
export DS4_BENCH_REPO=$artifact DS4_PEER_REPO=$DS4_PEER_RESEARCH_ROOT/builds/glm53-full-3f1fe7d
export DS4_PEER_MGMT=wkljohn@192.168.99.2
export DS4_BENCH_RDMA_PROFILE=roce-v2 DS4_RDMA_GID_INDEX=3
export DS4_COORDINATOR_ADDR=192.168.99.1
export DS4_LOCAL_RDMA_DEVICE=mlx5_0 DS4_PEER_RDMA_DEVICE=mlx5_1
export DS4_BENCH_FRONTIER=4096 DS4_BENCH_FRONTIER_MAX=4096
export DS4_BENCH_STEP_INCR=2048 DS4_BENCH_STEP_MUL=1
export DS4_BENCH_TOKENS=10 DS4_BENCH_CONTEXT=8192
export DS4_BENCH_PREFILL_CHUNK=2048 DS4_BENCH_TP_TIMEOUT_SEC=600
export DS4_BENCH_PROMPT_FILE=$DS4_RESEARCH_ROOT/bench-prompts/cross-discipline-v1.md
export DS4_BENCH_OUT=$dossier/bench-runs
export DS4_BENCH_LANE=B DS4_BENCH_CANDIDATE=0 DS4_BENCH_EXPECT_FNV64= DS4_BENCH_BASELINE_ID=
export DS4_BENCH_QUALITY=0 DS4_BENCH_DSPARK=0
export DS4_BENCH_ROCPROF=1 DS4_BENCH_ROCPROF_RUNTIME=2
export DS4_BENCH_ROCPROF_REGION=$phase DS4_BENCH_ROCPROF_RANK=coordinator
export DS4_BENCH_ROCPROF_BIN=/usr/bin/rocprofv3
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
[[ ! ${seen[DS4_GLM5_PHASE_PROFILE]+present} && ! ${seen[DS4_ROCM_GLM5_INDEXER_SCORE_WARP32]+present} ]]
tag=original-${phase}-profile-3f1fe7d-r${repeat}
[[ ! -e $DS4_BENCH_OUT/$tag.manifest && ! -e $DS4_BENCH_OUT/$tag.launcher.log ]]
cd "$artifact"
bash ./run-tp-ds4-bench.sh "$tag" \
  /home/wkljohn/Desktop/cc/models/antirez-glm-5.3-flash-gguf/GLM-5.3-Flash-Q4_K.gguf \
  "${extra[@]}" DS4_GLM5_PHASE_PROFILE=1 DS4_ROCM_GLM5_INDEXER_SCORE_WARP32=0 \
  > "$DS4_BENCH_OUT/$tag.launcher.log" 2>&1
cat "$DS4_BENCH_OUT/$tag.csv"
