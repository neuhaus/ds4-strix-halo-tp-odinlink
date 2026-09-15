#!/usr/bin/env bash
# Reproduce the retained 10.81 t/s research configuration; not a promotion.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo/scripts/ds4-research-root.sh"
ds4_resolve_research_roots "$repo"
export DS4_RESEARCH_ROOT DS4_PEER_RESEARCH_ROOT
frontier=${1:?usage: run-glm53-q8-recovery.sh 2048|4096|8192 0|3 REP}
mode=${2:?missing Q8 mode (0 control, 3 retained fast schedule)}
repeat=${3:?missing repetition number}
[[ $frontier =~ ^(2048|4096|8192)$ && $mode =~ ^[03]$ && $repeat =~ ^[1-9][0-9]*$ ]]
artifact=$DS4_RESEARCH_ROOT/builds/glm53-q8-decode-c5aa432
dossier=$DS4_RESEARCH_ROOT/candidates/glm53-flash-roce-v2-20260912
record=$dossier/bench-runs/q8-fastlane-4096-mode3-c5aa432-r1.manifest
export DS4_BENCH_CONFIG=/home/wkljohn/Desktop/cc/ds4-glm5-next-tp2/bench.env.local
export DS4_BENCH_REPO=$artifact DS4_PEER_REPO=$DS4_PEER_RESEARCH_ROOT/builds/glm53-q8-decode-c5aa432
export DS4_PEER_MGMT=wkljohn@192.168.99.2
export DS4_BENCH_RDMA_PROFILE=roce-v2 DS4_RDMA_GID_INDEX=3
export DS4_COORDINATOR_ADDR=192.168.99.1
export DS4_LOCAL_RDMA_DEVICE=mlx5_0 DS4_PEER_RDMA_DEVICE=mlx5_1
export DS4_BENCH_FRONTIER=$frontier DS4_BENCH_FRONTIER_MAX=$frontier
export DS4_BENCH_STEP_INCR=2048 DS4_BENCH_STEP_MUL=1
export DS4_BENCH_TOKENS=300 DS4_BENCH_PREFILL_CHUNK=2048
export DS4_BENCH_TP_TIMEOUT_SEC=600
export DS4_BENCH_CONTEXT=8192
export DS4_BENCH_PROMPT_FILE=$DS4_RESEARCH_ROOT/bench-prompts/cross-discipline-v1.md
if [[ $frontier == 2048 ]]; then export DS4_BENCH_CONTEXT=4096; fi
if [[ $frontier == 8192 ]]; then
  export DS4_BENCH_CONTEXT=9216
  export DS4_BENCH_PROMPT_FILE=$DS4_RESEARCH_ROOT/bench-prompts/cross-discipline-long10k-v1.md
fi
export DS4_BENCH_OUT=$dossier/bench-runs
export DS4_BENCH_LANE=B DS4_BENCH_CANDIDATE=0
export DS4_BENCH_EXPECT_FNV64= DS4_BENCH_BASELINE_ID=
export DS4_BENCH_QUALITY=0 DS4_BENCH_DSPARK=0 DS4_BENCH_ROCPROF=0
# Verify the immutable binary before launching. The launcher also compares
# local/peer identities and records effective environments for both ranks.
[[ $(sha256sum "$record" | cut -d' ' -f1) == 2c9555056eb1160b0a9b269f05841a368409e827cf739d47409508ddeb718130 ]]
[[ $(sha256sum "$DS4_BENCH_CONFIG" | cut -d' ' -f1) == d1fe73588e7ee4c6377533422dde42a497badd20e6ad87caa9bc214cc5448ca0 ]]
prompt_hash=24d19432acab4d4cd2971d938b3c013fcfad1010ed701218bc7bdc1b630ecfef
if [[ $frontier == 8192 ]]; then prompt_hash=28b70108e9ea1be081bf46e55dafbdde72bda61a5e3656e2fa5208e3df66e98d; fi
[[ $(sha256sum "$DS4_BENCH_PROMPT_FILE" | cut -d' ' -f1) == "$prompt_hash" ]]
[[ $(sha256sum "$artifact/ds4" | cut -d' ' -f1) == 492744abb093c59c1ce258eb9809a7ba5a77b8d2cb3e52c23dce191190379d8c ]]
[[ $(sha256sum "$artifact/ds4-bench-tp" | cut -d' ' -f1) == e57bdd74d086936c0618e37bee11b56395c19247962abc1eec944c7f00878633 ]]
[[ $(sha256sum "$artifact/run-tp-ds4-bench.sh" | cut -d' ' -f1) == eabe664ece4bc064450557760faa5cc48807c67b3a48e1cd0ade66ae70f9f884 ]]
recorded=()
while IFS= read -r line; do
  if [[ $line == extra_env=* ]]; then read -r -a recorded <<< "${line#extra_env=}"; fi
done < "$record"
(( ${#recorded[@]} > 0 ))
extra=()
declare -A seen=()
for kv in "${recorded[@]}"; do
  [[ $kv =~ ^DS4_[A-Z0-9_]+=[0-9]+$ ]]
  [[ $kv != DS4_ROCM_GLM5_Q8_DECODE_TILE=* ]] || continue
  key=${kv%%=*}
  [[ ! ${seen[$key]+present} ]]
  seen[$key]=1
  extra+=("$kv")
done
extra+=(DS4_ROCM_GLM5_Q8_DECODE_TILE="$mode")
tag=q8-recovery-${frontier}-mode${mode}-c5aa432-r${repeat}
[[ ! -e $DS4_BENCH_OUT/$tag.manifest && ! -e $DS4_BENCH_OUT/$tag.launcher.log ]]
cd "$artifact"
bash ./run-tp-ds4-bench.sh "$tag" \
  /home/wkljohn/Desktop/cc/models/antirez-glm-5.3-flash-gguf/GLM-5.3-Flash-Q4_K.gguf \
  "${extra[@]}" > "$DS4_BENCH_OUT/$tag.launcher.log" 2>&1
cat "$DS4_BENCH_OUT/$tag.csv"
"$repo/scripts/check-ds4-bench-result.sh" "$DS4_BENCH_OUT/$tag.csv" \
  "$DS4_BENCH_OUT/coordinator-$tag.log" "$DS4_BENCH_OUT/worker-$tag.log" \
  '' 300 0 roce-v2 mlx5_0 3
