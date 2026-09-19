#!/bin/bash
# Score tracked official continuations through a real TP=2 RDMA path.
# Usage: ./run-tp-quality-score.sh TAG MODEL.gguf [NAME=VALUE ...]
set -euo pipefail

TAG=${1:?usage: run-tp-quality-score.sh TAG MODEL.gguf [NAME=VALUE ...]}
MODEL=${2:?usage: run-tp-quality-score.sh TAG MODEL.gguf [NAME=VALUE ...]}
shift 2
EXTRA_ENV=("$@")
[[ $TAG =~ ^[A-Za-z0-9._-]+$ ]] || { echo "error: invalid tag" >&2; exit 2; }

normalize_quality_environment() {
  local -n assignments=$1
  local -A positions=()
  local -a normalized=()
  local assignment name
  for assignment in "${assignments[@]}"; do
    name=${assignment%%=*}
    if [[ ${positions[$name]+present} ]]; then
      normalized[${positions[$name]}]=$assignment
    else
      positions[$name]=${#normalized[@]}
      normalized+=("$assignment")
    fi
  done
  assignments=("${normalized[@]}")
}

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$REPO/scripts/ds4-research-root.sh"
BENCH_CONFIG=${DS4_BENCH_CONFIG:-$REPO/bench.env.local}
# shellcheck disable=SC1090
[[ ! -r $BENCH_CONFIG ]] || source "$BENCH_CONFIG"
# The launched ranks run under env -i for reproducibility.  Preserve explicitly
# set runtime switches instead of silently dropping them; command-line NAME=VALUE
# arguments still take precedence and are the preferred recorded form.
for _name in DS4_ROCM_GLM5_Q4K_WMMA DS4_ROCM_GLM5_Q4K_KSHARD \
             DS4_ROCM_Q4K_KSHARD_RESEARCH DS4_GLM5_KDA_TP \
             DS4_GLM5_KDA_OUTPUT_KSLICE \
             DS4_ROCM_GLM5_BATCH_KSLICE_OUTPUT \
             DS4_ROCM_KSLICE_ROUTE_TOKTILE \
             DS4_GLM5_NEXT_PREFILL_BATCH \
             DS4_GLM5_NEXT_ENABLE_ORDINARY DS4_TP_BIG_DIRECT; do
  if [[ -n ${!_name+x} ]] && [[ ! " ${EXTRA_ENV[*]} " == *" $_name="* ]]; then
    EXTRA_ENV+=("$_name=${!_name}")
  fi
done
ds4_resolve_research_roots "$REPO"
PEER_REPO=${DS4_PEER_REPO:-$REPO}
PEER_MGMT=${DS4_PEER_MGMT:-}
PEER_HOST_KEY_ALIAS=${DS4_PEER_HOST_KEY_ALIAS:-${PEER_MGMT#*@}}
COORDINATOR_ADDR=${DS4_COORDINATOR_ADDR:-}
PORT=${DS4_QUALITY_PORT:-9002}
CONTEXT=${DS4_QUALITY_CONTEXT:-4096}
MAX_CASES=${DS4_QUALITY_MAX_CASES:-100}
START_CASE=${DS4_QUALITY_START_CASE:-0}
OUT=${DS4_QUALITY_OUT:-$DS4_RESEARCH_ROOT/accuracy-acceleration-2026-08-14}
PEER_OUT=${DS4_PEER_QUALITY_OUT:-$DS4_PEER_RESEARCH_ROOT/accuracy-acceleration-2026-08-14}
[[ $OUT == /* && $PEER_OUT == /* ]] || {
  echo "error: quality output directories must be absolute" >&2; exit 2;
}
SCORER=$REPO/gguf-tools/quality-testing/score_official
# Keep explicit input paths relative to the caller while the scorer below
# resolves the fixture's repository-relative entries from the attested root.
MODEL=$(realpath -e -- "$MODEL")
MODEL_ARCH=$(python3 "$REPO/scripts/gguf_tensor_types.py" --architecture "$MODEL") || {
  echo "error: unable to inspect model architecture" >&2; exit 1;
}
if [[ -n ${DS4_QUALITY_MANIFEST:-} ]]; then
  MANIFEST=$DS4_QUALITY_MANIFEST
elif [[ $MODEL_ARCH == glm5-next ]]; then
  MANIFEST=$REPO/gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv
else
  MANIFEST=$REPO/gguf-tools/quality-testing/data/flash/manifest.tsv
fi
MANIFEST=$(realpath -e -- "$MANIFEST")
TEACHER_LOGITS_DIR=${DS4_QUALITY_TEACHER_LOGITS_DIR:-}
TEACHER_ARM=${DS4_QUALITY_TEACHER_ARM:-}
COORD_LOG=$OUT/coordinator-$TAG.log
WORKER_LOG=$OUT/worker-$TAG.log
REMOTE_WORKER_LOG=$PEER_OUT/worker-$TAG.log
SCORES=$OUT/$TAG.tsv
SCORES_MANIFEST=$OUT/$TAG.manifest
WORKER_PIDFILE=$PEER_OUT/worker-$TAG.pid
COORD_STATUS=$OUT/coordinator-$TAG.status
WORKER_STATUS=$OUT/worker-$TAG.status
REMOTE_WORKER_STATUS=$PEER_OUT/worker-$TAG.status
SUPERVISOR=$REPO/scripts/tp-worker-supervisor.sh
PEER_SUPERVISOR=$PEER_REPO/scripts/tp-worker-supervisor.sh
RUN_ID=$(< /proc/sys/kernel/random/uuid)
PEER_SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes
  -o "HostKeyAlias=$PEER_HOST_KEY_ALIAS" "$PEER_MGMT")
PEER_SCP=(scp -o BatchMode=yes -o StrictHostKeyChecking=yes
  -o "HostKeyAlias=$PEER_HOST_KEY_ALIAS")
RDMA_PROFILE=${DS4_QUALITY_RDMA_PROFILE:-${DS4_BENCH_RDMA_PROFILE:-odinlink}}
case $RDMA_PROFILE in
  odinlink)
    LOCAL_RDMA_DEVICE=${DS4_LOCAL_RDMA_DEVICE:-odl_tb5_0}
    PEER_RDMA_DEVICE=${DS4_PEER_RDMA_DEVICE:-odl_tb5_0}
    ODINLINK_ROOT=${DS4_ODINLINK_ROOT:-}
    [[ -n $ODINLINK_ROOT ]] || {
      echo "error: OdinLink quality scoring requires DS4_ODINLINK_ROOT" >&2; exit 2;
    }
    VERBS_LIB=$ODINLINK_ROOT/build/verbs/libodl_tb5_verbs.so.0.1.0
    ODL_LD_PATH=$ODINLINK_ROOT/build/lib:$ODINLINK_ROOT/build/verbs
    ;;
  roce-v2)
    LOCAL_RDMA_DEVICE=${DS4_LOCAL_RDMA_DEVICE:-mlx5_0}
    PEER_RDMA_DEVICE=${DS4_PEER_RDMA_DEVICE:-mlx5_1}
    RDMA_GID_INDEX=${DS4_RDMA_GID_INDEX:-3}
    ;;
  *)
    echo "error: DS4_QUALITY_RDMA_PROFILE must be odinlink or roce-v2" >&2
    exit 2
    ;;
esac

for required in PEER_MGMT COORDINATOR_ADDR; do
  [[ -n ${!required} ]] || { echo "error: set $required in bench.env.local or the environment" >&2; exit 2; }
done

[[ $PORT =~ ^[1-9][0-9]*$ && $CONTEXT =~ ^[1-9][0-9]*$ &&
   $MAX_CASES =~ ^[1-9][0-9]*$ && $START_CASE =~ ^[0-9]+$ ]] || {
  echo "error: invalid port, context, or max-case count" >&2; exit 2;
}
for kv in "${EXTRA_ENV[@]}"; do
  [[ $kv =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { echo "error: expected NAME=VALUE: $kv" >&2; exit 2; }
  case $kv in
    DS4_BENCH_RUN_ID=*)
      echo "error: quality run identity is assigned by the launcher" >&2; exit 2 ;;
    DS4_ROCM_ENABLE_Q8_F16_CACHE=*|DS4_ROCM_STREAM_Q8_F16_CACHE_GB=*)
      echo "error: persistent Q8-to-FP16 caches are excluded from accuracy tests" >&2; exit 2 ;;
    DS4_TP_GREEDY_TOP2=*)
      echo "error: quality scoring requires full-vocabulary RDMA logits" >&2; exit 2 ;;
    DS4_TP_EXPERT_SPLIT=*)
      echo "error: the accuracy baseline requires the balanced 128/128 split" >&2; exit 2 ;;
  esac
done
[[ -r $MODEL && -r $MANIFEST && -x $SCORER && -x $REPO/ds4 ]] || {
  echo "error: missing model, manifest, scorer, or ds4 binary" >&2; exit 1;
}
if [[ -n $TEACHER_LOGITS_DIR ]]; then
  case $TEACHER_LOGITS_DIR/ in
    "$DS4_RESEARCH_ROOT/"*) ;;
    *) echo "error: teacher-logit directory must be under $DS4_RESEARCH_ROOT" >&2; exit 2 ;;
  esac
  mkdir -p "$TEACHER_LOGITS_DIR"
  if find "$TEACHER_LOGITS_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "error: teacher-logit directory must be empty: $TEACHER_LOGITS_DIR" >&2
    exit 2
  fi
  if [[ $TEACHER_ARM == deepseek-ordinary ]]; then
    [[ $MODEL_ARCH == deepseek4 && $RDMA_PROFILE == roce-v2 ]] || {
      echo "error: deepseek-ordinary teacher capture requires deepseek4 over RoCE v2" >&2; exit 2;
    }
    for kv in "${EXTRA_ENV[@]}"; do
      case $kv in
        DS4_GLM5_NATIVE_DRAFT=0|DS4_ROCM_GLM5_Q8_DECODE_TILE=1) ;;
        DS4_DSPARK_*=*|DS4_*MTP*=*|DS4_GLM5_*=*|DS4_GLM_*=*|DS4_ROCM_GLM*=*)
          echo "error: deepseek-ordinary forbids speculative/GLM controls: ${kv%%=*}" >&2; exit 2 ;;
        DS4_TP_RDMA_LOGITS=*)
          [[ $kv == DS4_TP_RDMA_LOGITS=1 ]] || {
            echo "error: DeepSeek teacher capture requires full-vocabulary RDMA logits" >&2; exit 2;
          } ;;
      esac
    done
    # The two allowed GLM entries above preserve existing DeepSeek recipes:
    # native drafting is explicitly off; the GLM-only Q8 selector is inert.
    TEACHER_VALIDATOR=$REPO/scripts/compare-teacher-logits.py
    TEACHER_VALIDATOR_HASH=$(sha256sum "$TEACHER_VALIDATOR" | awk '{print $1}')
    TEACHER_FIXTURE_HASH=$(python3 "$TEACHER_VALIDATOR" --validate-capture --root "$REPO" \
      --fixture "$MANIFEST" --start-case "$START_CASE" --cases "$MAX_CASES")
  else
  if [[ " ${EXTRA_ENV[*]} " == *' DS4_ROCM_GLM5_BATCH_KSLICE_OUTPUT='* ]]; then
    echo "error: teacher-logit diagnostics isolate decode KDA output K-slice; batch K-slice must be unset" >&2
    exit 2
  fi
  [[ $MODEL_ARCH == glm5-next ]] || {
    echo "error: score_official full-logit arm labels currently cover GLM5 only" >&2
    exit 2
  }
  case $TEACHER_ARM in
    kda-off)
      EXPECTED_KDA_TP=0; EXPECTED_KDA_KSLICE=0 ;;
    kda-tp)
      EXPECTED_KDA_TP=1; EXPECTED_KDA_KSLICE=0 ;;
    kda-kslice)
      EXPECTED_KDA_TP=1; EXPECTED_KDA_KSLICE=1 ;;
    attn-scalar)
      EXPECTED_KDA_TP=1; EXPECTED_KDA_KSLICE=0
      [[ " ${EXTRA_ENV[*]} " == *' DS4_ROCM_GLM_CAUSAL_ATTN_EXACT_SPLIT=0 '* ]] || {
        echo "error: attn-scalar teacher arm requires exact-split rollback" >&2
        exit 2
      }
      [[ " ${EXTRA_ENV[*]} " != *' DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE='* &&
         " ${EXTRA_ENV[*]} " != *' DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_F32='* ]] || {
        echo "error: attn-scalar teacher arm forbids NoPE GEMM selectors" >&2
        exit 2
      } ;;
    attn-gemm-f32)
      EXPECTED_KDA_TP=1; EXPECTED_KDA_KSLICE=0
      [[ " ${EXTRA_ENV[*]} " == *' DS4_ROCM_GLM_CAUSAL_ATTN_EXACT_SPLIT=0 '* ]] || {
        echo "error: attn-gemm-f32 teacher arm requires exact-split rollback" >&2
        exit 2
      }
      [[ " ${EXTRA_ENV[*]} " == *' DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE=1 '* &&
         " ${EXTRA_ENV[*]} " == *' DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_F32=1 '* ]] || {
        echo "error: attn-gemm-f32 teacher arm requires both NoPE GEMM selectors" >&2
        exit 2
      } ;;
    attn-exact-split)
      EXPECTED_KDA_TP=1; EXPECTED_KDA_KSLICE=0
      [[ " ${EXTRA_ENV[*]} " != *' DS4_ROCM_GLM_CAUSAL_ATTN_EXACT_SPLIT=0 '* &&
         " ${EXTRA_ENV[*]} " != *' DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE='* &&
         " ${EXTRA_ENV[*]} " != *' DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_F32='* &&
         " ${EXTRA_ENV[*]} " != *' DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_POSTDIV='* &&
         " ${EXTRA_ENV[*]} " != *' DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_PV_SCALAR='* &&
         " ${EXTRA_ENV[*]} " != *' DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_SCORE_SCALAR='* ]] || {
        echo "error: attn-exact-split teacher arm requires the unmodified default" >&2
        exit 2
      } ;;
    *)
      echo "error: invalid DS4_QUALITY_TEACHER_ARM" >&2
      exit 2 ;;
  esac
  [[ " ${EXTRA_ENV[*]} " == *" DS4_GLM5_KDA_TP=$EXPECTED_KDA_TP "* &&
     " ${EXTRA_ENV[*]} " == *" DS4_GLM5_KDA_OUTPUT_KSLICE=$EXPECTED_KDA_KSLICE "* ]] || {
    echo "error: teacher arm $TEACHER_ARM requires KDA_TP=$EXPECTED_KDA_TP and KDA_OUTPUT_KSLICE=$EXPECTED_KDA_KSLICE" >&2
    exit 2
  }
  [[ " ${EXTRA_ENV[*]} " == *' DS4_GLM5_NEXT_PREFILL_BATCH='* ]] || {
    echo "error: teacher-logit diagnostics require an explicit GLM5 prefill batch" >&2
    exit 2
  }
  fi
fi

# score_official and ds4 are different executables linked against the same
# core objects.  A stale scorer can therefore negotiate a different graph
# even when the two rank-side ds4 binaries match.  Fail before model loading
# whenever either executable predates a core input.
CORE_BUILD_INPUTS=(
  "$REPO/Makefile"
  "$REPO/ds4.c" "$REPO/ds4.h" "$REPO/ds4_gpu.h"
  "$REPO/ds4_distributed.c" "$REPO/ds4_distributed.h"
  "$REPO/ds4_tp.c" "$REPO/ds4_tp.h"
  "$REPO/ds4_ssd.c" "$REPO/ds4_ssd.h"
  "$REPO/ds4_rocm.cu" "$REPO/ds4_rocm_compat.cu"
  "$REPO/ds4_rocm_unavailable.cu"
  "$REPO/ds4_layer_pack.c" "$REPO/ds4_layer_pack.h"
  "$REPO/ds4_glm5_kda.c" "$REPO/ds4_glm5_kda.h"
  "$REPO/ds4_glm5_next_runtime.c" "$REPO/ds4_glm5_next_runtime.h"
  "$REPO/ds4_glm5_next_state.c" "$REPO/ds4_glm5_next_state.h"
  "$REPO/ds4_glm5_next_exec.c" "$REPO/ds4_glm5_next_exec.h"
  "$REPO/rax.c" "$REPO/rax.h"
  "$REPO"/rocm/*.cuh
)
for binary in "$REPO/ds4" "$SCORER"; do
  for source in "${CORE_BUILD_INPUTS[@]}"; do
    [[ ! -e $source || ! $source -nt $binary ]] || {
      echo "error: $(basename "$binary") is stale relative to ${source#"$REPO"/}; rebuild ds4 and the quality scorer together" >&2
      exit 1
    }
  done
done
[[ ! $REPO/gguf-tools/quality-testing/score_official.c -nt $SCORER ]] || {
  echo "error: score_official is stale; rebuild ds4 and the quality scorer together" >&2
  exit 1
}
if [[ $RDMA_PROFILE == odinlink ]]; then
  [[ -r /dev/odl_tb5_0 && -r $VERBS_LIB ]] || {
    echo "error: local OdinLink device or provider unavailable" >&2; exit 1;
  }
  "${PEER_SSH[@]}" "test -r '$MODEL' -a -r /dev/odl_tb5_0 -a -r '$VERBS_LIB' -a -x '$PEER_REPO/ds4'" || {
    echo "error: peer model, binary, OdinLink device, or provider unavailable" >&2; exit 1;
  }
else
  [[ $RDMA_GID_INDEX =~ ^[0-9]+$ && $RDMA_GID_INDEX -le 255 ]] || {
    echo "error: DS4_RDMA_GID_INDEX must be in 0..255" >&2; exit 2;
  }
  grep -qx 'RoCE v2' "/sys/class/infiniband/$LOCAL_RDMA_DEVICE/ports/1/gid_attrs/types/$RDMA_GID_INDEX" || {
    echo "error: local RoCE v2 GID unavailable" >&2; exit 1;
  }
  "${PEER_SSH[@]}" "test -r '$MODEL' -a -x '$PEER_REPO/ds4' && test \"\$(cat '/sys/class/infiniband/$PEER_RDMA_DEVICE/ports/1/gid_attrs/types/$RDMA_GID_INDEX')\" = 'RoCE v2'" || {
    echo "error: peer model, binary, or RoCE v2 GID unavailable" >&2; exit 1;
  }
fi

LOCAL_SIZE=$(stat -c %s "$MODEL")
PEER_SIZE=$("${PEER_SSH[@]}" "stat -c %s '$MODEL'")
[[ $LOCAL_SIZE == "$PEER_SIZE" ]] || { echo "error: model sizes differ" >&2; exit 1; }
sample_fingerprint() {
  local path=$1 size half tail
  size=$(stat -c %s "$path")
  half=$((size / 2 > 4194304 ? size / 2 - 4194304 : 0))
  tail=$((size > 8388608 ? size - 8388608 : 0))
  {
    printf '%s\n' "$size"
    dd if="$path" iflag=skip_bytes,count_bytes skip=0 count=8388608 status=none
    dd if="$path" iflag=skip_bytes,count_bytes skip="$half" count=8388608 status=none
    dd if="$path" iflag=skip_bytes,count_bytes skip="$tail" count=8388608 status=none
  } | sha256sum | awk '{print $1}'
}
LOCAL_MODEL_FINGERPRINT=$(sample_fingerprint "$MODEL")
printf -v MODEL_Q '%q' "$MODEL"
PEER_MODEL_FINGERPRINT=$("${PEER_SSH[@]}" "p=$MODEL_Q; s=\$(stat -c %s \"\$p\"); h=\$((s / 2 > 4194304 ? s / 2 - 4194304 : 0)); t=\$((s > 8388608 ? s - 8388608 : 0)); { printf '%s\\n' \"\$s\"; dd if=\"\$p\" iflag=skip_bytes,count_bytes skip=0 count=8388608 status=none; dd if=\"\$p\" iflag=skip_bytes,count_bytes skip=\"\$h\" count=8388608 status=none; dd if=\"\$p\" iflag=skip_bytes,count_bytes skip=\"\$t\" count=8388608 status=none; } | sha256sum | awk '{print \$1}'")
[[ $LOCAL_MODEL_FINGERPRINT == "$PEER_MODEL_FINGERPRINT" ]] || {
  echo "error: sampled model fingerprints differ" >&2; exit 1;
}
LOCAL_HASH=$(sha256sum "$REPO/ds4" | awk '{print $1}')
PEER_HASH=$("${PEER_SSH[@]}" "sha256sum '$PEER_REPO/ds4'" | awk '{print $1}')
[[ $LOCAL_HASH == "$PEER_HASH" ]] || { echo "error: rank binary hashes differ" >&2; exit 1; }
SOURCE_COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)
INFERENCE_SOURCE_COMMIT=${DS4_QUALITY_INFERENCE_SOURCE_COMMIT:-$SOURCE_COMMIT}
if [[ -n ${DS4_QUALITY_INFERENCE_SOURCE_COMMIT:-} ]]; then
  [[ $INFERENCE_SOURCE_COMMIT =~ ^[0-9a-f]{40}$ &&
     ${DS4_QUALITY_EXPECT_DS4_SHA256:-} == "$LOCAL_HASH" &&
     ${DS4_QUALITY_EXPECT_SCORER_SHA256:-} == "$(sha256sum "$SCORER" | awk '{print $1}')" ]] || {
    echo "error: explicit inference source requires matching ds4/scorer hashes" >&2; exit 2;
  }
  if ! git -C "$REPO" cat-file -e "$INFERENCE_SOURCE_COMMIT^{commit}" ||
     ! git -C "$REPO" diff --quiet "$INFERENCE_SOURCE_COMMIT" -- \
       '*.c' '*.h' '*.cu' '*.cuh' '*.cpp' '*.inc'; then
    echo "error: reused inference binaries require identical implementation source" >&2; exit 2;
  fi
elif [[ -n ${DS4_QUALITY_EXPECT_DS4_SHA256:-}${DS4_QUALITY_EXPECT_SCORER_SHA256:-} ]]; then
  echo "error: binary hash pins require an explicit inference source" >&2; exit 2
fi
if git -C "$REPO" diff --quiet --ignore-submodules -- 2>/dev/null &&
   git -C "$REPO" diff --cached --quiet --ignore-submodules -- 2>/dev/null &&
   [[ -z $(git -C "$REPO" ls-files --others --exclude-standard) ]]; then
  SOURCE_DIRTY=0
else
  SOURCE_DIRTY=1
fi

if pgrep -af '[s]core_official.*--role coordinator' >/dev/null; then
  echo "error: a quality coordinator is already running" >&2; exit 1
fi
if "${PEER_SSH[@]}" "pgrep -af '[d]s4 .*--role worker.*--tensor-parallel'" >/dev/null; then
  echo "error: a TP worker is already running on the peer" >&2; exit 1
fi

mkdir -p "$OUT"
for artifact in "$COORD_LOG" "$WORKER_LOG" "$COORD_STATUS" "$WORKER_STATUS" \
                "$SCORES" "$SCORES_MANIFEST"; do
  [[ ! -e $artifact && ! -L $artifact ]] || {
    echo "error: refusing to overwrite quality evidence: $artifact" >&2; exit 2;
  }
done
# Retain this reservation even after failure: a tag always names one attempt.
mkdir "$OUT/.quality-$TAG.reserved" || {
  echo "error: quality tag already reserved: $TAG" >&2; exit 2;
}
printf -v PEER_OUT_Q '%q' "$PEER_OUT"
"${PEER_SSH[@]}" "mkdir -p $PEER_OUT_Q"
printf -v PEER_RESERVATION_Q '%q' "$PEER_OUT/.quality-$TAG.reserved"
"${PEER_SSH[@]}" "mkdir $PEER_RESERVATION_Q" || {
  echo "error: peer quality tag already reserved: $TAG" >&2; exit 2;
}
for artifact in "$REMOTE_WORKER_LOG" "$REMOTE_WORKER_STATUS" "$WORKER_PIDFILE"; do
  printf -v ARTIFACT_Q '%q' "$artifact"
  "${PEER_SSH[@]}" "test ! -e $ARTIFACT_Q && test ! -L $ARTIFACT_Q" || {
    echo "error: refusing to overwrite peer quality evidence: $artifact" >&2; exit 2;
  }
done
SUPERVISOR_HASH=$(sha256sum "$SUPERVISOR" | awk '{print $1}')
LAUNCHER_HASH=$(sha256sum "$REPO/run-tp-quality-score.sh" | awk '{print $1}')
printf -v PEER_SUPERVISOR_Q '%q' "$PEER_SUPERVISOR"
PEER_SUPERVISOR_HASH=$("${PEER_SSH[@]}" "sha256sum $PEER_SUPERVISOR_Q" | awk '{print $1}')
[[ $SUPERVISOR_HASH == "$PEER_SUPERVISOR_HASH" ]] || {
  echo "error: quality worker supervisor hashes differ" >&2; exit 1;
}
COMMON_ENV=(
  env -i PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8
  "DS4_BENCH_RUN_ID=$RUN_ID"
  DS4_TP_TIMEOUT_SEC=60
  DS4_TP_RDMA_LOGITS=1
  DS4_ROCM_Q4K_DECODE_STAGE_XQ=1
  DS4_ROCM_Q8_DECODE_PAIR_DP4A=0
  DS4_ROCM_Q4K_DECODE_SPLIT_GATE_UP=1
  DS4_ROCM_TP_SKIP_UNOWNED=1
  DS4_ROCM_TP_PREFILL_SKIP_UNOWNED=0
  DS4_ROCM_TP_ZERO_WEIGHT_TILE_SKIP=1
  DS4_ROCM_SHARED_GU_SWIGLU_FUSE=1
  "${EXTRA_ENV[@]}"
)
LOCAL_RDMA_ARGS=(--rdma-device "$LOCAL_RDMA_DEVICE")
PEER_RDMA_ARGS=(--rdma-device "$PEER_RDMA_DEVICE")
if [[ $RDMA_PROFILE == odinlink ]]; then
  COMMON_ENV+=(
    DS4_TP_ODINLINK_BATCH_ASYNC=1
    DS4_TP_VERBS_LIB="$VERBS_LIB"
    LD_LIBRARY_PATH="$ODL_LD_PATH"
  )
else
  LOCAL_RDMA_ARGS+=(--rdma-gid-index "$RDMA_GID_INDEX")
  PEER_RDMA_ARGS+=(--rdma-gid-index "$RDMA_GID_INDEX")
fi
PREFILL_ARGS=(--prefill-chunk 4096)
if [[ $MODEL_ARCH == glm5-next ]]; then
  PREFILL_ARGS=()
  COMMON_ENV+=(
    DS4_GLM5_NEXT_ENABLE_ORDINARY=1
    DS4_TP_BIG_DIRECT=1
    DS4_TP_GREEDY_TOP2=0
    DS4_ROCM_TEMPORAL_COMPRESSOR=0
  )
fi

# Match env's last-assignment-wins behavior, then launch and record the same
# unambiguous array. Defaults, explicit overrides and architecture requirements
# have all been assembled by this point.
normalize_quality_environment COMMON_ENV

echo "quality_tag=$TAG"
echo "ds4_sha256=$LOCAL_HASH"
echo "scorer_sha256=$(sha256sum "$SCORER" | awk '{print $1}')"
echo "model_sample_sha256=$LOCAL_MODEL_FINGERPRINT"
echo "manifest_sha256=$(sha256sum "$MANIFEST" | awk '{print $1}')"

WORKER_STARTED=0
COORD_PID=0
worker_action() {
  local command
  printf -v command 'bash -s -- %q %q %q %q' \
    "$WORKER_PIDFILE" "$PEER_SUPERVISOR" "$REMOTE_WORKER_STATUS" "$1"
  "${PEER_SSH[@]}" "$command" <<'WORKER_ACTION'
pidfile=$1; supervisor=$2; status=$3; action=$4
[[ -r $pidfile ]] || exit 3
read -r pid < "$pidfile"
[[ $pid =~ ^[1-9][0-9]*$ && $pid -gt 1 ]] || exit 2
[[ -r /proc/$pid/cmdline ]] || exit 1
mapfile -d '' -t args < "/proc/$pid/cmdline"
if (( ${#args[@]} == 0 )); then
  # A reaped child may briefly remain as a zombie under the SSH session.
  [[ ! -r /proc/$pid/stat ]] && exit 1
  read -r proc_stat < "/proc/$pid/stat"
  [[ ${proc_stat##*) } == Z\ * ]] && exit 1
  exit 3
fi
# Match the actual script and this run's status argument, not a substring.
[[ ${args[1]:-} == "$supervisor" && ${args[2]:-} == "$status" ]] || exit 3
case $action in
  check) exit 0 ;;
  term) kill -TERM "$pid" ;;
  *) exit 2 ;;
esac
WORKER_ACTION
}
wait_worker() {
  local limit=$1 i rc
  for ((i = 0; i < limit; i++)); do
    if worker_action check; then
      sleep 1
    else
      rc=$?
      [[ $rc == 1 ]] && return 0
      echo "error: worker identity or peer access could not be established (status=$rc)" >&2
      return "$rc"
    fi
  done
  return 1
}
copy_worker_evidence() {
  local rc=0
  "${PEER_SCP[@]}" "$PEER_MGMT:$REMOTE_WORKER_LOG" "$WORKER_LOG" || rc=1
  "${PEER_SCP[@]}" "$PEER_MGMT:$REMOTE_WORKER_STATUS" "$WORKER_STATUS" || rc=1
  return "$rc"
}
wait_coordinator() {
  local i
  for ((i = 0; i < 60; i++)); do
    if ! kill -0 "$COORD_PID" 2>/dev/null; then
      wait "$COORD_PID" || true
      COORD_PID=0
      return 0
    fi
    sleep 1
  done
  echo "error: coordinator was not reaped after TERM; leaving it intact" >&2
  return 1
}
cleanup() {
  local rc=$?
  trap '' INT TERM
  if (( WORKER_STARTED == 1 )); then
    worker_action term || true
  fi
  if (( COORD_PID > 0 )) && kill -0 "$COORD_PID" 2>/dev/null; then
    kill -TERM "$COORD_PID" 2>/dev/null || true
    wait_coordinator || true
  fi
  if (( WORKER_STARTED == 1 )); then
    wait_worker 60 || echo "error: worker completion could not be established" >&2
    copy_worker_evidence || true
  fi
  return "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

WORKER_APP=(./ds4 --role worker --tensor-parallel --coordinator "$COORDINATOR_ADDR" "$PORT"
  --transport rdma --rocm -m "$MODEL" -c "$CONTEXT"
  "${PREFILL_ARGS[@]}" "${PEER_RDMA_ARGS[@]}")
printf -v WORKER_CMD_Q '%q ' "${COMMON_ENV[@]}" "${WORKER_APP[@]}"
printf -v PEER_REPO_Q '%q' "$PEER_REPO"
printf -v WORKER_LOG_Q '%q' "$REMOTE_WORKER_LOG"
printf -v WORKER_PIDFILE_Q '%q' "$WORKER_PIDFILE"
# Set ownership before SSH so a lost reply still reaches safe cleanup.
WORKER_STARTED=1
printf -v WORKER_STATUS_Q '%q' "$REMOTE_WORKER_STATUS"
"${PEER_SSH[@]}" "cd $PEER_REPO_Q || exit 1; set -C; nohup /usr/bin/setsid $PEER_SUPERVISOR_Q $WORKER_STATUS_Q $WORKER_CMD_Q > $WORKER_LOG_Q 2>&1 < /dev/null & p=\$!; echo \$p > $WORKER_PIDFILE_Q"

SCORER_EXTRA_ARGS=()
[[ -z $TEACHER_LOGITS_DIR ]] || SCORER_EXTRA_ARGS+=(--teacher-logits-dir "$TEACHER_LOGITS_DIR")
(
  cd -- "$REPO"
  exec "$SUPERVISOR" "$COORD_STATUS" "${COMMON_ENV[@]}" "$SCORER" "$MODEL" "$MANIFEST" "$SCORES" "$CONTEXT" \
    --start-case "$START_CASE" --max-cases "$MAX_CASES" \
    "${SCORER_EXTRA_ARGS[@]}" \
    --role coordinator --tensor-parallel \
    --listen 0.0.0.0 "$PORT" --transport rdma --rocm \
    "${LOCAL_RDMA_ARGS[@]}"
) >"$COORD_LOG" 2>&1 &
COORD_PID=$!
COORD_RC=0
wait "$COORD_PID" || COORD_RC=$?
COORD_PID=0
(( COORD_RC == 0 )) || exit "$COORD_RC"

wait_worker 180 || { echo "error: worker did not exit after STOP" >&2; exit 1; }
copy_worker_evidence || { echo "error: incomplete worker evidence" >&2; exit 1; }
WORKER_STARTED=0
trap - EXIT INT TERM
(( COORD_RC == 0 )) || exit "$COORD_RC"
for status in "$COORD_STATUS" "$WORKER_STATUS"; do
  cmp -s "$status" <(printf 'exit_code=0\nsignal=0\n') || {
    echo "error: missing or unsuccessful rank terminal status: $status" >&2; exit 1;
  }
done
for log in "$COORD_LOG" "$WORKER_LOG"; do
  [[ $(grep '^ds4-tp: benchmark run_id=' "$log") == "ds4-tp: benchmark run_id=$RUN_ID" ]] || {
    echo "error: quality run identity missing or duplicated in $log" >&2; exit 1;
  }
done

[[ $(awk 'NR > 1 {n++} END {print n+0}' "$SCORES") == "$MAX_CASES" ]] || {
  echo "error: incomplete score table" >&2; exit 1;
}
for log in "$COORD_LOG" "$WORKER_LOG"; do
  grep -q 'transport=rdma' "$log" || { echo "error: RDMA not confirmed in $log" >&2; exit 1; }
  ! grep -Eq 'fallback_calls=[1-9]|transport fallback|Connection timed out|Protocol error' "$log" || {
    echo "error: fallback or transport failure in $log" >&2; exit 1;
  }
done
if [[ " ${EXTRA_ENV[*]} " == *' DS4_ROCM_GLM5_Q4K_KSHARD=1 '* ]]; then
  for log in "$COORD_LOG" "$WORKER_LOG"; do
    grep -Eq 'Q4_K WMMA startup .*kshard=1([[:space:]]|$)' "$log" || {
      echo "error: GLM Q4_K K-shard was requested but did not negotiate in $log" >&2
      exit 1
    }
  done
fi
if [[ " ${EXTRA_ENV[*]} " == *' DS4_GLM5_KDA_TP=1 '* ]]; then
  for log in "$COORD_LOG" "$WORKER_LOG"; do
    grep -Eq 'GLM5 TP features: kda_tp=1([[:space:]]|.*kda_output_kslice=)' "$log" || {
      echo "error: GLM5 KDA-TP was requested but did not negotiate in $log" >&2
      exit 1
    }
  done
fi
if [[ " ${EXTRA_ENV[*]} " == *' DS4_GLM5_KDA_OUTPUT_KSLICE=1 '* ]]; then
  [[ " ${EXTRA_ENV[*]} " == *' DS4_GLM5_KDA_TP=1 '* ]] || {
    echo "error: GLM5 KDA output K-slice requires DS4_GLM5_KDA_TP=1" >&2
    exit 1
  }
  [[ " ${EXTRA_ENV[*]} " == *' DS4_GLM5_NEXT_PREFILL_BATCH='* ]] || {
    echo "error: GLM5 KDA output K-slice quality runs require an explicit prefill batch" >&2
    exit 1
  }
  for log in "$COORD_LOG" "$WORKER_LOG"; do
    grep -Eq 'GLM5 TP features: kda_tp=1 kda_output_kslice=1' "$log" || {
      echo "error: GLM5 KDA output K-slice was requested but did not negotiate in $log" >&2
      exit 1
    }
    grep -q 'GLM5 KDA output K-slice engaged' "$log" || {
      echo "error: GLM5 KDA output K-slice negotiated but never executed in $log" >&2
      exit 1
    }
  done
fi
if [[ $TEACHER_ARM == attn-gemm-f32 || $TEACHER_ARM == attn-exact-split ]]; then
  for log in "$COORD_LOG" "$WORKER_LOG"; do
    grep -Eq 'GLM causal indexed prefill using fp32([-a-z]+)? .* attention GEMMs' "$log" || {
      echo "error: FP32 NoPE GEMM teacher arm never engaged in $log" >&2
      exit 1
    }
  done
fi
TARGET_TOKENS=$(awk -F'\t' 'NR > 1 {sum += $3} END {print sum+0}' "$SCORES")
printf -v EXTRA_ENV_Q '%q ' "${EXTRA_ENV[@]}"
EFFECTIVE_ENV=()
for _item in "${COMMON_ENV[@]}"; do
  [[ $_item =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] && EFFECTIVE_ENV+=("$_item")
done
printf -v EFFECTIVE_ENV_Q '%q ' "${EFFECTIVE_ENV[@]}"
[[ $(sha256sum "$REPO/run-tp-quality-score.sh" | awk '{print $1}') == "$LAUNCHER_HASH" &&
   $(sha256sum "$SUPERVISOR" | awk '{print $1}') == "$SUPERVISOR_HASH" ]] || {
  echo "error: quality launcher or supervisor changed during the run" >&2; exit 1;
}
completion_manifest() {
  local name path
  printf 'completion_schema=quality-terminal-v1\nrun_id=%s\n' "$RUN_ID"
  printf 'worker_supervisor_sha256=%s\n' "$SUPERVISOR_HASH"
  printf 'quality_launcher_sha256=%s\n' "$LAUNCHER_HASH"
  for name in scores coordinator_status worker_status coordinator_log worker_log; do
    case $name in
      scores) path=$SCORES ;;
      coordinator_status) path=$COORD_STATUS ;;
      worker_status) path=$WORKER_STATUS ;;
      coordinator_log) path=$COORD_LOG ;;
      worker_log) path=$WORKER_LOG ;;
    esac
    printf '%s_path=%s\n' "$name" "$path"
    printf '%s_sha256=%s\n' "$name" "$(sha256sum "$path" | awk '{print $1}')"
  done
}
if [[ -n $TEACHER_LOGITS_DIR ]]; then
  ACTUAL_DUMPS=$(find "$TEACHER_LOGITS_DIR" -maxdepth 1 -type f \
    -name 'decode_*.logits.json' | wc -l)
  [[ $TARGET_TOKENS =~ ^[1-9][0-9]*$ && $ACTUAL_DUMPS == "$TARGET_TOKENS" ]] || {
    echo "error: teacher-logit count mismatch: expected=$TARGET_TOKENS actual=$ACTUAL_DUMPS" >&2
    exit 1
  }
  (cd "$TEACHER_LOGITS_DIR" && sha256sum decode_*.logits.json) \
    > "$TEACHER_LOGITS_DIR/files.sha256"
  git -C "$REPO" diff --binary HEAD -- > "$TEACHER_LOGITS_DIR/source.diff"
  git -C "$REPO" status --porcelain=v1 --untracked-files=all \
    > "$TEACHER_LOGITS_DIR/source.status"
  if [[ $TEACHER_ARM == deepseek-ordinary ]]; then
    [[ $(sha256sum "$TEACHER_VALIDATOR" | awk '{print $1}') == "$TEACHER_VALIDATOR_HASH" &&
       $(python3 "$TEACHER_VALIDATOR" --validate-capture --root "$REPO" --fixture "$MANIFEST" \
          --start-case "$START_CASE" --cases "$MAX_CASES" \
          --dumps "$TEACHER_LOGITS_DIR" --scores "$SCORES" --model "$MODEL") == "$TEACHER_FIXTURE_HASH" ]] || {
      echo "error: invalid teacher capture or changed fixture/validator" >&2; exit 1;
    }
    for log in "$COORD_LOG" "$WORKER_LOG"; do
      if [[ $(grep -c 'Q4_K WMMA startup rank=' "$log") != 1 ||
            $(grep '^ds4-tp: transport proof ' "$log") != 'ds4-tp: transport proof requested=rdma active=rdma payload_fallback_calls=0 failed=0' ]] ||
         ! grep -Fq "rdma GID index $RDMA_GID_INDEX (RoCE v2)" "$log" ||
         ! grep -q 'expanded_weight_cache_bytes=0\([[:space:]]\|$\)' "$log" ||
         grep -Eq 'expanded_weight_cache_bytes=[1-9]' "$log"; then
        echo "error: DeepSeek capture lacks startup/RoCE/zero-fallback/cache proof in $log" >&2; exit 1;
      fi
    done
    COORD_FEATURES=$(grep 'Q4_K WMMA startup rank=' "$COORD_LOG")
    WORKER_FEATURES=$(grep 'Q4_K WMMA startup rank=' "$WORKER_LOG")
    [[ $COORD_FEATURES == *'startup rank=0 '* && $WORKER_FEATURES == *'startup rank=1 '* &&
       ${COORD_FEATURES/startup rank=0 /startup rank=1 } == "$WORKER_FEATURES" ]] || {
      echo "error: DeepSeek rank negotiation differs" >&2; exit 1;
    }
    for field in kda_tp kda_output_kslice quality kill_switch; do
      [[ " $COORD_FEATURES " == *" $field=0 "* ]] || {
        echo "error: invalid DeepSeek startup $field" >&2; exit 1;
      }
    done
    [[ $COORD_FEATURES =~ negotiated=(0x[0-9a-fA-F]+)([[:space:]]|$) ]] || {
      echo "error: missing DeepSeek negotiation mask" >&2; exit 1;
    }
    COORD_NEGOTIATED=${BASH_REMATCH[1]}
    WORKER_NEGOTIATED=$COORD_NEGOTIATED
    # DeepSeek Q4 uses the existing packed K-shard path; mixed Q2 does not.
    # Admit either boolean only when it matches negotiated feature bit19.
    [[ $COORD_FEATURES =~ (^|[[:space:]])kshard=([01])([[:space:]]|$) ]] || {
      echo "error: invalid DeepSeek startup kshard" >&2; exit 1;
    }
    KSHARD=${BASH_REMATCH[2]}
    (( KSHARD == ((COORD_NEGOTIATED >> 19) & 1) )) || {
      echo "error: DeepSeek K-shard state disagrees with negotiation" >&2; exit 1;
    }
    if (( KSHARD )); then
      [[ $(python3 "$REPO/scripts/gguf_tensor_types.py" --routed-family "$MODEL") == Q4_K ]] || {
        echo "error: DeepSeek K-shard requires Q4_K routed weights" >&2; exit 1;
      }
    fi
  else
  COORD_FEATURES=$(grep -E 'GLM5 TP features: kda_tp=[01] kda_output_kslice=[01]' \
    "$COORD_LOG" | tail -1 || true)
  WORKER_FEATURES=$(grep -E 'GLM5 TP features: kda_tp=[01] kda_output_kslice=[01]' \
    "$WORKER_LOG" | tail -1 || true)
  COORD_NEGOTIATED=$(grep -E 'negotiated=0x[0-9a-fA-F]+' "$COORD_LOG" | tail -1 || true)
  WORKER_NEGOTIATED=$(grep -E 'negotiated=0x[0-9a-fA-F]+' "$WORKER_LOG" | tail -1 || true)
  [[ -n $COORD_FEATURES && -n $WORKER_FEATURES &&
     -n $COORD_NEGOTIATED && -n $WORKER_NEGOTIATED ]] || {
    echo "error: teacher-logit diagnostic lacks negotiated feature proof" >&2
    exit 1
  }
  fi
  {
    printf 'producer=gguf-tools/quality-testing/score_official.c\n'
    printf 'tag=%s\n' "$TAG"
    printf 'source_commit=%s\n' "$SOURCE_COMMIT"
    printf 'inference_source_commit=%s\n' "$INFERENCE_SOURCE_COMMIT"
    printf 'model_arch=%s\ncontext=%s\n' "$MODEL_ARCH" "$CONTEXT"
    if [[ $TEACHER_ARM == deepseek-ordinary ]]; then
      printf 'fixture_content_sha256=%s\n' "$TEACHER_FIXTURE_HASH"
      printf 'capture_validator_sha256=%s\n' "$TEACHER_VALIDATOR_HASH"
    fi
    printf 'source_dirty=%s\n' "$SOURCE_DIRTY"
    printf 'model=%s\n' "$MODEL"
    printf 'model_size=%s\n' "$LOCAL_SIZE"
    printf 'model_sample_sha256=%s\n' "$LOCAL_MODEL_FINGERPRINT"
    printf 'ds4_sha256=%s\n' "$LOCAL_HASH"
    printf 'scorer_sha256=%s\n' "$(sha256sum "$SCORER" | awk '{print $1}')"
    printf 'quality_input_sha256=%s\n' "$(sha256sum "$MANIFEST" | awk '{print $1}')"
    printf 'start_case=%s\n' "$START_CASE"
    printf 'cases=%s\n' "$MAX_CASES"
    printf 'teacher_positions=%s\n' "$TARGET_TOKENS"
    printf 'teacher_arm=%s\n' "$TEACHER_ARM"
    printf 'rdma_profile=%s\n' "$RDMA_PROFILE"
    printf 'common_env=%s\n' "$EFFECTIVE_ENV_Q"
    printf 'worker_env=%s\n' "$EFFECTIVE_ENV_Q"
    printf 'coordinator_env=%s\n' "$EFFECTIVE_ENV_Q"
    printf 'extra_env=%s\n' "$EXTRA_ENV_Q"
    printf 'coordinator_features=%s\n' "$COORD_FEATURES"
    printf 'worker_features=%s\n' "$WORKER_FEATURES"
    printf 'coordinator_negotiated=%s\n' "$COORD_NEGOTIATED"
    printf 'worker_negotiated=%s\n' "$WORKER_NEGOTIATED"
    completion_manifest
    printf 'source_diff_sha256=%s\n' "$(sha256sum "$TEACHER_LOGITS_DIR/source.diff" | awk '{print $1}')"
    printf 'source_status_sha256=%s\n' "$(sha256sum "$TEACHER_LOGITS_DIR/source.status" | awk '{print $1}')"
    printf 'files_sha256=%s\n' "$(sha256sum "$TEACHER_LOGITS_DIR/files.sha256" | awk '{print $1}')"
  } > "$TEACHER_LOGITS_DIR/manifest"
fi
{
  printf 'tag=%s\n' "$TAG"
  printf 'source_commit=%s\n' "$SOURCE_COMMIT"
  printf 'inference_source_commit=%s\n' "$INFERENCE_SOURCE_COMMIT"
  printf 'source_dirty=%s\n' "$SOURCE_DIRTY"
  printf 'model=%s\n' "$MODEL"
  printf 'model_size=%s\n' "$LOCAL_SIZE"
  printf 'model_sample_sha256=%s\n' "$LOCAL_MODEL_FINGERPRINT"
  printf 'ds4_sha256=%s\n' "$LOCAL_HASH"
  printf 'scorer_sha256=%s\n' "$(sha256sum "$SCORER" | awk '{print $1}')"
  printf 'quality_input_sha256=%s\n' "$(sha256sum "$MANIFEST" | awk '{print $1}')"
  printf 'cases=%s\n' "$MAX_CASES"
  printf 'start_case=%s\n' "$START_CASE"
  printf 'target_tokens=%s\n' "$TARGET_TOKENS"
  printf 'context=%s\n' "$CONTEXT"
  printf 'rdma_profile=%s\n' "$RDMA_PROFILE"
  printf 'local_rdma_device=%s\n' "$LOCAL_RDMA_DEVICE"
  printf 'peer_rdma_device=%s\n' "$PEER_RDMA_DEVICE"
  printf 'rdma_gid_index=%s\n' "${RDMA_GID_INDEX:-n/a}"
  printf 'dspark=0\n'
  printf 'common_env=%s\n' "$EFFECTIVE_ENV_Q"
  printf 'worker_env=%s\n' "$EFFECTIVE_ENV_Q"
  printf 'coordinator_env=%s\n' "$EFFECTIVE_ENV_Q"
  printf 'extra_env=%s\n' "$EXTRA_ENV_Q"
  completion_manifest
} > "$SCORES_MANIFEST"
echo "scores=$SCORES"
echo "scores_manifest=$SCORES_MANIFEST"
grep -E 'cases=|api_ref_tokens=' "$COORD_LOG" | tail -2
echo QUALITY_RUN_DONE
