#!/bin/bash
# Run the tracked official-continuation quality test for a successor artifact.
# This is independent of generated-token FNV checks.
set -euo pipefail

TAG=${1:?usage: run-successor-quality-test.sh TAG MODEL.gguf REFERENCE.tsv [NAME=VALUE ...]}
MODEL=${2:?usage: run-successor-quality-test.sh TAG MODEL.gguf REFERENCE.tsv [NAME=VALUE ...]}
REFERENCE=${3:?usage: run-successor-quality-test.sh TAG MODEL.gguf REFERENCE.tsv [NAME=VALUE ...]}
shift 3
[[ $TAG =~ ^[A-Za-z0-9._-]+$ ]] || { echo "error: invalid tag" >&2; exit 2; }
[[ -r $MODEL && -r $REFERENCE ]] || { echo "error: model or reference TSV is unreadable" >&2; exit 1; }
MODEL=$(realpath "$MODEL")
REFERENCE=$(realpath "$REFERENCE")

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$REPO/scripts/ds4-research-root.sh"
BENCH_CONFIG=${DS4_BENCH_CONFIG:-$REPO/bench.env.local}
# shellcheck disable=SC1090
[[ ! -r $BENCH_CONFIG ]] || source "$BENCH_CONFIG"
ds4_resolve_research_roots "$REPO"
OUT=${DS4_QUALITY_OUT:-$DS4_RESEARCH_ROOT/accuracy-acceleration-2026-08-14}
mkdir -p "$OUT"
CANDIDATE="$OUT/$TAG.tsv"
THRESHOLDS=${DS4_QUALITY_THRESHOLDS:-$REPO/quality-thresholds.successor.json}
ARCH=$(python3 "$REPO/scripts/gguf_tensor_types.py" --architecture "$MODEL")
case $ARCH in
  glm5-next) FIXTURE=glm53-flash-openrouter-zai-fp8-100 ;;
  deepseek4) FIXTURE=flash ;;
  *) echo "error: no quality fixture for architecture $ARCH" >&2; exit 2 ;;
esac
MANIFEST=${DS4_QUALITY_MANIFEST:-$REPO/gguf-tools/quality-testing/data/$FIXTURE/manifest.tsv}
[[ -r $THRESHOLDS && -r $MANIFEST ]] || { echo "error: missing quality thresholds or tracked fixture manifest" >&2; exit 1; }
for evidence in "$CANDIDATE" "$OUT/$TAG.manifest" "$OUT/$TAG.comparison.json" \
                "$OUT/coordinator-$TAG.log" "$OUT/worker-$TAG.log" \
                "$OUT/coordinator-$TAG.status" "$OUT/worker-$TAG.status" \
                "$OUT/.quality-$TAG.reserved"; do
  [[ ! -e $evidence && ! -L $evidence ]] || { echo "error: refusing to overwrite $evidence" >&2; exit 1; }
done
[[ ${DS4_QUALITY_RDMA_PROFILE:-roce-v2} == roce-v2 ]] || {
  echo "error: successor quality tests require RoCE v2" >&2; exit 2;
}

# Manifest paths inside the tracked fixture are relative to the checkout.
cd "$REPO"

DS4_QUALITY_MANIFEST="$MANIFEST" \
  DS4_QUALITY_RDMA_PROFILE=${DS4_QUALITY_RDMA_PROFILE:-roce-v2} \
  DS4_QUALITY_MAX_CASES=${DS4_QUALITY_MAX_CASES:-100} \
  "$REPO/run-tp-quality-score.sh" "$TAG" "$MODEL" "$@"

python3 "$REPO/scripts/compare-quality-scores.py" \
  "$REFERENCE" "$CANDIDATE" --thresholds "$THRESHOLDS" \
  --require-candidate-status \
  --output "$OUT/$TAG.comparison.json"
echo "successor_quality_test=PASS score=$CANDIDATE comparison=$OUT/$TAG.comparison.json"
