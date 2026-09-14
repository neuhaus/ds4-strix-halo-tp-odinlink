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

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$REPO/scripts/ds4-research-root.sh"
BENCH_CONFIG=${DS4_BENCH_CONFIG:-$REPO/bench.env.local}
[[ ! -r $BENCH_CONFIG ]] || source "$BENCH_CONFIG"
ds4_resolve_research_roots "$REPO"
OUT=${DS4_QUALITY_OUT:-$DS4_RESEARCH_ROOT/accuracy-acceleration-2026-08-14}
mkdir -p "$OUT"
CANDIDATE="$OUT/$TAG.tsv"
THRESHOLDS=${DS4_QUALITY_THRESHOLDS:-$REPO/quality-thresholds.successor.json}
MANIFEST=${DS4_QUALITY_MANIFEST:-$REPO/gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv}
[[ -r $THRESHOLDS && -r $MANIFEST ]] || { echo "error: missing quality thresholds or tracked fixture manifest" >&2; exit 1; }
[[ ! -e $CANDIDATE ]] || { echo "error: refusing to overwrite $CANDIDATE" >&2; exit 1; }

DS4_QUALITY_MANIFEST="$MANIFEST" \
  DS4_QUALITY_RDMA_PROFILE=${DS4_QUALITY_RDMA_PROFILE:-roce-v2} \
  DS4_QUALITY_MAX_CASES=${DS4_QUALITY_MAX_CASES:-100} \
  "$REPO/run-tp-quality-score.sh" "$TAG" "$MODEL" "$@"

python3 "$REPO/scripts/compare-quality-scores.py" \
  "$REFERENCE" "$CANDIDATE" --thresholds "$THRESHOLDS" \
  --output "$OUT/$TAG.comparison.json"
echo "successor_quality_test=PASS score=$CANDIDATE comparison=$OUT/$TAG.comparison.json"
