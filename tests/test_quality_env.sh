#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# Exercise the production normalizer without starting model or peer processes.
# shellcheck disable=SC1090
source <(sed -n '/^normalize_quality_environment()/,/^}/p' "$repo/run-tp-quality-score.sh")
settings=(DS4_TP_TIMEOUT_SEC=60 DS4_ROCM_TP_PREFILL_SKIP_UNOWNED=0
          DS4_TP_TIMEOUT_SEC=600 DS4_ROCM_TP_PREFILL_SKIP_UNOWNED=1
          DS4_TP_BIG_DIRECT=1 DS4_TP_BIG_DIRECT=1
          'LITERAL=spaces and = signs' 'EMPTY=')
expected=$(env -i "${settings[@]}" /usr/bin/env | LC_ALL=C sort)
normalize_quality_environment settings
[[ ${#settings[@]} == 5 ]]
[[ $(env -i "${settings[@]}" /usr/bin/env | LC_ALL=C sort) == "$expected" ]]
printf -v encoded '%q ' "${settings[@]}"
python3 - "$repo" "$encoded" <<'PY'
import runpy
import sys
gate = runpy.run_path(sys.argv[1] + '/scripts/candidate-gate.py')
manifest = {'worker_env': sys.argv[2], 'coordinator_env': sys.argv[2]}
names = {'DS4_TP_TIMEOUT_SEC': {}, 'DS4_ROCM_TP_PREFILL_SKIP_UNOWNED': {}}
assert gate['manifest_switch_values'](manifest, names, 'quality fixture') == {
    'DS4_TP_TIMEOUT_SEC': '600', 'DS4_ROCM_TP_PREFILL_SKIP_UNOWNED': '1'}
PY
echo 'PASS quality launch environment retains effective values and passes gate parsing'
