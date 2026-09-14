#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
supervisor="$repo/scripts/tp-worker-supervisor.sh"
fixture=$(mktemp -d)
supervisor_pid=
cleanup() {
  if [[ -n ${supervisor_pid:-} ]] && kill -0 "$supervisor_pid" 2>/dev/null; then
    pkill -TERM -P "$supervisor_pid" 2>/dev/null || true
    kill -KILL "$supervisor_pid" 2>/dev/null || true
  fi
  rm -r -- "$fixture"
}
trap cleanup EXIT

normal_status="$fixture/normal.status"
normal_rc=0
"$supervisor" "$normal_status" bash -c 'exit 7' || normal_rc=$?
[[ $normal_rc == 7 ]]
grep -Fxq 'exit_code=7' "$normal_status"
grep -Fxq 'signal=0' "$normal_status"

term_status="$fixture/term.status"
ready="$fixture/ready"
# The child shell, not this test shell, expands $1.
# shellcheck disable=SC2016
"$supervisor" "$term_status" bash -c '
  trap "sleep 1; exit 143" TERM
  : > "$1"
  while :; do sleep 1; done
' worker "$ready" &
supervisor_pid=$!
for _ in {1..50}; do
  [[ -e $ready ]] && break
  sleep 0.02
done
[[ -e $ready ]]

kill -TERM "$supervisor_pid"
sleep 0.2
kill -0 "$supervisor_pid" 2>/dev/null || {
  echo "error: supervisor published status before its worker exited" >&2
  exit 1
}

for _ in {1..100}; do
  kill -0 "$supervisor_pid" 2>/dev/null || break
  sleep 0.05
done
kill -0 "$supervisor_pid" 2>/dev/null && {
  echo "error: supervisor did not reap its TERM-forwarded worker" >&2
  exit 1
}
term_rc=0
wait "$supervisor_pid" || term_rc=$?
supervisor_pid=
[[ $term_rc == 143 ]]
grep -Fxq 'exit_code=143' "$term_status"
grep -Fxq 'signal=15' "$term_status"

echo "test_tp_worker_supervisor: PASS"
