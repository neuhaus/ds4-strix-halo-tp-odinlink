#!/usr/bin/env bash
set -euo pipefail

test_path=${1:?usage: run-clean-gate-python.sh tests/TEST.py}
shift
[[ $test_path == tests/*.py && $test_path != *..* ]] || {
  echo "error: gate test must be a tests/*.py path" >&2
  exit 2
}
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ${DS4_GATE_CLEAN_TEST:-0} != 1 ]]; then
  clean_fixture=$(mktemp -d)
  trap 'rm -r -- "$clean_fixture"' EXIT
  git clone -q --no-local "$repo" "$clean_fixture/repo"
  if [[ -n $(git -C "$repo" status --porcelain=v1 -uall) ]]; then
    git -C "$repo" diff --binary HEAD | git -C "$clean_fixture/repo" apply
    while IFS= read -r -d '' path; do
      mkdir -p "$clean_fixture/repo/$(dirname -- "$path")"
      cp -a -- "$repo/$path" "$clean_fixture/repo/$path"
    done < <(git -C "$repo" ls-files -o --exclude-standard -z)
    git -C "$clean_fixture/repo" add -A
    git -C "$clean_fixture/repo" -c user.name=DS4-Gate-Test \
      -c user.email=gate-test.invalid commit -qm 'test fixture snapshot'
  fi
  (cd "$clean_fixture/repo" &&
    DS4_GATE_CLEAN_TEST=1 ./tests/run-clean-gate-python.sh "$test_path" "$@")
  exit
fi

exec python3 "$repo/$test_path" "$@"
