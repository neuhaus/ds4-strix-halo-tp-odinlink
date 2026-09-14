#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if command -v sha256sum >/dev/null 2>&1; then
  expected=$(sha256sum "$repo/ds4_bench.c" | awk '{print $1}')
else
  expected=$(shasum -a 256 "$repo/ds4_bench.c" | awk '{print $1}')
fi

make -s -B -C "$repo" ds4_bench.o
grep -Fxq "$expected" < <(strings "$repo/ds4_bench.o") || {
  echo "error: ds4_bench.o does not embed its producer source SHA-256" >&2
  exit 1
}

echo "test_bench_producer_identity: PASS"
