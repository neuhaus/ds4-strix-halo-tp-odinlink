#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec "$repo/tests/run-clean-gate-python.sh" tests/test_baseline_genesis.py
