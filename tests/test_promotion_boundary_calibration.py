#!/usr/bin/env python3
"""Check the frozen 5/7/9 repeated-Student design and its generator."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "scripts" / "promotion-boundary-repeated-student-v1.json"
GENERATOR = REPO / "scripts" / "calibrate-promotion-boundary.py"
EXPECTED_SHA256 = \
    "ed429ce50d58016e01a8276f2004f5c4777f896f1a23c98eb17f81c87cb5abde"


def main() -> int:
    assert hashlib.sha256(RECORD.read_bytes()).hexdigest() == EXPECTED_SHA256
    frozen = json.loads(RECORD.read_text())
    assert frozen["design"]["looks"] == [5, 7, 9]
    assert frozen["boundary"] == 2.52
    assert frozen["paths"] == 100_000_000
    assert frozen["crossings"] == 4_901_843
    assert frozen["monte_carlo"]["crossing_probability_upper"] < \
        frozen["familywise_alpha"]
    assert frozen["passed"] is True

    # A short prefix is insufficient to certify alpha, but it must reproduce
    # the exact PCG64 stream used by the frozen 100-million-path run.
    result = subprocess.run([
        sys.executable, str(GENERATOR), "--paths", "1000000",
        "--batch-size", "100000",
    ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 1, result.stderr
    prefix = json.loads(result.stdout)
    assert prefix["crossings"] == 49_203
    assert prefix["rng"] == frozen["rng"]
    assert prefix["boundary"] == frozen["boundary"]
    print("test_promotion_boundary_calibration: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
