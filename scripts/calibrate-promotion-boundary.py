#!/usr/bin/env python3
"""Reproduce the DS4 5/7/9 repeated-Student boundary calibration."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=100_000_000)
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0xD54A53)
    parser.add_argument("--boundary", type=float, default=2.52)
    parser.add_argument("--mc-alpha", type=float, default=0.0005)
    args = parser.parse_args()
    if (args.paths <= 0 or args.batch_size <= 0 or
            not math.isfinite(args.boundary) or args.boundary <= 0 or
            not 0 < args.mc_alpha < 1):
        parser.error("paths, batch size, boundary, and mc-alpha must be positive")

    rng = np.random.default_rng(args.seed)
    crossings = 0
    remaining = args.paths
    while remaining:
        count = min(remaining, args.batch_size)
        sample = rng.standard_normal((count, 9), dtype=np.float64)
        crossed = np.zeros(count, dtype=bool)
        for look in (5, 7, 9):
            prefix = sample[:, :look]
            statistic = (np.sqrt(look) * prefix.mean(axis=1) /
                         prefix.std(axis=1, ddof=1))
            crossed |= statistic >= args.boundary
        crossings += int(crossed.sum())
        remaining -= count

    estimate = crossings / args.paths
    # Hoeffding's inequality gives a distribution-free upper confidence bound
    # for Monte Carlo error; it is deliberately more conservative than a
    # normal approximation to the simulated crossing count.
    upper = estimate + math.sqrt(math.log(1.0 / args.mc_alpha) /
                                 (2.0 * args.paths))
    result = {
        "schema_version": 1,
        "kind": "ds4-repeated-student-boundary-calibration",
        "design": {
            "looks": [5, 7, 9],
            "statistic": "sqrt(n)*mean(x[0:n])/sample_stddev(x[0:n])",
            "null_model": "iid-standard-normal",
            "crossing_rule": "any-statistic-greater-than-or-equal-boundary",
        },
        "rng": {
            "library": "numpy",
            "version": np.__version__,
            "generator": "PCG64",
            "seed": args.seed,
        },
        "paths": args.paths,
        "boundary": args.boundary,
        "crossings": crossings,
        "crossing_probability": estimate,
        "monte_carlo": {
            "method": "hoeffding-one-sided-upper",
            "confidence_level": 1.0 - args.mc_alpha,
            "crossing_probability_upper": upper,
        },
        "familywise_alpha": 0.05,
        "passed": upper < 0.05,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
