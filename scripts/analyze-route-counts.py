#!/usr/bin/env python3
"""Summarize per-expert routed-token counts from frozen top-k captures.

The input files are the existing route captures (one int32 expert id per
token/slot).  No engine path is touched and no device-to-host copy is added
to timed inference.  Counts are capped at ``--cap`` for compact reporting;
the uncapped count and assignment-weighted count are both emitted.
"""

import argparse
import glob
import os
import struct
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("route_dir")
    ap.add_argument("--pattern", default="routes_layer*_pos0_topk.i32")
    ap.add_argument("--used", type=int, default=6)
    ap.add_argument("--cap", type=int, default=64)
    args = ap.parse_args()

    paths = glob.glob(os.path.join(args.route_dir, args.pattern))
    paths.sort(key=lambda p: int(os.path.basename(p).split("layer", 1)[1].split("_", 1)[0]))
    if not paths:
        raise SystemExit("no route captures found")

    aggregate = Counter()
    weighted = Counter()
    remainders = Counter()
    print("# Route-count histogram")
    print("\n| Layer | Tokens | Active experts | Min | P50 | P90 | Max | <=3 | 4-5 | >=6 |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for path in paths:
        raw = open(path, "rb").read()
        if len(raw) % 4:
            raise SystemExit(f"unaligned capture: {path}")
        vals = struct.unpack(f"<{len(raw) // 4}i", raw)
        if len(vals) % args.used:
            raise SystemExit(f"route count is not divisible by used={args.used}: {path}")
        counts = Counter(vals).values()
        counts = sorted(counts)
        for count in counts:
            bucket = min(count, args.cap)
            aggregate[bucket] += 1
            weighted[bucket] += count
            remainders[count % 16] += 1
        layer = os.path.basename(path).split("layer", 1)[1].split("_", 1)[0]
        p50 = counts[len(counts) // 2]
        p90 = counts[int(0.9 * len(counts))]
        print(f"| {layer} | {len(vals) // args.used} | {len(counts)} | {counts[0]} | {p50} | {p90} | {counts[-1]} | {sum(c <= 3 for c in counts)} | {sum(4 <= c <= 5 for c in counts)} | {sum(c >= 6 for c in counts)} |")

    print("\n## Aggregate")
    print(f"files={len(paths)} expert_entries={sum(aggregate.values())}")
    print("assignment_count_histogram=" + ",".join(f"{k}:{aggregate[k]}" for k in sorted(aggregate)))
    print("assignment_weighted_histogram=" + ",".join(f"{k}:{weighted[k]}" for k in sorted(weighted)))
    print("remainder_mod16=" + ",".join(f"{k}:{remainders[k]}" for k in sorted(remainders)))


if __name__ == "__main__":
    main()
