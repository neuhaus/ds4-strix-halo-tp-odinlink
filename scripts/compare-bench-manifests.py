#!/usr/bin/env python3
"""Compare recorded workloads and effective per-rank settings, without eval.

Only run labels are ignored by default. Declare intended source/artifact or
environment changes explicitly. This checks manifests, not build provenance,
quality, actual rank execution, or promotion readiness.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import sys

ENV_FIELDS = {"common_env", "worker_env", "coordinator_env", "extra_env"}
RUN_FIELDS = {"tag", "run_id"}


def read_manifest(path):
    fields = {}
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            raise ValueError(f"{path}: invalid or duplicate field {key!r}")
        if key in ENV_FIELDS:
            effective = {}
            for assignment in shlex.split(value):
                name, separator, setting = assignment.partition("=")
                if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
                    raise ValueError(f"{path}: invalid assignment in {key}")
                # env processes assignments in order; the last one wins.
                effective[name] = setting
            value = effective
        fields[key] = value
    required = {"source_commit", "ds4_sha256", "model_sample_sha256", "frontier",
                "generated_tokens", "rdma_profile", "worker_env", "coordinator_env"}
    if required - fields.keys():
        raise ValueError(f"{path}: missing fields {sorted(required - fields.keys())}")
    return fields


def compare(left, right, allow_fields=(), allow_env=()):
    rejected, allowed = [], []
    for field in sorted(left.keys() | right.keys()):
        if field in RUN_FIELDS:
            continue
        a, b = left.get(field), right.get(field)
        if a == b:
            continue
        if field in ENV_FIELDS and isinstance(a, dict) and isinstance(b, dict):
            for name in sorted(a.keys() | b.keys()):
                if a.get(name) != b.get(name):
                    entry = {"field": field, "env": name, "before": a.get(name), "after": b.get(name)}
                    (allowed if name in allow_env else rejected).append(entry)
        else:
            entry = {"field": field, "before": a, "after": b}
            (allowed if field in allow_fields else rejected).append(entry)
    return {"matched": not rejected, "allowed_changes": allowed, "unexpected_changes": rejected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control")
    parser.add_argument("candidate")
    parser.add_argument("--allow-field", action="append", default=[])
    parser.add_argument("--allow-env", action="append", default=[])
    args = parser.parse_args()
    if set(args.allow_field) & ENV_FIELDS:
        parser.error("use --allow-env NAME for individual runtime settings")
    try:
        result = compare(read_manifest(args.control), read_manifest(args.candidate),
                         args.allow_field, args.allow_env)
    except (OSError, ValueError) as error:
        print(f"manifest comparison failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["matched"] else 1


if __name__ == "__main__":
    sys.exit(main())
