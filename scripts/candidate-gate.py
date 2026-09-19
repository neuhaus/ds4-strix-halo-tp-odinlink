#!/usr/bin/env python3
"""Create and validate DS4 performance-candidate promotion dossiers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import secrets
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds4_gate_controls import (  # noqa: E402
    ControlError,
    append_event,
    artifact_hashes,
    candidate_is_open,
    candidate_scope,
    candidate_values,
    evaluate_calibration,
    event_sequences,
    governance_mutation,
    journal_entries,
    register_control,
    verifier_sha256,
)


ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
LANES = {"A", "B", "C"}
MIN_TEACHER_STEPS = 300
MIN_QUALITY_CASES = 100
MIN_QUALITY_TOKENS = 2289
COMMON_KINDS = {"promotion-proof"}
REVIEW_KINDS = {"fable-review", "grok-review"}
LANE_KINDS = {
    "A": set(),
    "B": {"numerical-envelope", "reference-score"},
    "C": {"numerical-envelope", "reference-score"},
}
DSPARK_KINDS = {"same-stack-ordinary", "verifier-logits", "dspark-acceptance"}
BASE_WORKLOAD_MANIFEST_FIELDS = (
    "prompt_sha256", "frontier", "generated_tokens", "context",
    "prefill_chunk", "dspark",
)
ENV_FIELDS = ("common_env", "worker_env", "coordinator_env", "extra_env")
RANK_ENV_FIELDS = ("worker_env", "coordinator_env")
REPEATED_STUDENT_CALIBRATION_SHA256 = \
    "ed429ce50d58016e01a8276f2004f5c4777f896f1a23c98eb17f81c87cb5abde"
TIMING_NOISE_PAIR_COUNT = 9
TIMING_NOISE_MAX_ABS_SKEWNESS = 1.0
TIMING_NOISE_MAX_STANDARDIZED_RESIDUAL = 3.0
TIMING_NOISE_MAX_BIAS_STATISTIC = 2.52
MAX_HEADLINE_PAIR_ATTEMPTS = 2
MAX_HEADLINE_INVALIDATIONS = 2
VOLATILE_BENCH_ENV = {"DS4_BENCH_RUN_ID"}


class GateError(RuntimeError):
    pass


def workload_manifest_fields(workload: dict) -> tuple[str, ...]:
    """Return identity fields, including an explicitly scoped prefill batch."""
    fields = BASE_WORKLOAD_MANIFEST_FIELDS
    if "prefill_batch" in workload:
        try:
            batch = int(workload["prefill_batch"])
        except (TypeError, ValueError) as error:
            raise GateError("workload prefill_batch is not an integer") from error
        if batch < 2 or batch > 1024:
            raise GateError(
                "promotion prefill_batch must describe batched prefill (2..1024)")
        fields += ("prefill_batch",)
    return fields


def validate_performance_thresholds(value: object, label: str) -> dict:
    required = {
        "schema_version", "method", "qualification_pairs", "merge_looks",
        "formal_test", "futility_confidence_level", "minimum_gain",
        "maximum_untargeted_regression", "maximum_control_regression",
        "absolute_floor", "screens", "required_provider",
    }
    if (not isinstance(value, dict) or set(value) != required or
            value.get("schema_version") != 2 or
            value.get("method") != "paired-log-ratio-two-tier-v2" or
            value.get("qualification_pairs") != 3 or
            value.get("merge_looks") != [5, 7, 9] or
            value.get("futility_confidence_level") != 0.95 or
            value.get("required_provider") != "roce-v2"):
        raise GateError(f"{label} has an unsupported sequential design")
    formal_test = value["formal_test"]
    if not isinstance(formal_test, dict):
        raise GateError(f"{label}.formal_test must be an object")
    if formal_test.get("kind") == "repeated-student-v1":
        if (set(formal_test) != {
                "kind", "boundary", "familywise_alpha",
                "design_calibration_sha256", "timing_noise_qualification_sha256"} or
                formal_test.get("boundary") != 2.52 or
                formal_test.get("familywise_alpha") != 0.05 or
                formal_test.get("design_calibration_sha256") !=
                    REPEATED_STUDENT_CALIBRATION_SHA256 or
                not re.fullmatch(r"[0-9a-f]{64}", str(
                    formal_test.get("timing_noise_qualification_sha256", "")))):
            raise GateError(f"{label} repeated-Student method is not frozen and qualified")
    elif formal_test.get("kind") == "exact-sign-fixed-nine-v1":
        if (set(formal_test) != {
                "kind", "pairs", "minimum_positive", "ties",
                "familywise_alpha"} or
                formal_test.get("pairs") != 9 or
                formal_test.get("minimum_positive") != 8 or
                formal_test.get("ties") != "fail" or
                formal_test.get("familywise_alpha") != 0.05):
            raise GateError(f"{label} exact-sign fallback is invalid")
    else:
        raise GateError(f"{label} has an unsupported formal test")
    for name in ("minimum_gain", "maximum_untargeted_regression",
                 "maximum_control_regression"):
        number = value[name]
        if (not isinstance(number, (int, float)) or
                not math.isfinite(number) or not 0 <= number <= 0.10):
            raise GateError(f"{label}.{name} must be between zero and 0.10")
    floors = value["absolute_floor"]
    screens = value["screens"]
    if (not isinstance(floors, dict) or set(floors) != {"prefill", "decode"} or
            not isinstance(screens, dict) or
            set(screens) != {"max_prefill_regression", "max_decode_regression"}):
        raise GateError(f"{label} floor or screen fields are invalid")
    for name, number in {**floors, **screens}.items():
        if (not isinstance(number, (int, float)) or
                not math.isfinite(number) or number < 0 or
                (name.startswith("max_") and number > 0.10)):
            raise GateError(f"{label}.{name} is invalid")
    return value


def _bound_timing_artifact(value: object, root: Path, label: str) -> Path:
    if (not isinstance(value, dict) or set(value) != {"path", "sha256"} or
            not re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", "")))):
        raise GateError(f"{label} has an invalid content-addressed artifact")
    path = Path(str(value["path"])).expanduser().resolve()
    if path != root and root not in path.parents:
        raise GateError(f"{label} escapes DS4_RESEARCH_ROOT")
    if not path.is_file() or sha256(path) != value["sha256"]:
        raise GateError(f"{label} hash mismatch")
    return path


def _timing_noise_run(repo: Path, root: Path, value: object, *,
                      baseline: dict, model_path: Path, provider: str,
                      pair_id: str, order: str, arm: str) -> dict:
    required = {"csv", "manifest", "coordinator_log", "worker_log",
                "worker_status"}
    if not isinstance(value, dict) or set(value) != required:
        raise GateError("timing-noise run must bind CSV, manifest, logs, and status")
    paths = {
        name: _bound_timing_artifact(item, root, f"timing-noise {name}")
        for name, item in value.items()
    }
    row, manifest, adjacent_manifest = read_benchmark(paths["csv"])
    if paths["manifest"] != adjacent_manifest:
        raise GateError("timing-noise manifest must be adjacent to its CSV")
    verify_manifest_run_id_environment(manifest, "timing-noise manifest")
    status = read_manifest(paths["worker_status"])
    if status.get("exit_code") != "0" or status.get("signal") != "0":
        raise GateError("timing-noise worker did not exit cleanly")

    key = baseline["key"]
    workload = key["workload"]
    anchor = baseline["reference"]["performance"][provider]
    if (manifest.get("pair_id") != pair_id or
            manifest.get("pair_order") != order or
            manifest.get("pair_arm") != arm or
            manifest.get("source_dirty") != "0" or
            manifest.get("source_commit") != key.get("source_commit") or
            manifest.get("model") != str(model_path) or
            manifest.get("model_size") != str(key.get("model_size")) or
            manifest.get("model_sample_sha256") != key.get("model_sample_sha256") or
            manifest.get("toolchain_id") != key.get("toolchain_id") or
            manifest.get("rdma_profile") != provider or
            manifest.get("dspark") != "0" or
            manifest.get("ds4_sha256") != anchor["ds4_sha256"] or
            manifest.get("peer_ds4_sha256") != anchor["ds4_sha256"]):
        raise GateError("timing-noise run differs from the baseline scope")
    bench_binary = verify_benchmark_producer(
        repo, manifest, key["source_commit"], "timing-noise benchmark")
    if bench_binary != anchor.get("ds4_bench_tp_sha256"):
        raise GateError("timing-noise run used a different benchmark binary")
    environment = normalized_manifest_environment(
        manifest, "timing-noise manifest")
    if environment != anchor["environment"]:
        raise GateError("timing-noise run differs from the baseline environment")
    for field in workload_manifest_fields(workload):
        if manifest.get(field) != str(workload.get(field, "")):
            raise GateError(f"timing-noise run differs in {field}")
    verify_tp_layout_manifest(
        manifest, tp_layout_contract(key, "timing-noise baseline"),
        "timing-noise run layout")
    if row["gen_token_fnv64"].lower() != baseline["reference"]["fnv64"]:
        raise GateError("timing-noise run changed the baseline fingerprint")
    command = [
        str(repo / "scripts" / "check-ds4-bench-result.sh"),
        str(paths["csv"]), str(paths["coordinator_log"]),
        str(paths["worker_log"]), baseline["reference"]["fnv64"],
        str(workload["generated_tokens"]), "0", provider,
        manifest.get("coordinator_rdma_device", ""),
        manifest.get("rdma_gid_index", ""),
        manifest.get("worker_rdma_device", ""),
        manifest.get("run_id", ""),
    ]
    checked = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    if checked.returncode != 0:
        detail = checked.stderr.strip() or checked.stdout.strip()
        raise GateError(f"timing-noise transport/cache validation failed: {detail}")
    try:
        metrics = {
            "prefill": float(row["prefill_tps"]),
            "decode": float(row["gen_tps"]),
        }
    except (KeyError, ValueError) as error:
        raise GateError("timing-noise run has invalid throughput") from error
    if any(not math.isfinite(item) or item <= 0 for item in metrics.values()):
        raise GateError("timing-noise throughput must be positive and finite")
    return {"paths": paths, "manifest": manifest, "metrics": metrics}


def _timing_noise_metric(control: list[float], candidate: list[float],
                         orders: list[str], maximum_regression: float) -> dict:
    effects = [math.log(right / left)
               for left, right in zip(control, candidate)]
    count = len(effects)
    mean = sum(effects) / count
    variance = sum((item - mean) ** 2 for item in effects) / (count - 1)
    sigma = math.sqrt(variance)
    if not math.isfinite(sigma) or sigma <= 0:
        raise GateError("timing-noise sample variance must be positive")
    second = sum((item - mean) ** 2 for item in effects) / count
    third = sum((item - mean) ** 3 for item in effects) / count
    skewness = (math.sqrt(count * (count - 1)) / (count - 2) *
                third / (second ** 1.5))
    maximum_residual = max(abs(item - mean) / sigma for item in effects)
    label_bias = abs(mean) * math.sqrt(count) / sigma
    position_effects = [item if order == "AB" else -item
                        for item, order in zip(effects, orders)]
    position_mean = sum(position_effects) / count
    position_variance = sum(
        (item - position_mean) ** 2 for item in position_effects) / (count - 1)
    position_sigma = math.sqrt(position_variance)
    position_bias = (abs(position_mean) * math.sqrt(count) / position_sigma
                     if position_sigma > 0 else
                     (0.0 if position_mean == 0 else math.inf))
    geometric_mean_change = math.expm1(mean)
    if (abs(skewness) > TIMING_NOISE_MAX_ABS_SKEWNESS or
            maximum_residual > TIMING_NOISE_MAX_STANDARDIZED_RESIDUAL or
            label_bias >= TIMING_NOISE_MAX_BIAS_STATISTIC or
            position_bias >= TIMING_NOISE_MAX_BIAS_STATISTIC or
            abs(geometric_mean_change) > maximum_regression):
        raise GateError(
            "timing-noise observations are incompatible with repeated Student")
    return {
        "pairs": count,
        "mean_log_ratio": mean,
        "sigma_log": sigma,
        "sample_skewness": skewness,
        "maximum_standardized_residual": maximum_residual,
        "label_bias_statistic": label_bias,
        "position_bias_statistic": position_bias,
        "geometric_mean_change": geometric_mean_change,
    }


def verify_performance_method_evidence(repo: Path, root: Path, contract: dict,
                                       noise_path_value: object,
                                       label: str, *, baseline: dict) -> dict:
    """Bind the formal timing method to its design and scope evidence."""
    formal_test = contract["formal_test"]
    if formal_test["kind"] == "exact-sign-fixed-nine-v1":
        if noise_path_value is not None:
            raise GateError(f"{label} exact-sign method must not claim timing-model evidence")
        return {"formal_test": formal_test["kind"]}

    calibration_path = repo / "scripts" / "promotion-boundary-repeated-student-v1.json"
    if (not calibration_path.is_file() or
            sha256(calibration_path) != REPEATED_STUDENT_CALIBRATION_SHA256):
        raise GateError("versioned repeated-Student design calibration changed")
    noise_path = Path(str(noise_path_value or "")).expanduser().resolve()
    if noise_path != root and root not in noise_path.parents:
        raise GateError(f"{label} timing-noise evidence escapes DS4_RESEARCH_ROOT")
    if (not noise_path.is_file() or
            sha256(noise_path) != formal_test["timing_noise_qualification_sha256"]):
        raise GateError(f"{label} timing-noise evidence hash mismatch")
    try:
        noise = json.loads(noise_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GateError(f"{label} timing-noise evidence is not JSON") from error
    required = {
        "schema_version", "kind", "scope_sha256", "formal_test", "provider",
        "source_commit", "ds4_sha256", "model", "pairs",
    }
    if (not isinstance(noise, dict) or set(noise) != required or
            noise.get("schema_version") != 2 or
            noise.get("kind") != "ds4-timing-noise-qualification" or
            noise.get("formal_test") != "repeated-student-v1" or
            noise.get("scope_sha256") != baseline_scope_sha256(baseline) or
            noise.get("provider") != contract["required_provider"] or
            noise.get("source_commit") != baseline["key"].get("source_commit") or
            noise.get("ds4_sha256") != baseline["reference"]["performance"]
                [contract["required_provider"]]["ds4_sha256"] or
            not isinstance(noise.get("pairs"), list) or
            len(noise["pairs"]) != TIMING_NOISE_PAIR_COUNT):
        raise GateError(f"{label} timing-noise qualification is invalid")
    model_path = _bound_timing_artifact(noise["model"], root, "timing-noise model")
    if (sha256(model_path) != baseline["key"].get("model_sha256") or
            model_path.stat().st_size != baseline["key"].get("model_size")):
        raise GateError(f"{label} timing-noise model differs from the baseline")

    pair_ids: set[str] = set()
    run_ids: set[str] = set()
    artifact_digests: set[str] = set()
    artifact_paths: set[Path] = set()
    orders: list[str] = []
    values = {"prefill": {"control": [], "candidate": []},
              "decode": {"control": [], "candidate": []}}
    for index, pair in enumerate(noise["pairs"]):
        if (not isinstance(pair, dict) or
                set(pair) != {"pair_id", "order", "control", "candidate"}):
            raise GateError(f"{label} timing-noise pair has invalid fields")
        pair_id = str(pair["pair_id"])
        order = str(pair["order"])
        if (not ID_RE.fullmatch(pair_id) or pair_id in pair_ids or
                order not in {"AB", "BA"} or
                (orders and orders[-1] == order)):
            raise GateError(f"{label} timing-noise schedule is not unique and alternating")
        pair_ids.add(pair_id)
        orders.append(order)
        runs = {}
        for arm in ("control", "candidate"):
            runs[arm] = _timing_noise_run(
                repo, root, pair[arm], baseline=baseline,
                model_path=model_path, provider=contract["required_provider"],
                pair_id=pair_id, order=order, arm=arm)
            run_id = runs[arm]["manifest"].get("run_id", "")
            if not run_id or run_id in run_ids:
                raise GateError(f"{label} timing-noise run IDs are not distinct")
            run_ids.add(run_id)
            for artifact_name, artifact_value in pair[arm].items():
                artifact_path = Path(str(artifact_value["path"])).resolve()
                if artifact_path in artifact_paths:
                    raise GateError(f"{label} timing-noise reuses a run artifact")
                artifact_paths.add(artifact_path)
                if artifact_name in {"csv", "manifest"}:
                    artifact_digests.add(str(artifact_value["sha256"]))
        control_id = runs["control"]["manifest"]["run_id"]
        candidate_id = runs["candidate"]["manifest"]["run_id"]
        if ((order == "AB" and not control_id < candidate_id) or
                (order == "BA" and not candidate_id < control_id)):
            raise GateError(f"{label} timing-noise run IDs contradict pair order")
        compared = subprocess.run([
            sys.executable, str(repo / "scripts" / "compare-bench-manifests.py"),
            str(runs["control"]["paths"]["manifest"]),
            str(runs["candidate"]["paths"]["manifest"]),
            "--allow-field", "pair_arm",
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if compared.returncode != 0:
            detail = compared.stderr.strip() or compared.stdout.strip()
            raise GateError(f"{label} timing-noise pair is not self-matched: {detail}")
        for metric in values:
            for arm in ("control", "candidate"):
                values[metric][arm].append(runs[arm]["metrics"][metric])
    if (abs(orders.count("AB") - orders.count("BA")) > 1 or
            set(orders) != {"AB", "BA"}):
        raise GateError(f"{label} timing-noise schedule is not order-balanced")

    for dossier, candidate in candidate_values(root):
        candidate_hashes = artifact_hashes(candidate)
        for evidence in candidate.get("evidence", []):
            if not isinstance(evidence, dict):
                continue
            path = Path(str(evidence.get("path", "")))
            if not path.is_absolute():
                path = dossier / path
            if path.is_file():
                try:
                    candidate_hashes.update(artifact_hashes(json.loads(
                        path.read_text(encoding="utf-8"))))
                except (OSError, json.JSONDecodeError):
                    pass
        if artifact_digests & candidate_hashes:
            raise GateError(
                f"{label} timing-noise artifacts overlap candidate evidence")

    metrics = {
        metric: _timing_noise_metric(
            arms["control"], arms["candidate"], orders,
            contract["maximum_control_regression"])
        for metric, arms in values.items()
    }
    return {
        "formal_test": formal_test["kind"],
        "design_calibration_sha256": REPEATED_STUDENT_CALIBRATION_SHA256,
        "timing_noise_qualification_sha256": sha256(noise_path),
        "scope_sha256": noise["scope_sha256"],
        "provider": noise["provider"],
        "pairs": TIMING_NOISE_PAIR_COUNT,
        "metrics": metrics,
    }


def validate_promotion_intent(value: object) -> dict:
    if (not isinstance(value, dict) or
            set(value) != {"target_metrics", "candidate_switches",
                           "headline_pair_order", "invalidation_policy"}):
        raise GateError("candidate promotion_intent has invalid fields")
    targets = value["target_metrics"]
    switches = value["candidate_switches"]
    order = value["headline_pair_order"]
    if (not isinstance(targets, list) or not targets or
            len(set(targets)) != len(targets) or
            set(targets) - {"prefill", "decode"} or
            not isinstance(switches, dict) or
            not isinstance(order, list) or len(order) != 9 or
            any(item not in {"AB", "BA"} for item in order) or
            any(left == right for left, right in zip(order, order[1:])) or
            value["invalidation_policy"] !=
                "whole-pair-protocol-failure-only-v1"):
        raise GateError("candidate promotion intent is invalid")
    for name, arms in switches.items():
        if (not isinstance(name, str) or
                not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or
                not isinstance(arms, dict) or set(arms) != {"control", "candidate"} or
                any(not isinstance(item, str) or "\n" in item
                    for item in arms.values()) or
                arms["control"] == arms["candidate"]):
            raise GateError("candidate promotion intent has an invalid switch")
    return value


def parse_switch_declarations(values: list[str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for declaration in values:
        name, separator, arms = declaration.partition("=")
        control, comma, candidate = arms.partition(",")
        if (not separator or not comma or
                not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or
                name in result or "\n" in control or "\n" in candidate or
                control == candidate):
            raise GateError(
                "--switch requires one unique NAME=CONTROL,CANDIDATE declaration")
        result[name] = {"control": control, "candidate": candidate}
    return result


def parse_env(value: str, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        assignments = shlex.split(value)
    except ValueError as error:
        raise GateError(f"{label} has malformed shell quoting") from error
    for assignment in assignments:
        name, separator, setting = assignment.partition("=")
        if (not separator or
                not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or
                name in result):
            raise GateError(f"{label} has a duplicate or invalid assignment")
        result[name] = setting
    return result


def normalized_manifest_environment(manifest: dict[str, str], label: str) -> dict[str, str]:
    """Return stable effective settings while retaining per-run IDs in raw evidence."""
    result = {}
    for field in ENV_FIELDS:
        value = manifest.get(field)
        if not isinstance(value, str):
            raise GateError(f"{label} is missing {field}")
        environment = parse_env(value, f"{label} {field}")
        for name in VOLATILE_BENCH_ENV:
            environment.pop(name, None)
        result[field] = shlex.join(
            f"{name}={setting}" for name, setting in sorted(environment.items()))
    return result


def verify_manifest_run_id_environment(manifest: dict[str, str], label: str) -> None:
    run_id = manifest.get("run_id", "")
    if not run_id:
        raise GateError(f"{label} has no run_id")
    for field in ("common_env", *RANK_ENV_FIELDS):
        environment = parse_env(manifest.get(field, ""), f"{label} {field}")
        if environment.get("DS4_BENCH_RUN_ID") != run_id:
            raise GateError(f"{label} {field} does not bind its run_id")


def manifest_switch_values(manifest: dict[str, str], switches: dict,
                           label: str) -> dict[str, str]:
    rank_values = []
    for field in RANK_ENV_FIELDS:
        if field not in manifest:
            raise GateError(f"{label} is missing {field}")
        rank_values.append(parse_env(manifest[field], f"{label} {field}"))
    result = {}
    for name in switches:
        values = {environment.get(name) for environment in rank_values}
        if len(values) != 1:
            raise GateError(f"{label} ranks disagree on initialized switch {name}")
        result[name] = values.pop()
    return result


def run_git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def committed_source_sha256(repo: Path, commit: str, relative: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "blob", f"{commit}:{relative}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise GateError(
            f"cannot resolve {relative} from benchmark source commit {commit}")
    return hashlib.sha256(result.stdout).hexdigest()


def verify_benchmark_producer(repo: Path, manifest: dict[str, str],
                              source_commit: str, label: str) -> str:
    bench_binary = manifest.get("ds4_bench_tp_sha256", "")
    producer_source = manifest.get("ds4_bench_producer_source_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", bench_binary):
        raise GateError(f"{label} has no valid benchmark binary identity")
    if (not re.fullmatch(r"[0-9a-f]{64}", producer_source) or
            producer_source != committed_source_sha256(
                repo, source_commit, "ds4_bench.c")):
        raise GateError(
            f"{label} producer binary does not match committed ds4_bench.c")
    return bench_binary


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve_scoped_path(repo: Path, root: Path, item: dict) -> Path:
    scope = item.get("scope")
    relative = item.get("path")
    if scope not in {"repo", "research", "system"} or not isinstance(relative, str) or not relative:
        raise GateError("oracle closure entries require repo/research/system scope and a path")
    if scope == "system":
        path = Path(relative)
        if not path.is_absolute():
            raise GateError("system-scoped oracle closure paths must be absolute")
        return path.resolve()
    if Path(relative).is_absolute():
        raise GateError("repo/research oracle closure paths must be relative")
    base = repo if scope == "repo" else root
    path = (base / relative).resolve()
    if path == base or base not in path.parents:
        raise GateError("oracle closure path escapes its declared scope")
    return path


def verify_generator_closure(repo: Path, root: Path,
                             generator: dict) -> tuple[Path, Path]:
    closure = generator.get("closure")
    entrypoint = generator.get("entrypoint")
    runner = generator.get("runner")
    environment = generator.get("environment")
    if (not isinstance(closure, list) or not closure or
            not isinstance(entrypoint, dict) or not isinstance(runner, dict) or
            not isinstance(environment, dict) or not environment or
            not re.fullmatch(r"[0-9a-f]{64}",
                             str(generator.get("environment_id", ""))) or
            generator["environment_id"] != canonical_sha256(environment) or
            generator.get("closure_sha256") != canonical_sha256(closure)):
        raise GateError("approved oracle generator has an invalid dependency closure")
    resolved: dict[tuple[str, str], Path] = {}
    for item in closure:
        if (not isinstance(item, dict) or
                not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))):
            raise GateError("approved oracle closure has an invalid file hash")
        path = resolve_scoped_path(repo, root, item)
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise GateError(f"approved oracle dependency differs: {path}")
        key = (str(item["scope"]), str(item["path"]))
        if key in resolved:
            raise GateError("approved oracle closure contains duplicate paths")
        resolved[key] = path
    entry_key = (str(entrypoint.get("scope")), str(entrypoint.get("path")))
    runner_key = (str(runner.get("scope")), str(runner.get("path")))
    if entry_key not in resolved or entrypoint.get("sha256") != sha256(resolved[entry_key]):
        raise GateError("oracle generator entrypoint is not bound by its closure")
    if runner_key not in resolved or runner.get("sha256") != sha256(resolved[runner_key]):
        raise GateError("oracle generator runner is not bound by its closure")
    return resolved[entry_key], resolved[runner_key]


def validate_numerical_thresholds(thresholds: object, label: str) -> dict:
    if not isinstance(thresholds, dict):
        raise GateError(f"{label} thresholds are missing")
    if thresholds.get("schema_version") == 2:
        required = {
            "schema_version", "min_teacher_steps", "allow_quality_difference",
            "decision", "distribution", "safety",
        }
        if set(thresholds) != required:
            raise GateError(f"{label} v2 thresholds have invalid fields")
        if (not isinstance(thresholds["min_teacher_steps"], int) or
                thresholds["min_teacher_steps"] < MIN_TEACHER_STEPS):
            raise GateError(
                f"{label} must require at least {MIN_TEACHER_STEPS} teacher steps")
        if type(thresholds["allow_quality_difference"]) is not bool:
            raise GateError(f"{label}.allow_quality_difference must be boolean")
        decision = thresholds["decision"]
        distribution = thresholds["distribution"]
        safety = thresholds["safety"]
        if not all(isinstance(item, dict)
                   for item in (decision, distribution, safety)):
            raise GateError(f"{label} v2 threshold sections must be objects")
        if set(decision) != {
                "e_bound", "confidence_level",
                "max_near_tie_cluster_rate_upper"}:
            raise GateError(f"{label}.decision has invalid fields")
        if set(distribution) != {
                "bootstrap_method", "bootstrap_resamples", "bootstrap_seed",
                "cluster_mode", "block_size", "min_clusters",
                "max_mean_kl_upper", "max_mean_tvd_upper",
                "max_mean_teacher_nll_delta_upper",
                "min_same_top1_cluster_rate_lower", "soft_limits",
                "max_soft_exceedance_cluster_rate_upper"}:
            raise GateError(f"{label}.distribution has invalid fields")
        if distribution.get("bootstrap_method") != "bca":
            raise GateError(f"{label} bootstrap method must be bca")
        if distribution.get("cluster_mode") != "case-or-contiguous-block":
            raise GateError(
                f"{label} cluster mode must be case-or-contiguous-block")
        if (not isinstance(distribution["bootstrap_resamples"], int) or
                distribution["bootstrap_resamples"] < 1000 or
                not isinstance(distribution["bootstrap_seed"], int) or
                isinstance(distribution["bootstrap_seed"], bool) or
                distribution["bootstrap_seed"] < 0):
            raise GateError(f"{label} has invalid deterministic bootstrap settings")
        if (not isinstance(distribution["block_size"], int) or
                isinstance(distribution["block_size"], bool) or
                distribution["block_size"] < 4 or
                not isinstance(distribution["min_clusters"], int) or
                isinstance(distribution["min_clusters"], bool) or
                distribution["min_clusters"] < 2):
            raise GateError(f"{label} has invalid cluster coverage settings")
        soft = distribution["soft_limits"]
        if not isinstance(soft, dict) or set(soft) != {
                "centered_p99_abs", "centered_nrms", "kl", "tvd"}:
            raise GateError(f"{label}.soft_limits has invalid fields")
        if set(safety) != {
                "max_centered_abs", "max_centered_nrms", "max_kl", "max_tvd",
                "max_abs_teacher_nll_delta"}:
            raise GateError(f"{label}.safety has invalid fields")
        scalar_values = {
            "e_bound": decision["e_bound"],
            "confidence_level": decision["confidence_level"],
            "max_near_tie_cluster_rate_upper":
                decision["max_near_tie_cluster_rate_upper"],
            "max_mean_kl_upper": distribution["max_mean_kl_upper"],
            "max_mean_tvd_upper": distribution["max_mean_tvd_upper"],
            "max_mean_teacher_nll_delta_upper":
                distribution["max_mean_teacher_nll_delta_upper"],
            "min_same_top1_cluster_rate_lower":
                distribution["min_same_top1_cluster_rate_lower"],
            "max_soft_exceedance_cluster_rate_upper":
                distribution["max_soft_exceedance_cluster_rate_upper"],
            **{f"soft.{key}": value for key, value in soft.items()},
            **{f"safety.{key}": value for key, value in safety.items()},
        }
        for key, value in scalar_values.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise GateError(f"{label}.{key} must be finite")
        for key, value in scalar_values.items():
            if key not in {"max_mean_teacher_nll_delta_upper"} and value < 0:
                raise GateError(f"{label}.{key} must be nonnegative")
        if not 0.5 < decision["confidence_level"] < 1.0:
            raise GateError(f"{label}.confidence_level must be between 0.5 and 1")
        for key in ("max_near_tie_cluster_rate_upper",
                    "min_same_top1_cluster_rate_lower",
                    "max_soft_exceedance_cluster_rate_upper"):
            if not 0 <= scalar_values[key] <= 1:
                raise GateError(f"{label}.{key} must be between zero and one")
        soft_to_safety = {
            "centered_p99_abs": "max_centered_abs",
            "centered_nrms": "max_centered_nrms",
            "kl": "max_kl",
            "tvd": "max_tvd",
        }
        for soft_key, safety_key in soft_to_safety.items():
            if soft[soft_key] > safety[safety_key]:
                raise GateError(
                    f"{label}.soft_limits.{soft_key} exceeds hard safety {safety_key}")
        if (distribution["max_mean_kl_upper"] > safety["max_kl"] or
                distribution["max_mean_tvd_upper"] > safety["max_tvd"] or
                abs(distribution["max_mean_teacher_nll_delta_upper"]) >
                safety["max_abs_teacher_nll_delta"]):
            raise GateError(f"{label} distribution limit exceeds hard safety")
        return thresholds
    required = {"e_bound", "max_abs", "p99_abs", "nmse", "tvd", "kl",
                "min_top5_overlap", "min_top20_overlap", "min_teacher_steps"}
    if required - thresholds.keys():
        raise GateError(f"{label} thresholds are incomplete")
    if set(thresholds) - (required | {"allow_quality_difference"}):
        raise GateError(f"{label} thresholds contain unknown keys")
    for key in ("e_bound", "max_abs", "p99_abs", "nmse", "tvd", "kl"):
        value = thresholds[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise GateError(f"{label}.{key} must be finite and nonnegative")
    for key, limit in (("min_top5_overlap", 5), ("min_top20_overlap", 20)):
        if not isinstance(thresholds[key], int) or not 0 <= thresholds[key] <= limit:
            raise GateError(f"{label}.{key} is outside its valid range")
    if (not isinstance(thresholds["min_teacher_steps"], int) or
            thresholds["min_teacher_steps"] < MIN_TEACHER_STEPS):
        raise GateError(f"{label} must require at least {MIN_TEACHER_STEPS} teacher steps")
    if ("allow_quality_difference" in thresholds and
            type(thresholds["allow_quality_difference"]) is not bool):
        raise GateError(f"{label}.allow_quality_difference must be boolean")
    return thresholds


def validate_quality_thresholds(thresholds: object, label: str) -> dict:
    if not isinstance(thresholds, dict):
        raise GateError(f"{label} thresholds are missing")
    if thresholds.get("schema_version") == 2:
        required = {
            "schema_version", "min_cases", "min_target_tokens", "bootstrap",
            "nll", "api",
        }
        if set(thresholds) != required:
            raise GateError(f"{label} v2 thresholds have invalid fields")
        if (not isinstance(thresholds["min_cases"], int) or
                thresholds["min_cases"] < MIN_QUALITY_CASES or
                not isinstance(thresholds["min_target_tokens"], int) or
                thresholds["min_target_tokens"] < MIN_QUALITY_TOKENS):
            raise GateError(f"{label} coverage is below production floors")
        bootstrap = thresholds["bootstrap"]
        nll = thresholds["nll"]
        api = thresholds["api"]
        if not all(isinstance(item, dict) for item in (bootstrap, nll, api)):
            raise GateError(f"{label} v2 threshold sections must be objects")
        if (set(bootstrap) != {
                "method", "resamples", "seed", "nll_confidence_level",
                "api_confidence_level"} or bootstrap.get("method") != "bca"):
            raise GateError(f"{label} has an invalid bootstrap contract")
        if (not isinstance(bootstrap["resamples"], int) or
                bootstrap["resamples"] < 1000 or
                not isinstance(bootstrap["seed"], int) or
                isinstance(bootstrap["seed"], bool) or bootstrap["seed"] < 0):
            raise GateError(f"{label} has invalid deterministic bootstrap settings")
        for key in ("nll_confidence_level", "api_confidence_level"):
            value = bootstrap[key]
            if not isinstance(value, (int, float)) or not 0.5 < value < 1.0:
                raise GateError(f"{label}.{key} must be between 0.5 and 1")
        if set(nll) != {"max_delta_upper", "max_case_delta"}:
            raise GateError(f"{label}.nll has invalid fields")
        if (set(api) != {"required", "min_cases", "min_top1_delta_lower",
                         "min_pair_delta_lower"} or
                type(api.get("required")) is not bool or
                not isinstance(api.get("min_cases"), int) or
                api["min_cases"] < 0):
            raise GateError(f"{label}.api has invalid fields")
        for key, value in {
                "nll.max_delta_upper": nll["max_delta_upper"],
                "nll.max_case_delta": nll["max_case_delta"],
                "api.min_top1_delta_lower": api["min_top1_delta_lower"],
                "api.min_pair_delta_lower": api["min_pair_delta_lower"],
        }.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise GateError(f"{label}.{key} must be finite")
        if (nll["max_case_delta"] < 0 or
                nll["max_delta_upper"] > nll["max_case_delta"]):
            raise GateError(
                f"{label} aggregate NLL limit exceeds its hard per-case cap")
        for key in ("min_top1_delta_lower", "min_pair_delta_lower"):
            if not -1 <= api[key] <= 1:
                raise GateError(f"{label}.api.{key} must be between -1 and 1")
        return thresholds
    required = {"min_cases", "min_target_tokens", "max_mean_nll_delta",
                "max_ci95_high_nll_delta", "min_api_top1_rate_delta",
                "min_api_pair_rate_delta"}
    if required - thresholds.keys():
        raise GateError(f"{label} thresholds are incomplete")
    if set(thresholds) != required:
        raise GateError(f"{label} thresholds contain unknown keys")
    if (not isinstance(thresholds["min_cases"], int) or
            thresholds["min_cases"] < MIN_QUALITY_CASES or
            not isinstance(thresholds["min_target_tokens"], int) or
            thresholds["min_target_tokens"] < MIN_QUALITY_TOKENS):
        raise GateError(f"{label} coverage is below production floors")
    for key in ("max_mean_nll_delta", "max_ci95_high_nll_delta",
                "min_api_top1_rate_delta", "min_api_pair_rate_delta"):
        value = thresholds[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise GateError(f"{label}.{key} must be finite")
    if thresholds["max_ci95_high_nll_delta"] <= 0:
        raise GateError(f"{label}.max_ci95_high_nll_delta must be positive")
    return thresholds


def sampled_model_sha256(path: Path) -> tuple[int, str]:
    size = path.stat().st_size
    offsets = (0, max(0, size // 2 - 4 * 1024 * 1024),
               max(0, size - 8 * 1024 * 1024))
    digest = hashlib.sha256(f"{size}\n".encode())
    with path.open("rb") as stream:
        for offset in offsets:
            stream.seek(offset)
            digest.update(stream.read(8 * 1024 * 1024))
    return size, digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def roots() -> tuple[Path, Path]:
    repo = Path(run_git(Path.cwd(), "rev-parse", "--show-toplevel")).resolve()
    default_root = repo.parent / "research-results"
    root = Path(os.environ.get("DS4_RESEARCH_ROOT", default_root)).expanduser()
    if not root.is_absolute():
        raise GateError("DS4_RESEARCH_ROOT must be absolute")
    root = root.resolve()
    if root == repo or repo in root.parents:
        raise GateError("DS4_RESEARCH_ROOT must be outside the Git worktree")
    local = repo / "research-results"
    if local.exists() or local.is_symlink():
        raise GateError(f"forbidden worktree-local research path: {local}")
    if not root.is_dir():
        raise GateError(f"canonical research root is missing: {root}")
    return repo, root


def dossier_path(root: Path, candidate_id: str) -> Path:
    if not ID_RE.fullmatch(candidate_id):
        raise GateError("candidate id must use 1-96 letters, digits, '.', '_' or '-'")
    return root / "candidates" / candidate_id


def tp_layout_contract(value: dict, label: str) -> dict:
    """Return a canonical, fail-closed TP weight-layout contract.

    Baseline schema v1 and existing candidate dossiers describe DeepSeek's
    historical expert ownership with ``expert_split=128/128``.  Schema v2
    records use a structured discriminated layout so an FFN-intermediate
    shard cannot be mistaken for expert ownership.
    """
    if not isinstance(value, dict):
        raise GateError(f"{label} must be an object")
    has_legacy = "expert_split" in value
    has_structured = "tp_layout" in value
    if has_legacy == has_structured:
        raise GateError(
            f"{label} must contain exactly one of expert_split or tp_layout")
    if has_legacy:
        if value.get("expert_split") != "128/128":
            raise GateError(f"{label} has an unsupported legacy expert split")
        return {"expert_split": "128/128"}

    layout = value.get("tp_layout")
    if not isinstance(layout, dict):
        raise GateError(f"{label}.tp_layout must be an object")
    expected_fields = {
        "kind", "intermediate_size", "shards", "expert_count",
        "experts_used", "reduction",
    }
    if set(layout) != expected_fields:
        raise GateError(
            f"{label}.tp_layout fields must be exactly " +
            ", ".join(sorted(expected_fields)))
    if layout.get("kind") != "q4k-ffn-intermediate":
        raise GateError(f"{label}.tp_layout has an unsupported kind")
    tp_degree = value.get("tp_degree")
    intermediate_size = layout.get("intermediate_size")
    shards = layout.get("shards")
    expert_count = layout.get("expert_count")
    experts_used = layout.get("experts_used")
    if (type(tp_degree) is not int or tp_degree <= 0 or
            type(intermediate_size) is not int or intermediate_size <= 0 or
            not isinstance(shards, list) or len(shards) != tp_degree or
            any(type(shard) is not int or shard <= 0 for shard in shards) or
            sum(shards) != intermediate_size):
        raise GateError(
            f"{label}.tp_layout shards must be positive integers matching "
            "tp_degree and intermediate_size")
    if (type(expert_count) is not int or expert_count <= 0 or
            type(experts_used) is not int or experts_used <= 0 or
            experts_used > expert_count):
        raise GateError(f"{label}.tp_layout has invalid expert topology")
    reduction = layout.get("reduction")
    reduction_fields = {"op", "scope", "count", "width", "dtype"}
    if (not isinstance(reduction, dict) or
            set(reduction) != reduction_fields or
            reduction.get("op") != "sum" or
            reduction.get("scope") != "all-ranks" or
            reduction.get("dtype") != "f32" or
            type(reduction.get("count")) is not int or
            reduction["count"] <= 0 or
            type(reduction.get("width")) is not int or
            reduction["width"] <= 0):
        raise GateError(f"{label}.tp_layout has an invalid reduction contract")
    return {"tp_layout": json.loads(json.dumps(layout, sort_keys=True))}


def tp_layout_manifest_fields(contract: dict) -> dict[str, str]:
    if "expert_split" in contract:
        return {}
    layout = contract["tp_layout"]
    reduction = layout["reduction"]
    return {
        "tp_weight_layout": layout["kind"],
        "tp_intermediate_size": str(layout["intermediate_size"]),
        "tp_intermediate_shards": "/".join(str(item) for item in layout["shards"]),
        "tp_expert_count": str(layout["expert_count"]),
        "tp_experts_used": str(layout["experts_used"]),
        "tp_reduce_op": reduction["op"],
        "tp_reduce_scope": reduction["scope"],
        "tp_reduce_count": str(reduction["count"]),
        "tp_reduce_width": str(reduction["width"]),
        "tp_reduce_dtype": reduction["dtype"],
    }


def verify_tp_layout_manifest(manifest: dict[str, str], contract: dict,
                              label: str) -> None:
    for key, expected in tp_layout_manifest_fields(contract).items():
        if manifest.get(key) != expected:
            raise GateError(f"{label} differs in {key}")


def load_baseline(root: Path, baseline_id: str) -> tuple[Path, dict]:
    match = re.fullmatch(r"sha256:([0-9a-f]{64})", baseline_id)
    if not match:
        raise GateError("baseline_id must be sha256:<64 lowercase hex digits>")
    digest = match.group(1)
    path = root / "baselines" / "sha256" / f"{digest}.json"
    if not path.is_file():
        raise GateError(f"missing content-addressed baseline: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"invalid baseline JSON: {error}") from error
    if canonical_sha256(value) != digest:
        raise GateError(f"baseline content digest does not match its id: {path}")
    if value.get("schema_version") not in {1, 2} or value.get("kind") != "ds4-numerical-baseline":
        raise GateError(f"unsupported baseline schema: {path}")
    for section in ("key", "reference", "thresholds", "provenance"):
        if not isinstance(value.get(section), dict):
            raise GateError(f"baseline is missing {section}: {path}")
    key = value["key"]
    layout = tp_layout_contract(key, "baseline key")
    if value["schema_version"] == 1 and "expert_split" not in layout:
        raise GateError(f"baseline schema v1 requires legacy expert ownership: {path}")
    if value["schema_version"] == 2 and "tp_layout" not in layout:
        raise GateError(f"baseline schema v2 requires structured TP layout: {path}")
    fnv = str(value["reference"].get("fnv64", ""))
    if not re.fullmatch(r"[0-9a-f]{16}", fnv):
        raise GateError(f"baseline has invalid reference fingerprint: {path}")
    numerical = value["reference"].get("numerical")
    quality = value["reference"].get("quality")
    if (not isinstance(numerical, dict) or
            not isinstance(numerical.get("files"), list) or not numerical["files"]):
        raise GateError(f"baseline has no bound numerical reference: {path}")
    for item in numerical["files"]:
        if (not isinstance(item, dict) or not isinstance(item.get("name"), str) or
                not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))):
            raise GateError(f"baseline has invalid numerical reference hashes: {path}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(numerical.get("manifest_sha256", ""))):
        raise GateError(f"baseline has no bound numerical manifest: {path}")
    oracle = value["reference"].get("oracle_numerical")
    if oracle is not None:
        if (not isinstance(oracle, dict) or not isinstance(oracle.get("definition_id"), str) or
                not oracle["definition_id"] or not isinstance(oracle.get("files"), list) or
                not oracle["files"] or not re.fullmatch(
                    r"[0-9a-f]{64}", str(oracle.get("manifest_sha256", "")))):
            raise GateError(f"baseline has an invalid canonical oracle: {path}")
    generators = value.get("oracle_generators", [])
    if not isinstance(generators, list):
        raise GateError(f"baseline oracle_generators must be a list: {path}")
    for generator in generators:
        if (not isinstance(generator, dict) or not isinstance(generator.get("id"), str) or
                not generator["id"] or not isinstance(generator.get("entrypoint"), dict) or
                not isinstance(generator.get("runner"), dict) or
                not isinstance(generator.get("environment"), dict) or
                not re.fullmatch(r"[0-9a-f]{64}",
                                 str(generator.get("environment_id", ""))) or
                not isinstance(generator.get("closure"), list) or not generator["closure"] or
                not re.fullmatch(r"[0-9a-f]{64}",
                                 str(generator.get("closure_sha256", "")))):
            raise GateError(f"baseline has an invalid approved oracle generator: {path}")
        if generator.get("closure_sha256") != canonical_sha256(generator["closure"]):
            raise GateError(f"baseline oracle generator closure digest is invalid: {path}")
        if generator.get("environment_id") != canonical_sha256(generator["environment"]):
            raise GateError(f"baseline oracle generator environment digest is invalid: {path}")
    if (not isinstance(quality, dict) or
            not re.fullmatch(r"[0-9a-f]{64}", str(quality.get("sha256", ""))) or
            not re.fullmatch(r"[0-9a-f]{64}", str(quality.get("manifest_sha256", "")))):
        raise GateError(f"baseline has no bound quality reference: {path}")
    quality_anchor = value["reference"].get("quality_anchor", quality)
    if (not isinstance(quality_anchor, dict) or
            not re.fullmatch(r"[0-9a-f]{64}", str(quality_anchor.get("sha256", ""))) or
            not re.fullmatch(r"[0-9a-f]{64}",
                             str(quality_anchor.get("manifest_sha256", "")))):
        raise GateError(f"baseline has no bound quality anchor: {path}")
    numerical_thresholds = validate_numerical_thresholds(
        value["thresholds"].get("numerical"), "numerical")
    if len(numerical["files"]) < numerical_thresholds["min_teacher_steps"]:
        raise GateError(f"baseline numerical reference is shorter than its threshold: {path}")
    oracle_thresholds = value["thresholds"].get("oracle_numerical")
    if oracle_thresholds is not None:
        validate_numerical_thresholds(oracle_thresholds, "oracle_numerical")
    quality_thresholds = validate_quality_thresholds(
        value["thresholds"].get("quality"), "quality")
    performance_thresholds = value["thresholds"].get("performance")
    if value["schema_version"] == 2:
        validate_performance_thresholds(performance_thresholds, "performance")
        performance = value["reference"].get("performance")
        providers = key.get("rdma_providers")
        if (not isinstance(performance, dict) or
                not isinstance(providers, list) or set(performance) != set(providers)):
            raise GateError(f"production baseline has incomplete performance anchors: {path}")
        for provider, anchor in performance.items():
            if (not isinstance(anchor, dict) or
                    set(anchor) != {"runs", "geometric_mean_tps", "environment",
                                    "ds4_sha256", "ds4_bench_tp_sha256"} or
                    not isinstance(anchor["runs"], int) or anchor["runs"] < 3 or
                    not isinstance(anchor["geometric_mean_tps"], dict) or
                    set(anchor["geometric_mean_tps"]) != {"prefill", "decode"} or
                    any(not isinstance(number, (int, float)) or
                        not math.isfinite(number) or number <= 0
                        for number in anchor["geometric_mean_tps"].values()) or
                    not isinstance(anchor["environment"], dict) or
                    set(anchor["environment"]) != set(ENV_FIELDS) or
                    any(not isinstance(item, str)
                        for item in anchor["environment"].values()) or
                    not re.fullmatch(r"[0-9a-f]{64}",
                                     str(anchor["ds4_sha256"])) or
                    not re.fullmatch(r"[0-9a-f]{64}",
                                     str(anchor["ds4_bench_tp_sha256"]))):
                raise GateError(
                    f"production baseline has invalid {provider} performance anchor: {path}")
    if (value["schema_version"] == 2 and
            (numerical_thresholds.get("schema_version") != 2 or
             quality_thresholds.get("schema_version") != 2 or
             performance_thresholds is None or
             (oracle_thresholds is not None and
              oracle_thresholds.get("schema_version") != 2))):
        raise GateError(f"production schema v2 requires v2 threshold contracts: {path}")
    declared_scope = value.get("scope_sha256")
    if value["schema_version"] == 2 and declared_scope is None:
        raise GateError(f"production baseline has no scope digest: {path}")
    if declared_scope is not None and declared_scope != baseline_scope_sha256(value):
        raise GateError(f"baseline scope digest is invalid: {path}")
    if value["schema_version"] == 2:
        calibration = value["provenance"].get("calibration")
        hashes = calibration.get("comparator_sha256") if isinstance(calibration, dict) else None
        if (not isinstance(hashes, dict) or
                set(hashes) != set(verifier_sha256(Path(__file__).resolve().parents[1])) or
                any(not re.fullmatch(r"[0-9a-f]{64}", str(item))
                    for item in hashes.values())):
            raise GateError(f"production baseline has no complete verifier identity: {path}")
        verifier_identity = value["provenance"].get("verifier_sha256")
        expected_verifier_names = set(verifier_sha256(
            Path(__file__).resolve().parents[1]))
        if (not isinstance(verifier_identity, dict) or
                set(verifier_identity) != expected_verifier_names or
                any(not re.fullmatch(r"[0-9a-f]{64}", str(item))
                    for item in verifier_identity.values())):
            raise GateError(f"production baseline has no active verifier identity: {path}")
        formal_test = performance_thresholds["formal_test"]
        performance_method = value["provenance"].get("performance_method")
        expected_method_fields = {"formal_test"}
        if formal_test["kind"] == "repeated-student-v1":
            expected_method_fields |= {
                "design_calibration_sha256",
                "timing_noise_qualification_sha256", "scope_sha256",
                "provider", "pairs", "metrics",
            }
        if (not isinstance(performance_method, dict) or
                set(performance_method) != expected_method_fields or
                performance_method.get("formal_test") != formal_test["kind"]):
            raise GateError(f"production baseline has no bound performance method: {path}")
        if formal_test["kind"] == "repeated-student-v1":
            metrics = performance_method.get("metrics")
            if (performance_method.get("design_calibration_sha256") !=
                    formal_test["design_calibration_sha256"] or
                    performance_method.get("timing_noise_qualification_sha256") !=
                    formal_test["timing_noise_qualification_sha256"] or
                    performance_method.get("scope_sha256") != declared_scope or
                    performance_method.get("provider") !=
                    performance_thresholds["required_provider"] or
                    performance_method.get("pairs") != TIMING_NOISE_PAIR_COUNT or
                    not isinstance(metrics, dict) or
                    set(metrics) != {"prefill", "decode"} or
                    any(not isinstance(item, dict) or
                        item.get("pairs") != TIMING_NOISE_PAIR_COUNT or
                        not isinstance(item.get("sigma_log"), (int, float)) or
                        not math.isfinite(item["sigma_log"]) or
                        item["sigma_log"] <= 0
                        for item in metrics.values())):
                raise GateError(
                    f"production baseline has invalid timing-noise evidence: {path}")
    return path, value


def baseline_scope_payload(baseline: dict) -> dict:
    """Return the immutable scope shared by one evolving baseline lineage."""
    key = baseline.get("key")
    if not isinstance(key, dict):
        raise GateError("baseline scope has no key")
    workload = key.get("workload")
    if not isinstance(workload, dict):
        raise GateError("baseline scope has no workload")
    return {
        "model_sha256": key.get("model_sha256"),
        "model_size": key.get("model_size"),
        "quantization": key.get("quantization"),
        "architecture": key.get("architecture"),
        "tp_degree": key.get("tp_degree"),
        "tp_layout": tp_layout_contract(key, "baseline scope key"),
        "decode_mode": key.get("decode_mode"),
        "workload_id": key.get("workload_id"),
        "workload": workload,
        "toolchain_family": key.get("toolchain_family", "unspecified"),
    }


def baseline_scope_sha256(baseline: dict) -> str:
    return canonical_sha256(baseline_scope_payload(baseline))


def require_active_baseline(root: Path, baseline_id: str,
                            baseline: dict) -> None:
    """Resolve one journal-authorized immutable lineage head."""
    scope = baseline_scope_sha256(baseline)
    authorized: set[str] = set()
    for entry in journal_entries(root):
        event = entry.get("event")
        if (not isinstance(event, dict) or event.get("type") not in {
                "baseline-genesis", "baseline-amend", "candidate-promote"}):
            continue
        new_baseline_id = event.get("new_baseline_id")
        if new_baseline_id is None and event.get("type") == "candidate-promote":
            continue
        if (not isinstance(new_baseline_id, str) or
                not re.fullmatch(r"sha256:[0-9a-f]{64}", new_baseline_id) or
                new_baseline_id in authorized):
            raise GateError("governance journal has an invalid baseline authority event")
        authorized.add(new_baseline_id)
    records: dict[str, dict] = {}
    directory = root / "baselines" / "sha256"
    for path in sorted(directory.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise GateError(f"cannot resolve active baseline through {path}: {error}") from error
        if (not re.fullmatch(r"[0-9a-f]{64}", path.stem) or
                canonical_sha256(value) != path.stem or
                value.get("kind") != "ds4-numerical-baseline"):
            raise GateError(f"invalid record in content-addressed baseline store: {path}")
        try:
            record_scope = baseline_scope_sha256(value)
        except GateError:
            continue
        if record_scope == scope:
            record_id = f"sha256:{path.stem}"
            if record_id not in authorized:
                raise GateError(
                    f"baseline scope contains unjournaled record {record_id}")
            records[record_id] = value
    if baseline_id not in records:
        raise GateError("requested baseline is absent from its resolved scope")
    replaced = {
        str(value.get("provenance", {}).get("replaces"))
        for value in records.values()
        if isinstance(value.get("provenance"), dict) and
        str(value["provenance"].get("replaces", "")) in records
    }
    heads = sorted(set(records) - replaced)
    if len(heads) != 1:
        raise GateError(
            "baseline scope has multiple or no active heads; governance repair required")
    if heads[0] != baseline_id:
        raise GateError(
            f"candidate names superseded baseline {baseline_id}; active head is {heads[0]}")


def review_payload_sha256(payload: dict) -> str:
    """Hash a reviewable payload without its circular review attachments."""
    reviewable = json.loads(json.dumps(payload))
    evidence = reviewable.get("evidence")
    if isinstance(evidence, list):
        reviewable["evidence"] = [
            item for item in evidence
            if not isinstance(item, dict) or item.get("kind") not in REVIEW_KINDS
        ]
    return canonical_sha256(reviewable)


def used_review_hashes(root: Path) -> set[str]:
    """Return reviews already consumed by a completed governance action."""
    result: set[str] = set()
    for path in (root / "baselines" / "sha256").glob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        evidence = value.get("provenance", {}).get("evidence", [])
        if isinstance(evidence, list):
            result.update(str(item.get("sha256")) for item in evidence
                          if isinstance(item, dict) and
                          item.get("kind") in REVIEW_KINDS)
    for promoted in (root / "candidates").glob("*/PROMOTED.json"):
        candidate_path = promoted.parent / "candidate.json"
        try:
            value = json.loads(candidate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        evidence = value.get("evidence", [])
        if isinstance(evidence, list):
            result.update(str(item.get("sha256")) for item in evidence
                          if isinstance(item, dict) and
                          item.get("kind") in REVIEW_KINDS)
    return result


def verify_review_evidence(root: Path, evidence: object, label: str,
                           payload: dict, *, reject_reuse: bool = True) -> list[dict]:
    if not isinstance(evidence, list):
        raise GateError(f"{label} requires review evidence")
    review_kinds = set()
    reviews = [item for item in evidence
               if isinstance(item, dict) and item.get("kind") in REVIEW_KINDS]
    if len(reviews) != len(REVIEW_KINDS):
        raise GateError(f"{label} requires exactly one Fable and one Grok review")
    expected_payload = review_payload_sha256(payload)
    consumed = used_review_hashes(root) if reject_reuse else set()
    expected_calibration = payload.get("calibration_sha256")
    for item in reviews:
        if (not isinstance(item, dict) or item.get("kind") not in
                REVIEW_KINDS or
                not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))):
            raise GateError(f"{label} has invalid review evidence")
        path = Path(str(item.get("path", ""))).resolve()
        if path != root and root not in path.parents:
            raise GateError(f"{label} review escapes canonical research root")
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise GateError(f"{label} review hash mismatch")
        if reject_reuse and item["sha256"] in consumed:
            raise GateError(f"{label} review was already consumed by another action")
        metadata: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            normalized = key.strip().lower().replace("-", "_")
            if separator and normalized in {
                    "ds4_review_schema", "reviewer", "reviewed_payload_sha256",
                    "calibration_sha256", "verdict"}:
                if normalized in metadata:
                    raise GateError(f"{label} review repeats {key.strip()}")
                metadata[normalized] = value.strip()
        expected_reviewer = item["kind"].removesuffix("-review")
        if (metadata.get("ds4_review_schema") != "1" or
                metadata.get("reviewer", "").lower() != expected_reviewer or
                metadata.get("reviewed_payload_sha256") != expected_payload or
                metadata.get("verdict", "").upper() != "GO"):
            raise GateError(
                f"{label} {expected_reviewer} review is not an explicit GO "
                f"bound to payload {expected_payload}")
        if (expected_calibration is not None and
                metadata.get("calibration_sha256") != expected_calibration):
            raise GateError(
                f"{label} {expected_reviewer} review is not bound to calibration "
                f"{expected_calibration}")
        review_kinds.add(item["kind"])
    if review_kinds != REVIEW_KINDS:
        raise GateError(f"{label} requires both Fable and Grok reviews")
    return reviews


@governance_mutation(1)
def bootstrap_baseline(repo: Path, root: Path, genesis_path: Path) -> None:
    """Create the first immutable numerical baseline from recomputable evidence."""
    genesis_path = genesis_path.resolve()
    if genesis_path != root and root not in genesis_path.parents:
        raise GateError("baseline genesis must live in the canonical research root")
    try:
        genesis = json.loads(genesis_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"invalid baseline genesis: {error}") from error
    required = {
        "schema_version", "kind", "genesis_id", "rationale", "record",
        "artifacts", "calibration", "calibration_sha256", "verifier",
        "evidence",
    }
    if (not isinstance(genesis, dict) or set(genesis) != required or
            genesis.get("schema_version") != 2 or
            genesis.get("kind") != "ds4-baseline-genesis" or
            not ID_RE.fullmatch(str(genesis.get("genesis_id", ""))) or
            not isinstance(genesis.get("rationale"), str) or
            not genesis["rationale"].strip()):
        raise GateError("invalid baseline genesis identity or rationale")
    record = genesis.get("record")
    artifacts = genesis.get("artifacts")
    if not isinstance(record, dict) or not isinstance(artifacts, dict):
        raise GateError("baseline genesis requires record and artifacts objects")
    record = json.loads(json.dumps(record))
    if (record.get("schema_version") != 2 or
            record.get("kind") != "ds4-numerical-baseline"):
        raise GateError("baseline genesis record must use production schema v2")
    for section in ("key", "reference", "thresholds"):
        if not isinstance(record.get(section), dict):
            raise GateError(f"baseline genesis record is missing {section}")

    key = record["key"]
    reference = record["reference"]
    if ("oracle_numerical" in reference or "performance" in reference or
            "oracle_numerical" in record["thresholds"] or
            record.get("oracle_generators") not in (None, [])):
        raise GateError(
            "baseline genesis cannot preapprove canonical oracles or generators")
    workload = key.get("workload")
    if (not isinstance(workload, dict) or key.get("architecture") != "gfx1151" or
            key.get("tp_degree") != 2 or
            key.get("decode_mode") != "ordinary-greedy"):
        raise GateError("baseline genesis is not balanced ordinary gfx1151 TP=2")
    layout = tp_layout_contract(key, "baseline genesis key")
    if "tp_layout" not in layout:
        raise GateError("baseline genesis schema v2 requires structured TP layout")
    if (not re.fullmatch(r"[0-9a-f]{40}", str(key.get("source_commit", ""))) or
            not isinstance(key.get("toolchain_id"), str) or
            not key["toolchain_id"]):
        raise GateError("baseline genesis requires source and toolchain identity")
    fnv = str(reference.get("fnv64", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{16}", fnv):
        raise GateError("baseline genesis has an invalid reference fingerprint")
    providers = key.get("rdma_providers")
    if (not isinstance(providers, list) or not providers or
            len(set(providers)) != len(providers) or
            any(provider not in {"odinlink", "roce-v2"}
                for provider in providers)):
        raise GateError(
            "baseline genesis requires one or more distinct validated RDMA providers")

    model_path = Path(str(artifacts.get("model_path", "")))
    if not model_path.is_absolute() or not model_path.is_file():
        raise GateError("baseline genesis model path is invalid")
    model_size, model_sample = sampled_model_sha256(model_path)
    if (model_size != key.get("model_size") or
            model_sample != key.get("model_sample_sha256") or
            sha256(model_path) != key.get("model_sha256")):
        raise GateError("baseline genesis model identity mismatch")

    benchmark_items = artifacts.get("benchmarks")
    if not isinstance(benchmark_items, list):
        raise GateError("baseline genesis requires benchmark artifacts")
    provider_counts = {provider: 0 for provider in providers}
    provider_metrics = {
        provider: {"prefill": [], "decode": []} for provider in providers}
    provider_environments: dict[str, dict[str, str]] = {}
    provider_binaries: dict[str, str] = {}
    provider_bench_binaries: dict[str, str] = {}
    run_ids = set()
    binary_hashes = set()
    bench_binary_hashes = set()
    for item in benchmark_items:
        if (not isinstance(item, dict) or
                not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))) or
                not re.fullmatch(r"[0-9a-f]{64}",
                                 str(item.get("manifest_sha256", "")))):
            raise GateError("baseline genesis has invalid benchmark binding")
        csv_path = Path(str(item.get("path", ""))).resolve()
        if csv_path != root and root not in csv_path.parents:
            raise GateError("baseline genesis benchmark escapes canonical root")
        if not csv_path.is_file() or sha256(csv_path) != item["sha256"]:
            raise GateError("baseline genesis benchmark hash mismatch")
        row, manifest, manifest_path = read_benchmark(csv_path)
        verify_manifest_run_id_environment(
            manifest, "baseline genesis benchmark manifest")
        if sha256(manifest_path) != item["manifest_sha256"]:
            raise GateError("baseline genesis benchmark manifest hash mismatch")
        provider = manifest.get("rdma_profile")
        if provider not in provider_counts:
            raise GateError("baseline genesis benchmark used an undeclared provider")
        provider_counts[provider] += 1
        environment = normalized_manifest_environment(
            manifest, "baseline genesis benchmark")
        if provider in provider_environments and provider_environments[provider] != environment:
            raise GateError("baseline genesis provider runs used different environments")
        provider_environments[provider] = environment
        try:
            prefill_tps = float(row["prefill_tps"])
            decode_tps = float(row["gen_tps"])
        except (KeyError, ValueError) as error:
            raise GateError("baseline genesis benchmark has invalid throughput") from error
        if (not math.isfinite(prefill_tps) or prefill_tps <= 0 or
                not math.isfinite(decode_tps) or decode_tps <= 0):
            raise GateError("baseline genesis benchmark throughput must be positive")
        provider_metrics[provider]["prefill"].append(prefill_tps)
        provider_metrics[provider]["decode"].append(decode_tps)
        run_id = manifest.get("run_id")
        if not run_id or run_id in run_ids:
            raise GateError("baseline genesis benchmark run IDs are not distinct")
        run_ids.add(run_id)
        if (row["gen_token_fnv64"].lower() != fnv or
                manifest.get("source_dirty") != "0" or
                manifest.get("source_commit") != key.get("source_commit") or
                manifest.get("model_size") != str(model_size) or
                manifest.get("model_sample_sha256") != model_sample or
                manifest.get("toolchain_id") != key.get("toolchain_id") or
                manifest.get("dspark") != "0"):
            raise GateError("baseline genesis benchmark identity mismatch")
        bench_binary = verify_benchmark_producer(
            repo, manifest, key["source_commit"], "baseline genesis benchmark")
        verify_tp_layout_manifest(manifest, layout,
                                  "baseline genesis benchmark layout")
        rank_paths = {}
        for rank in ("coordinator", "worker"):
            for suffix in ("log", "status"):
                name = f"{rank}_{suffix}"
                path = _bound_timing_artifact(
                    item.get(name), root, f"baseline genesis {name}")
                expected = csv_path.parent / f"{rank}-{manifest.get('tag', '')}.{suffix}"
                if path != expected:
                    raise GateError(f"baseline genesis {name} is not adjacent to its CSV")
                rank_paths[name] = path
            status = read_manifest(rank_paths[f"{rank}_status"])
            if status.get("exit_code") != "0" or status.get("signal") != "0":
                raise GateError(f"baseline genesis {rank} did not exit cleanly")
        checked = subprocess.run([
            str(repo / "scripts/check-ds4-bench-result.sh"), str(csv_path),
            str(rank_paths["coordinator_log"]), str(rank_paths["worker_log"]),
            fnv, str(workload["generated_tokens"]), "0", provider,
            manifest.get("coordinator_rdma_device", ""),
            manifest.get("rdma_gid_index", ""),
            manifest.get("worker_rdma_device", ""), manifest.get("run_id", ""),
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if checked.returncode != 0:
            raise GateError("baseline genesis runtime validation failed: " +
                            (checked.stderr.strip() or checked.stdout.strip()))
        for field in workload_manifest_fields(workload):
            if manifest.get(field) != str(workload.get(field, "")):
                raise GateError(f"baseline genesis benchmark differs in {field}")
        local_binary = str(manifest.get("ds4_sha256", ""))
        peer_binary = str(manifest.get("peer_ds4_sha256", ""))
        if (not re.fullmatch(r"[0-9a-f]{64}", local_binary) or
                not re.fullmatch(r"[0-9a-f]{64}", peer_binary) or
                local_binary != peer_binary):
            raise GateError("baseline genesis binaries differed across ranks")
        if provider in provider_binaries and provider_binaries[provider] != local_binary:
            raise GateError("baseline genesis provider runs used different binaries")
        provider_binaries[provider] = local_binary
        binary_hashes.add(local_binary)
        if (provider in provider_bench_binaries and
                provider_bench_binaries[provider] != bench_binary):
            raise GateError(
                "baseline genesis provider runs used different benchmark binaries")
        provider_bench_binaries[provider] = bench_binary
        bench_binary_hashes.add(bench_binary)
    if any(count < 3 for count in provider_counts.values()):
        raise GateError("baseline genesis requires three runs per RDMA provider")
    if len(binary_hashes) != 1:
        raise GateError("baseline genesis benchmarks used different binaries")
    if len(bench_binary_hashes) != 1:
        raise GateError("baseline genesis benchmarks used different benchmark binaries")
    reference["performance"] = {
        provider: {
            "runs": provider_counts[provider],
            "geometric_mean_tps": {
                metric: math.exp(sum(math.log(item) for item in values) / len(values))
                for metric, values in provider_metrics[provider].items()
            },
            "environment": provider_environments[provider],
            "ds4_sha256": provider_binaries[provider],
            "ds4_bench_tp_sha256": provider_bench_binaries[provider],
        }
        for provider in providers
    }

    token_path = Path(str(artifacts.get("frozen_token_file", ""))).resolve()
    if token_path != root and root not in token_path.parents:
        raise GateError("baseline genesis frozen-token file escapes canonical root")
    if (not token_path.is_file() or
            sha256(token_path) != workload.get("frozen_token_sha256")):
        raise GateError("baseline genesis frozen-token identity mismatch")

    numerical_dir = Path(str(artifacts.get("numerical_dir", ""))).resolve()
    if numerical_dir != root and root not in numerical_dir.parents:
        raise GateError("baseline genesis numerical directory escapes canonical root")
    numerical = reference.get("numerical")
    if not isinstance(numerical, dict) or not isinstance(numerical.get("files"), list):
        raise GateError("baseline genesis numerical reference is missing")
    numerical_manifest = numerical_dir / "manifest"
    if (not numerical_manifest.is_file() or
            sha256(numerical_manifest) != numerical.get("manifest_sha256")):
        raise GateError("baseline genesis numerical manifest mismatch")
    numerical_values = read_manifest(numerical_manifest)
    if (numerical_values.get("model_size") != str(model_size) or
            numerical_values.get("model_sample_sha256") != model_sample or
            numerical_values.get("source_commit") != key.get("source_commit") or
            numerical_values.get("source_dirty") != "0" or
            numerical_values.get("toolchain_id") != key.get("toolchain_id") or
            numerical_values.get("prefix_tokens") != str(workload.get("frontier")) or
            numerical_values.get("frozen_token_sha256") !=
            workload.get("frozen_token_sha256") or
            numerical_values.get("dspark") != "0"):
        raise GateError("baseline genesis numerical manifest identity mismatch")
    bound_names = set()
    for item in numerical["files"]:
        if (not isinstance(item, dict) or not isinstance(item.get("name"), str) or
                Path(item["name"]).name != item["name"] or
                not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))):
            raise GateError("baseline genesis has invalid numerical file binding")
        path = (numerical_dir / item["name"]).resolve()
        if (path != root and root not in path.parents) or not path.is_file():
            raise GateError("baseline genesis numerical file escapes canonical root")
        if sha256(path) != item["sha256"]:
            raise GateError("baseline genesis numerical file hash mismatch")
        bound_names.add(item["name"])
    if (len(bound_names) != len(numerical["files"]) or
            numerical_values.get("file_count") != str(len(bound_names))):
        raise GateError("baseline genesis numerical file count mismatch")

    quality_path = Path(str(artifacts.get("quality_tsv", ""))).resolve()
    quality_manifest = Path(str(artifacts.get("quality_manifest", ""))).resolve()
    for path in (quality_path, quality_manifest):
        if path != root and root not in path.parents:
            raise GateError("baseline genesis quality artifact escapes canonical root")
    quality = reference.get("quality")
    if (not isinstance(quality, dict) or not quality_path.is_file() or
            not quality_manifest.is_file() or sha256(quality_path) != quality.get("sha256") or
            sha256(quality_manifest) != quality.get("manifest_sha256")):
        raise GateError("baseline genesis quality binding mismatch")
    quality_values = read_manifest(quality_manifest)
    if (quality_values.get("model_size") != str(model_size) or
            quality_values.get("model_sample_sha256") != model_sample or
            quality_values.get("source_commit") != key.get("source_commit") or
            quality_values.get("source_dirty") != "0" or
            quality_values.get("dspark") != "0"):
        raise GateError("baseline genesis quality manifest identity mismatch")
    try:
        with quality_path.open(newline="", encoding="utf-8") as stream:
            quality_rows = list(csv.DictReader(stream, delimiter="\t"))
        target_tokens = sum(int(row["target_tokens"]) for row in quality_rows)
    except (OSError, KeyError, ValueError) as error:
        raise GateError(f"baseline genesis quality table is invalid: {error}") from error

    numerical_thresholds = validate_numerical_thresholds(
        record["thresholds"].get("numerical"), "numerical")
    quality_thresholds = validate_quality_thresholds(
        record["thresholds"].get("quality"), "quality")
    performance_thresholds = validate_performance_thresholds(
        record["thresholds"].get("performance"), "performance")
    performance_method = verify_performance_method_evidence(
        repo, root, performance_thresholds,
        artifacts.get("timing_noise_qualification"), "baseline genesis",
        baseline=record)
    if (numerical_thresholds.get("schema_version") != 2 or
            quality_thresholds.get("schema_version") != 2):
        raise GateError("baseline genesis requires production threshold schema v2")
    if len(bound_names) < numerical_thresholds["min_teacher_steps"]:
        raise GateError("baseline genesis numerical coverage is too small")
    if (len(quality_rows) < quality_thresholds["min_cases"] or
            target_tokens < quality_thresholds["min_target_tokens"]):
        raise GateError("baseline genesis quality coverage is too small")
    # Reopen the raw numerical and quality artifacts with the same independent
    # comparators used by candidate promotion.  A list of valid hashes is not
    # enough: malformed or incomplete self-authored payloads must not become
    # the immutable truth anchor merely because their bytes were bound.
    with tempfile.TemporaryDirectory(prefix="baseline-genesis-", dir=root) as temporary:
        temporary_path = Path(temporary)
        numerical_threshold_path = temporary_path / "numerical-thresholds.json"
        quality_threshold_path = temporary_path / "quality-thresholds.json"
        atomic_json(numerical_threshold_path,
                    {"baseline_id": "GENESIS", **numerical_thresholds})
        atomic_json(quality_threshold_path,
                    {"baseline_id": "GENESIS", **quality_thresholds})
        numerical_result = rerun_json_tool(
            [sys.executable, str(Path(__file__).with_name("compare-teacher-logits.py")),
             str(numerical_dir), str(numerical_dir),
             "--thresholds", str(numerical_threshold_path)],
            "baseline genesis numerical self-check")
        quality_result = rerun_json_tool(
            [sys.executable, str(Path(__file__).with_name("compare-quality-scores.py")),
             str(quality_path), str(quality_path),
             "--thresholds", str(quality_threshold_path)],
            "baseline genesis quality self-check")
    if (numerical_result.get("passed") is not True or
            numerical_result.get("steps") != len(bound_names)):
        raise GateError("baseline genesis numerical self-check did not pass")
    quality_metrics = quality_result.get("metrics", {})
    if (quality_result.get("passed") is not True or
            quality_metrics.get("cases") != len(quality_rows) or
            quality_metrics.get("target_tokens") != target_tokens):
        raise GateError("baseline genesis quality self-check did not pass")
    record["scope_sha256"] = baseline_scope_sha256(record)
    calibration = evaluate_calibration(
        repo, root, genesis.get("calibration"), record["scope_sha256"], None,
        {"numerical": numerical_thresholds, "quality": quality_thresholds})
    if genesis.get("calibration_sha256") != calibration["calibration_sha256"]:
        raise GateError("baseline genesis calibration digest does not recompute")
    reviewed_verifier = reviewed_verifier_identity(
        repo, genesis["verifier"], "baseline genesis")
    reviews = verify_review_evidence(root, genesis.get("evidence"),
                                     "baseline genesis", genesis)
    if "quality_anchor" not in reference:
        reference["quality_anchor"] = quality
    if reference["quality_anchor"] != quality:
        raise GateError("baseline genesis quality anchor must be its own reference")
    record["oracle_generators"] = []
    record["provenance"] = {
        "genesis_id": genesis["genesis_id"],
        "lane_origin": "bootstrap",
        "rationale": genesis["rationale"],
        "genesis_json_sha256": sha256(genesis_path),
        "calibration": calibration,
        "verifier_source_commit": reviewed_verifier["source_commit"],
        "verifier_sha256": reviewed_verifier["sha256"],
        "performance_method": performance_method,
        "evidence": reviews,
        "benchmarks": benchmark_items,
    }
    digest = canonical_sha256(record)
    baseline_path = root / "baselines" / "sha256" / f"{digest}.json"
    if baseline_path.exists():
        raise GateError(f"refusing to overwrite existing baseline: {baseline_path}")
    atomic_json(baseline_path, record)
    try:
        load_baseline(root, f"sha256:{digest}")
        append_event(root, {
            "type": "baseline-genesis", "genesis_id": genesis["genesis_id"],
            "new_baseline_id": f"sha256:{digest}",
            "calibration_sha256": calibration["calibration_sha256"],
        })
    except Exception:
        baseline_path.unlink(missing_ok=True)
        raise
    print(f"sha256:{digest}")


@governance_mutation(1)
def init_candidate(repo: Path, root: Path, candidate_id: str, lane: str,
                   target_metrics: list[str],
                   switch_declarations: list[str], first_order: str) -> None:
    lane = lane.upper()
    if lane not in LANES:
        raise GateError("lane must be A, B, or C")
    dossier = dossier_path(root, candidate_id)
    if dossier.exists():
        raise GateError(f"candidate dossier already exists: {dossier}")
    status = run_git(repo, "status", "--porcelain=v1", "-uall")
    if status:
        raise GateError("candidate initialization requires a clean source checkout")
    source_commit = run_git(repo, "rev-parse", "HEAD")
    targets = target_metrics or ["prefill", "decode"]
    other_order = "BA" if first_order == "AB" else "AB"
    intent = validate_promotion_intent({
        "target_metrics": targets,
        "candidate_switches": parse_switch_declarations(switch_declarations),
        "headline_pair_order": [
            first_order if index % 2 == 0 else other_order for index in range(9)
        ],
        "invalidation_policy": "whole-pair-protocol-failure-only-v1",
    })
    for entry in journal_entries(root):
        event = entry.get("event")
        if (not isinstance(event, dict) or
                event.get("type") != "candidate-init" or
                event.get("source_commit") != source_commit):
            continue
        validate_promotion_intent(event.get("promotion_intent"))
        raise GateError(
            f"source commit already defines candidate {event.get('candidate_id')!r}; "
            "every new formal hypothesis requires a new commit")
    value = {
        "schema_version": 2,
        "candidate_id": candidate_id,
        "lane": lane,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": str(repo),
            "branch": run_git(repo, "branch", "--show-current") or "DETACHED",
            "commit": source_commit,
            "dirty": bool(status),
        },
        "baseline_id": "",
        "model": {},
        "toolchain": {},
        "workload": {
            "architecture": "gfx1151",
            "tp_degree": 2,
            "tp_layout": {
                "kind": "q4k-ffn-intermediate",
                "intermediate_size": 2048,
                "shards": [1024, 1024],
                "expert_count": 288,
                "experts_used": 8,
                "reduction": {
                    "op": "sum", "scope": "all-ranks", "count": 42,
                    "width": 4096, "dtype": "f32",
                },
            },
            "decode_mode": "ordinary-greedy",
            "workload_id": "ds4-bench-tp-2048x300",
        },
        "transport": {"providers": []},
        "target_definition": {"changed": False, "id": ""},
        "promotion_intent": intent,
        "dspark": False,
        "evidence": [],
        "notes": "",
    }
    candidate_path = dossier / "candidate.json"
    atomic_json(candidate_path, value)
    try:
        append_event(root, {
            "type": "candidate-init", "candidate_id": candidate_id,
            "lane": lane, "source_commit": value["source"]["commit"],
            "promotion_intent": intent,
            "promotion_intent_sha256": canonical_sha256(intent),
            "candidate_json_sha256": sha256(candidate_path),
        })
    except Exception:
        candidate_path.unlink(missing_ok=True)
        try:
            dossier.rmdir()
        except OSError:
            pass
        raise
    print(dossier / "candidate.json")


@governance_mutation(0)
def close_candidate(root: Path, candidate_id: str, reason: str) -> None:
    dossier, value = load_candidate(root, candidate_id)
    if (dossier / "PROMOTED.json").exists():
        raise GateError("a promoted candidate cannot be closed")
    if (dossier / "CLOSED.json").exists():
        raise GateError("candidate is already closed")
    if not reason.strip():
        raise GateError("closing a candidate requires a nonempty reason")
    record = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "candidate_json_sha256": sha256(dossier / "candidate.json"),
        "source_commit": value.get("source", {}).get("commit"),
        "reason": reason.strip(),
        "closed_utc": datetime.now(timezone.utc).isoformat(),
    }
    closed_path = dossier / "CLOSED.json"
    atomic_json(closed_path, record)
    try:
        append_event(root, {
            "type": "candidate-close", "candidate_id": candidate_id,
            "source_commit": value.get("source", {}).get("commit"),
            "promotion_intent_sha256": canonical_sha256(
                validate_promotion_intent(value.get("promotion_intent"))),
            "closed_json_sha256": sha256(closed_path),
        })
    except Exception:
        closed_path.unlink(missing_ok=True)
        raise
    print(dossier / "CLOSED.json")


def load_candidate(root: Path, candidate_id: str) -> tuple[Path, dict]:
    dossier = dossier_path(root, candidate_id)
    path = dossier / "candidate.json"
    if not path.is_file():
        raise GateError(f"missing candidate dossier: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"invalid candidate JSON: {error}") from error
    return dossier, value


def require_candidate_init(root: Path, candidate_id: str, value: dict) -> dict:
    matches = []
    for entry in journal_entries(root):
        event = entry.get("event")
        if (isinstance(event, dict) and event.get("type") == "candidate-init" and
                event.get("candidate_id") == candidate_id):
            matches.append((entry, event))
    if len(matches) != 1:
        raise GateError("candidate lacks one authoritative initialization event")
    entry, event = matches[0]
    intent = validate_promotion_intent(value.get("promotion_intent"))
    if (event.get("lane") != value.get("lane") or
            event.get("source_commit") != value.get("source", {}).get("commit") or
            event.get("promotion_intent") != intent or
            event.get("promotion_intent_sha256") != canonical_sha256(intent)):
        raise GateError("candidate differs from its initialized lane/source/intent")
    return {"event": event, "sequence": int(entry["sequence"])}


def headline_result_is_complete(path: Path) -> bool:
    """Return whether a CSV exposes a complete headline timing observation."""
    if not path.exists():
        return False
    if not path.is_file():
        raise GateError(f"headline result path is not a regular file: {path}")
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error) as error:
        raise GateError(f"cannot inspect headline result CSV {path}: {error}") from error
    if len(rows) != 1:
        return False
    row = rows[0]
    try:
        generated = int(row.get("gen_tokens", ""))
        prefill_tps = float(row.get("prefill_tps", ""))
        decode_tps = float(row.get("gen_tps", ""))
    except (TypeError, ValueError):
        return False
    return (generated >= 300 and math.isfinite(prefill_tps) and prefill_tps > 0 and
            math.isfinite(decode_tps) and decode_tps > 0)


def bind_headline_result_csv(root: Path, manifest_path: Path) -> dict:
    """Bind the adjacent result's bytes or absence immediately after a run."""
    path = manifest_path.with_suffix(".csv").resolve()
    if path != root and root not in path.parents:
        raise GateError("headline result CSV escapes DS4_RESEARCH_ROOT")
    if path.exists() and not path.is_file():
        raise GateError("headline result CSV is not a regular file")
    if path.is_file():
        return {"path": str(path), "exists": True, "sha256": sha256(path)}
    return {"path": str(path), "exists": False}


def verify_headline_result_csv(root: Path, manifest_path: Path,
                               value: object) -> bool:
    expected_path = manifest_path.with_suffix(".csv").resolve()
    if expected_path != root and root not in expected_path.parents:
        raise GateError("candidate headline result CSV escapes research root")
    if not isinstance(value, dict) or value.get("path") != str(expected_path):
        raise GateError("candidate headline result CSV binding is malformed")
    exists = value.get("exists")
    if exists is True:
        if (set(value) != {"path", "exists", "sha256"} or
                not re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", ""))) or
                not expected_path.is_file() or sha256(expected_path) != value["sha256"]):
            raise GateError("candidate headline result CSV is not immutable")
    elif exists is False:
        if set(value) != {"path", "exists"} or expected_path.exists():
            raise GateError("candidate headline result CSV absence changed")
    else:
        raise GateError("candidate headline result CSV binding is malformed")
    return headline_result_is_complete(expected_path)


def process_status(path: Path, label: str) -> tuple[int, int]:
    status = read_manifest(path)
    if set(status) != {"exit_code", "signal"}:
        raise GateError(f"{label} process status is malformed")
    try:
        exit_code = int(status["exit_code"])
        signal = int(status["signal"])
    except ValueError as error:
        raise GateError(f"{label} process status is malformed") from error
    if (exit_code < 0 or signal < 0 or
            (signal != 0 and exit_code != 128 + signal)):
        raise GateError(f"{label} process status is malformed")
    return exit_code, signal


def required_log_flag(log_path: Path, prefix: str, label: str) -> bool:
    values = [line.removeprefix(prefix) for line in
              log_path.read_text(encoding="utf-8").splitlines()
              if line.startswith(prefix)]
    if len(values) != 1 or values[0] not in {"0", "1"}:
        raise GateError(f"{label} lacks one valid {prefix} attestation")
    return values[0] == "1"


def producer_csv_completion(log_path: Path, label: str) -> bool:
    prefix = "ds4-bench: headline_csv_complete="
    values = [line.removeprefix(prefix) for line in
              log_path.read_text(encoding="utf-8").splitlines()
              if line.startswith(prefix)]
    if len(values) > 1 or any(value != "1" for value in values):
        raise GateError(f"{label} has malformed producer completion attestations")
    return bool(values)


def headline_result_observations(root: Path, paths: dict[str, Path],
                                 result_csv: object, label: str) -> dict:
    csv_complete = verify_headline_result_csv(
        root, paths["manifest"], result_csv)
    launcher_complete = required_log_flag(
        paths["coordinator_log"],
        "ds4-bench-launcher: headline_csv_complete=", label)
    producer_complete = producer_csv_completion(paths["coordinator_log"], label)
    worker_terminated = required_log_flag(
        paths["coordinator_log"],
        "ds4-bench-launcher: worker_terminated_by_launcher=", label)
    if csv_complete and (not launcher_complete or not producer_complete):
        raise GateError(
            f"{label} complete CSV lacks both producer and launcher attestations")
    return {
        "csv_complete": csv_complete,
        "launcher_complete": launcher_complete,
        "producer_complete": producer_complete,
        "worker_terminated_by_launcher": worker_terminated,
    }


HEADLINE_RESULT_ARTIFACTS = {
    "manifest", "coordinator_log", "coordinator_status", "worker_log",
    "worker_status",
}
SUCCESSFUL_HEADLINE_OBSERVATIONS = {
    "csv_complete": True,
    "launcher_complete": True,
    "producer_complete": True,
    "worker_terminated_by_launcher": False,
}


def headline_source_matches(root: Path, value: dict, arm: str,
                            source_commit: str | None) -> bool:
    """Bind each arm to the candidate or its independently loaded baseline."""
    if (arm not in {"control", "candidate"} or
            not re.fullmatch(r"[0-9a-f]{40}", str(source_commit))):
        return False
    if source_commit == value.get("source", {}).get("commit"):
        return True
    if arm != "control":
        return False
    baseline_id = str(value.get("baseline_id", ""))
    _, baseline = load_baseline(root, baseline_id)
    require_active_baseline(root, baseline_id, baseline)
    return source_commit == baseline["key"]["source_commit"]


def validate_headline_run_result(root: Path, candidate_id: str, value: dict,
                                 current: dict, event: dict,
                                 label: str) -> dict[str, Path]:
    required = {
        "type", "candidate_id", "baseline_id", "pair_index", "attempt",
        "pair_id", "order", "arm", "run_id", "promotion_intent_sha256",
        "artifacts", "result_csv", "observations",
    }
    artifacts = event.get("artifacts")
    arm = str(event.get("arm", ""))
    journaled_run = current["runs"].get(arm, {})
    if (set(event) != required or event.get("type") != "headline-run-result" or
            event.get("candidate_id") != candidate_id or
            event.get("baseline_id") != value.get("baseline_id") or
            event.get("pair_index") != current.get("pair_index") or
            event.get("attempt") != current.get("attempt") or
            event.get("pair_id") != current.get("pair_id") or
            event.get("order") != current.get("order") or
            event.get("run_id") != journaled_run.get("run_id") or
            event.get("promotion_intent_sha256") != canonical_sha256(
                validate_promotion_intent(value.get("promotion_intent"))) or
            not isinstance(artifacts, dict) or
            set(artifacts) != HEADLINE_RESULT_ARTIFACTS or
            not isinstance(event.get("observations"), dict) or
            "result" in journaled_run):
        raise GateError(f"{label} differs from its journaled headline run")

    paths = {}
    for name, artifact in artifacts.items():
        if (not isinstance(artifact, dict) or
                set(artifact) != {"path", "sha256"}):
            raise GateError(f"{label} artifacts are malformed")
        path = Path(str(artifact["path"])).resolve()
        if ((path != root and root not in path.parents) or
                not path.is_file() or sha256(path) != artifact["sha256"]):
            raise GateError(f"{label} artifacts are not immutable")
        paths[name] = path

    manifest = read_manifest(paths["manifest"])
    tag = manifest.get("tag", "")
    if (not tag or paths["manifest"].name != f"{tag}.manifest" or
            paths["coordinator_log"].name != f"coordinator-{tag}.log" or
            paths["coordinator_status"].name != f"coordinator-{tag}.status" or
            paths["worker_log"].name != f"worker-{tag}.log" or
            paths["worker_status"].name != f"worker-{tag}.status" or
            len({path.parent for path in paths.values()}) != 1 or
            manifest.get("pair_id") != current.get("pair_id") or
            manifest.get("pair_order") != current.get("order") or
            manifest.get("pair_arm") != arm or
            manifest.get("run_id") != journaled_run.get("run_id") or
            manifest.get("candidate_id") != candidate_id or
            manifest.get("candidate") != "1" or
            manifest.get("candidate_lane") != value.get("lane") or
            manifest.get("baseline_id") != value.get("baseline_id") or
            not headline_source_matches(root, value, arm, manifest.get("source_commit")) or
            manifest.get("source_dirty") != "0"):
        raise GateError(f"{label} artifacts do not identify the journaled run")
    verify_manifest_run_id_environment(manifest, f"{label} manifest")
    expected_log_line = f"ds4-tp: benchmark run_id={event['run_id']}"
    for name in ("coordinator_log", "worker_log"):
        if expected_log_line not in paths[name].read_text(encoding="utf-8"):
            raise GateError(f"{label} rank logs do not bind its run_id")
    process_status(paths["coordinator_status"], f"{label} coordinator")
    process_status(paths["worker_status"], f"{label} worker")
    observations = headline_result_observations(
        root, paths, event.get("result_csv"), label)
    if event["observations"] != observations:
        raise GateError(f"{label} observations differ from their bound artifacts")
    return paths


def headline_run_succeeded(run: dict) -> bool:
    result = run.get("result", {})
    artifacts = result.get("artifacts", {})
    if result.get("observations") != SUCCESSFUL_HEADLINE_OBSERVATIONS:
        return False
    try:
        coordinator = process_status(
            Path(artifacts["coordinator_status"]["path"]), "headline coordinator")
        worker = process_status(
            Path(artifacts["worker_status"]["path"]), "headline worker")
    except (KeyError, TypeError):
        return False
    return coordinator == (0, 0) and worker == (0, 0)


def verify_invalidation_process_failure(paths: dict[str, Path], label: str) -> None:
    coordinator_exit, coordinator_signal = process_status(
        paths["coordinator_status"], f"{label} coordinator")
    worker_exit, worker_signal = process_status(
        paths["worker_status"], f"{label} worker")
    worker_terminated = required_log_flag(
        paths["coordinator_log"],
        "ds4-bench-launcher: worker_terminated_by_launcher=", label)
    if coordinator_signal != 0:
        raise GateError(f"{label} requires a nonsignal coordinator failure")
    if worker_signal != 0 and not (
            worker_exit == 143 and worker_signal == 15 and worker_terminated and
            coordinator_exit != 0):
        raise GateError(
            f"{label} requires a nonsignal worker or an attested cleanup TERM")
    if coordinator_exit == 0 and worker_exit == 0:
        raise GateError(f"{label} lacks a process failure")
    if (producer_csv_completion(paths["coordinator_log"], label) or
            required_log_flag(
                paths["coordinator_log"],
                "ds4-bench-launcher: headline_csv_complete=", label)):
        raise GateError("an attested complete timing result cannot be invalidated")


def headline_pair_state(root: Path, candidate_id: str, value: dict) -> list[dict]:
    """Return prospective active pairs and their pre-launch run registrations."""
    initialization = require_candidate_init(root, candidate_id, value)
    intent = validate_promotion_intent(value.get("promotion_intent"))
    intent_sha256 = canonical_sha256(intent)
    active: list[dict] = []
    seen_ids: set[str] = set()
    seen_run_ids: set[str] = set()
    attempts: dict[int, int] = {}
    invalidation_count = 0
    for entry in journal_entries(root):
        if int(entry["sequence"]) <= initialization["sequence"]:
            continue
        event = entry.get("event")
        if not isinstance(event, dict) or event.get("candidate_id") != candidate_id:
            continue
        if event.get("type") == "headline-pair-begin":
            required = {
                "type", "candidate_id", "baseline_id", "pair_index", "attempt",
                "pair_id", "order", "promotion_intent_sha256",
            }
            pair_index = event.get("pair_index")
            attempt = event.get("attempt")
            pair_id = str(event.get("pair_id", ""))
            if (set(event) != required or
                    event.get("baseline_id") != value.get("baseline_id") or
                    event.get("promotion_intent_sha256") != intent_sha256 or
                    not isinstance(pair_index, int) or
                    pair_index != len(active) + 1 or pair_index > 9 or
                    not isinstance(attempt, int) or
                    attempt != attempts.get(pair_index, 0) + 1 or
                    attempt > MAX_HEADLINE_PAIR_ATTEMPTS or
                    not ID_RE.fullmatch(pair_id) or pair_id in seen_ids or
                    event.get("order") != intent["headline_pair_order"][pair_index - 1]):
                raise GateError("candidate has an invalid prospective pair-begin event")
            attempts[pair_index] = attempt
            seen_ids.add(pair_id)
            active.append({**event, "sequence": int(entry["sequence"]),
                           "recorded_utc": entry["recorded_utc"], "runs": {}})
        elif event.get("type") == "headline-run-begin":
            required = {
                "type", "candidate_id", "baseline_id", "pair_index", "attempt",
                "pair_id", "order", "arm", "run_id",
                "promotion_intent_sha256",
            }
            if not active or set(event) != required:
                raise GateError("candidate has an invalid headline-run-begin event")
            current = active[-1]
            arm = event.get("arm")
            run_id = str(event.get("run_id", ""))
            expected_arms = (("control", "candidate") if current["order"] == "AB"
                             else ("candidate", "control"))
            if (event.get("baseline_id") != value.get("baseline_id") or
                    event.get("pair_index") != current["pair_index"] or
                    event.get("attempt") != current["attempt"] or
                    event.get("pair_id") != current["pair_id"] or
                    event.get("order") != current["order"] or
                    event.get("promotion_intent_sha256") != intent_sha256 or
                    len(current["runs"]) >= 2 or
                    arm != expected_arms[len(current["runs"])] or
                    (bool(current["runs"]) and
                     "result" not in current["runs"][expected_arms[
                         len(current["runs"]) - 1]]) or
                    run_id in seen_run_ids or
                    benchmark_run_utc_ns(run_id) <= iso_utc_ns(current["recorded_utc"])):
                raise GateError("candidate headline run differs from its prospective pair")
            seen_run_ids.add(run_id)
            current["runs"][arm] = {
                "run_id": run_id, "sequence": int(entry["sequence"]),
                "recorded_utc": entry["recorded_utc"],
            }
        elif event.get("type") == "headline-run-result":
            if not active:
                raise GateError("candidate has a result without an active headline pair")
            current = active[-1]
            validate_headline_run_result(
                root, candidate_id, value, current, event,
                "candidate headline result")
            current["runs"][event["arm"]]["result"] = {
                **event, "sequence": int(entry["sequence"]),
                "recorded_utc": entry["recorded_utc"],
            }
        elif event.get("type") == "headline-pair-invalidate":
            invalidation_count += 1
            if invalidation_count > MAX_HEADLINE_INVALIDATIONS:
                raise GateError("candidate exceeds the maximum two headline invalidations")
            required = {
                "type", "candidate_id", "baseline_id", "pair_index", "attempt",
                "pair_id", "run_id", "reason", "failure_artifacts", "result_csv",
                "run_result_sha256",
            }
            if not active or set(event) != required:
                raise GateError("candidate has an invalid pair-invalidation event")
            current = active[-1]
            if len(current["runs"]) != 1:
                raise GateError(
                    "only a first-arm pre-result failure can invalidate a headline pair")
            artifacts = event.get("failure_artifacts")
            if (event.get("baseline_id") != value.get("baseline_id") or
                    event.get("pair_index") != current["pair_index"] or
                    event.get("attempt") != current["attempt"] or
                    event.get("pair_id") != current["pair_id"] or
                    not isinstance(event.get("reason"), str) or
                    not event["reason"].strip() or
                    not isinstance(artifacts, dict) or
                    set(artifacts) != HEADLINE_RESULT_ARTIFACTS):
                raise GateError("candidate pair invalidation differs from its active attempt")
            paths = {}
            for name, artifact in artifacts.items():
                if (not isinstance(artifact, dict) or
                        set(artifact) != {"path", "sha256"}):
                    raise GateError("candidate pair invalidation artifacts are malformed")
                path = Path(str(artifact["path"])).resolve()
                if ((path != root and root not in path.parents) or
                        not path.is_file() or sha256(path) != artifact["sha256"]):
                    raise GateError("candidate pair invalidation artifacts are not immutable")
                paths[name] = path
            manifest = read_manifest(paths["manifest"])
            tag = manifest.get("tag", "")
            arm = manifest.get("pair_arm")
            journaled_run = current["runs"].get(str(arm), {})
            run_result = journaled_run.get("result", {})
            if (not tag or paths["manifest"].name != f"{tag}.manifest" or
                    paths["coordinator_log"].name != f"coordinator-{tag}.log" or
                    paths["coordinator_status"].name != f"coordinator-{tag}.status" or
                    paths["worker_log"].name != f"worker-{tag}.log" or
                    paths["worker_status"].name != f"worker-{tag}.status" or
                    len({path.parent for path in paths.values()}) != 1 or
                    manifest.get("pair_id") != current["pair_id"] or
                    manifest.get("pair_order") != current["order"] or
                    arm not in {"control", "candidate"} or
                    event.get("run_id") != journaled_run.get("run_id") or
                    manifest.get("run_id") != journaled_run.get("run_id") or
                    manifest.get("candidate_id") != candidate_id or
                    manifest.get("candidate") != "1" or
                    manifest.get("candidate_lane") != value.get("lane") or
                    manifest.get("baseline_id") != value.get("baseline_id") or
                    not headline_source_matches(root, value, arm, manifest.get("source_commit")) or
                    manifest.get("source_dirty") != "0" or
                    artifacts != run_result.get("artifacts") or
                    event.get("result_csv") != run_result.get("result_csv") or
                    event.get("run_result_sha256") != canonical_sha256({
                        key: item for key, item in run_result.items()
                        if key not in {"sequence", "recorded_utc"}
                    })):
                raise GateError("candidate pair invalidation artifacts do not identify its run")
            verify_manifest_run_id_environment(
                manifest, "candidate pair invalidation manifest")
            expected_log_line = f"ds4-tp: benchmark run_id={event['run_id']}"
            for name in ("coordinator_log", "worker_log"):
                if expected_log_line not in paths[name].read_text(encoding="utf-8"):
                    raise GateError(
                        "candidate pair invalidation rank logs do not bind its run_id")
            if verify_headline_result_csv(
                    root, paths["manifest"], event.get("result_csv")):
                raise GateError("a complete timing result cannot be invalidated")
            verify_invalidation_process_failure(
                paths, "candidate pair invalidation")
            active.pop()
    return active


def headline_invalidation_count(root: Path, candidate_id: str) -> int:
    return sum(
        1 for entry in journal_entries(root)
        if isinstance(entry.get("event"), dict) and
        entry["event"].get("type") == "headline-pair-invalidate" and
        entry["event"].get("candidate_id") == candidate_id)


def prior_formal_candidates(root: Path, candidate_id: str, value: dict,
                            candidate_binary_sha256: str) -> list[dict]:
    """Disclose all earlier same-lane candidates and identify switch matches."""
    initialization = require_candidate_init(root, candidate_id, value)
    switches = validate_promotion_intent(
        value.get("promotion_intent"))["candidate_switches"]
    entries = journal_entries(root)
    terminal_events: dict[str, tuple[str, dict]] = {}
    for entry in entries:
        event = entry.get("event")
        if not isinstance(event, dict):
            continue
        previous_id = str(event.get("candidate_id", ""))
        if event.get("type") == "candidate-close":
            terminal_events[previous_id] = ("closed", event)
        elif event.get("type") == "candidate-promote":
            terminal_events[previous_id] = ("promoted", event)

    result = []
    for entry in entries:
        if int(entry["sequence"]) >= initialization["sequence"]:
            continue
        event = entry.get("event")
        if (not isinstance(event, dict) or event.get("type") != "candidate-init" or
                event.get("lane") != value.get("lane")):
            continue
        previous_intent = validate_promotion_intent(event.get("promotion_intent"))
        previous_id = str(event.get("candidate_id", ""))
        status, terminal = terminal_events.get(previous_id, ("open", {}))
        previous_binary = terminal.get("candidate_binary_sha256")
        if previous_binary is not None and not re.fullmatch(
                r"[0-9a-f]{64}", str(previous_binary)):
            raise GateError("prior formal candidate has an invalid binary identity")
        result.append({
            "candidate_id": previous_id,
            "source_commit": event.get("source_commit"),
            "status": status,
            "candidate_binary_sha256": previous_binary,
            "switch_contrast_match": previous_intent["candidate_switches"] == switches,
            "same_candidate_binary": (
                None if previous_binary is None
                else previous_binary == candidate_binary_sha256),
        })
    return result


@governance_mutation(1)
def begin_headline_pair(repo: Path, root: Path, candidate_id: str) -> None:
    dossier, value = load_candidate(root, candidate_id)
    if not candidate_is_open(dossier):
        raise GateError("only an open candidate can begin a headline pair")
    initialization = require_candidate_init(root, candidate_id, value)
    _, baseline = load_baseline(root, str(value.get("baseline_id", "")))
    require_active_baseline(root, value["baseline_id"], baseline)
    if initialization["sequence"] <= baseline_authority_sequence(
            root, value["baseline_id"]):
        raise GateError("candidate must be initialized after its active baseline")
    active = headline_pair_state(root, candidate_id, value)
    if active and (set(active[-1]["runs"]) != {"control", "candidate"} or
                   any("result" not in run for run in active[-1]["runs"].values())):
        raise GateError(
            "both journaled arms must bind their results before beginning the next pair")
    if active and not all(headline_run_succeeded(run)
                          for run in active[-1]["runs"].values()):
        raise GateError(
            "a failed completed pair closes this candidate; repair in a new commit")
    if len(active) >= 9:
        raise GateError("candidate already began the maximum nine headline pairs")
    pair_index = len(active) + 1
    attempts = sum(
        1 for entry in journal_entries(root)
        if isinstance(entry.get("event"), dict) and
        entry["event"].get("type") == "headline-pair-begin" and
        entry["event"].get("candidate_id") == candidate_id and
        entry["event"].get("pair_index") == pair_index) + 1
    if attempts > MAX_HEADLINE_PAIR_ATTEMPTS:
        raise GateError("headline pair index exhausted its two permitted attempts")
    pair_id = f"h{pair_index}-a{attempts}-{secrets.token_hex(12)}"
    intent = validate_promotion_intent(value["promotion_intent"])
    event = {
        "type": "headline-pair-begin", "candidate_id": candidate_id,
        "baseline_id": value["baseline_id"], "pair_index": pair_index,
        "attempt": attempts, "pair_id": pair_id,
        "order": intent["headline_pair_order"][pair_index - 1],
        "promotion_intent_sha256": canonical_sha256(intent),
    }
    entry = append_event(root, event)
    print(json.dumps({"pair_id": pair_id, "pair_index": pair_index,
                      "order": event["order"],
                      "recorded_utc": entry["recorded_utc"]}, sort_keys=True))


@governance_mutation(0)
def record_headline_run(root: Path, candidate_id: str, pair_id: str,
                        order: str, arm: str, run_id: str) -> None:
    dossier, value = load_candidate(root, candidate_id)
    if not candidate_is_open(dossier):
        raise GateError("only an open candidate can record a headline run")
    active = headline_pair_state(root, candidate_id, value)
    if not active or active[-1]["pair_id"] != pair_id:
        raise GateError("headline run must belong to the latest active pair")
    current = active[-1]
    if order != current["order"]:
        raise GateError("headline run order differs from its prospective pair")
    expected_arms = (("control", "candidate") if current["order"] == "AB"
                     else ("candidate", "control"))
    if arm not in {"control", "candidate"}:
        raise GateError("headline run arm must be control or candidate")
    if len(current["runs"]) >= 2 or arm != expected_arms[len(current["runs"])]:
        raise GateError("headline run arm is duplicate or contradicts AB/BA order")
    if any(
            isinstance(entry.get("event"), dict) and
            entry["event"].get("type") == "headline-run-begin" and
            entry["event"].get("run_id") == run_id
            for entry in journal_entries(root)):
        raise GateError("headline run_id was already journaled")
    if current["runs"]:
        previous = current["runs"][expected_arms[len(current["runs"]) - 1]]
        if "result" not in previous:
            raise GateError("the preceding headline arm must bind its result first")
        if not headline_run_succeeded(previous):
            raise GateError(
                "a failed first arm must be invalidated before another arm starts")
    if benchmark_run_utc_ns(run_id) <= iso_utc_ns(current["recorded_utc"]):
        raise GateError("headline run predates its prospective pair-begin")
    intent = validate_promotion_intent(value["promotion_intent"])
    event = {
        "type": "headline-run-begin", "candidate_id": candidate_id,
        "baseline_id": value["baseline_id"],
        "pair_index": current["pair_index"], "attempt": current["attempt"],
        "pair_id": pair_id, "order": current["order"], "arm": arm,
        "run_id": run_id,
        "promotion_intent_sha256": canonical_sha256(intent),
    }
    entry = append_event(root, event)
    print(json.dumps({"pair_id": pair_id, "pair_index": current["pair_index"],
                      "arm": arm, "run_id": run_id,
                      "recorded_utc": entry["recorded_utc"]}, sort_keys=True))


@governance_mutation(0)
def record_headline_result(root: Path, candidate_id: str, pair_id: str,
                           run_id: str, manifest_path: Path,
                           coordinator_log_path: Path,
                           coordinator_status_path: Path,
                           worker_log_path: Path,
                           worker_status_path: Path) -> None:
    dossier, value = load_candidate(root, candidate_id)
    if not candidate_is_open(dossier):
        raise GateError("only an open candidate can record a headline result")
    active = headline_pair_state(root, candidate_id, value)
    if not active or active[-1]["pair_id"] != pair_id:
        raise GateError("headline result must belong to the latest active pair")
    current = active[-1]
    matches = [
        (arm, run) for arm, run in current["runs"].items()
        if run.get("run_id") == run_id
    ]
    if len(matches) != 1 or "result" in matches[0][1]:
        raise GateError("headline result does not identify one unbound journaled run")
    arm, _ = matches[0]
    raw_paths = {
        "manifest": manifest_path, "coordinator_log": coordinator_log_path,
        "coordinator_status": coordinator_status_path,
        "worker_log": worker_log_path, "worker_status": worker_status_path,
    }
    paths = {name: path.expanduser().resolve() for name, path in raw_paths.items()}
    for name, path in paths.items():
        if path != root and root not in path.parents:
            raise GateError(f"headline result {name} escapes DS4_RESEARCH_ROOT")
        if not path.is_file():
            raise GateError(f"headline result {name} is missing")
    result_csv = bind_headline_result_csv(root, paths["manifest"])
    observations = headline_result_observations(
        root, paths, result_csv, "headline result")
    intent = validate_promotion_intent(value["promotion_intent"])
    event = {
        "type": "headline-run-result", "candidate_id": candidate_id,
        "baseline_id": value["baseline_id"],
        "pair_index": current["pair_index"], "attempt": current["attempt"],
        "pair_id": pair_id, "order": current["order"], "arm": arm,
        "run_id": run_id,
        "promotion_intent_sha256": canonical_sha256(intent),
        "result_csv": result_csv, "observations": observations,
        "artifacts": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }
    validate_headline_run_result(
        root, candidate_id, value, current, event, "headline result")
    entry = append_event(root, event)
    print(json.dumps({
        "pair_id": pair_id, "pair_index": current["pair_index"],
        "arm": arm, "run_id": run_id,
        "observations": observations,
        "recorded_utc": entry["recorded_utc"],
    }, sort_keys=True))


@governance_mutation(0)
def invalidate_headline_pair(root: Path, candidate_id: str, pair_id: str,
                             run_id: str,
                             manifest_path: Path, coordinator_log_path: Path,
                             coordinator_status_path: Path, worker_log_path: Path,
                             worker_status_path: Path,
                             reason: str) -> None:
    dossier, value = load_candidate(root, candidate_id)
    if not candidate_is_open(dossier):
        raise GateError("only an open candidate can invalidate a headline pair")
    active = headline_pair_state(root, candidate_id, value)
    if not active or active[-1]["pair_id"] != pair_id:
        raise GateError("only the latest active headline pair can be invalidated")
    if len(active[-1]["runs"]) != 1:
        raise GateError(
            "only a first-arm pre-result failure can invalidate a headline pair")
    if "result" not in next(iter(active[-1]["runs"].values())):
        raise GateError("headline failure artifacts must be bound before invalidation")
    if headline_invalidation_count(root, candidate_id) >= MAX_HEADLINE_INVALIDATIONS:
        raise GateError("candidate exhausted its two permitted headline invalidations")
    raw_paths = {
        "manifest": manifest_path, "coordinator_log": coordinator_log_path,
        "coordinator_status": coordinator_status_path,
        "worker_log": worker_log_path, "worker_status": worker_status_path,
    }
    paths = {name: path.expanduser().resolve() for name, path in raw_paths.items()}
    for name, path in paths.items():
        if path != root and root not in path.parents:
            raise GateError(f"pair invalidation {name} escapes DS4_RESEARCH_ROOT")
        if not path.is_file():
            raise GateError(f"pair invalidation {name} is missing")
    manifest = read_manifest(paths["manifest"])
    tag = manifest.get("tag", "")
    current = active[-1]
    arm = manifest.get("pair_arm")
    journaled_run = current["runs"].get(str(arm), {})
    if (not tag or paths["manifest"].name != f"{tag}.manifest" or
            paths["coordinator_log"].name != f"coordinator-{tag}.log" or
            paths["coordinator_status"].name != f"coordinator-{tag}.status" or
            paths["worker_log"].name != f"worker-{tag}.log" or
            paths["worker_status"].name != f"worker-{tag}.status" or
            len({path.parent for path in paths.values()}) != 1 or
            manifest.get("pair_id") != pair_id or
            manifest.get("pair_order") != current["order"] or
            arm not in {"control", "candidate"} or
            run_id != journaled_run.get("run_id") or
            manifest.get("run_id") != run_id or
            manifest.get("candidate") != "1" or
            manifest.get("candidate_id") != candidate_id or
            manifest.get("candidate_lane") != value["lane"] or
            manifest.get("baseline_id") != value["baseline_id"] or
            not headline_source_matches(root, value, arm, manifest.get("source_commit")) or
            manifest.get("source_dirty") != "0"):
        raise GateError("pair invalidation artifacts do not identify the active attempt")
    verify_manifest_run_id_environment(manifest, "pair invalidation manifest")
    expected_log_line = f"ds4-tp: benchmark run_id={run_id}"
    for name in ("coordinator_log", "worker_log"):
        if expected_log_line not in paths[name].read_text(encoding="utf-8"):
            raise GateError("pair invalidation rank logs do not bind its run_id")
    if benchmark_run_utc_ns(manifest.get("run_id", "")) <= iso_utc_ns(
            current["recorded_utc"]):
        raise GateError("pair invalidation run predates its prospective pair-begin")
    run_result = journaled_run.get("result", {})
    if any(
            run_result.get("artifacts", {}).get(name, {}).get("path") != str(path)
            for name, path in paths.items()):
        raise GateError("pair invalidation paths differ from the bound headline result")
    result_csv = run_result.get("result_csv")
    if verify_headline_result_csv(root, paths["manifest"], result_csv):
        raise GateError("a complete timing result cannot be invalidated")
    verify_invalidation_process_failure(paths, "pair invalidation")
    if not reason.strip():
        raise GateError("pair invalidation requires a nonempty reason")
    append_event(root, {
        "type": "headline-pair-invalidate", "candidate_id": candidate_id,
        "baseline_id": value["baseline_id"],
        "pair_index": current["pair_index"], "attempt": current["attempt"],
        "pair_id": pair_id, "run_id": run_id, "reason": reason.strip(),
        "result_csv": result_csv,
        "failure_artifacts": run_result["artifacts"],
        "run_result_sha256": canonical_sha256({
            key: item for key, item in run_result.items()
            if key not in {"sequence", "recorded_utc"}
        }),
    })
    print(json.dumps({"invalidated_pair_id": pair_id,
                      "pair_index": current["pair_index"]}, sort_keys=True))


def baseline_authority_sequence(root: Path, baseline_id: str) -> int:
    matches = []
    for entry in journal_entries(root):
        event = entry.get("event")
        if (isinstance(event, dict) and event.get("type") in {
                "baseline-genesis", "baseline-amend", "candidate-promote"} and
                event.get("new_baseline_id") == baseline_id):
            matches.append(int(entry["sequence"]))
    if len(matches) != 1:
        raise GateError("active baseline lacks one authoritative journal event")
    return matches[0]


def reject_open_scope_candidates(root: Path, scope_sha256: str) -> None:
    initialized = event_sequences(root, "candidate-init", "candidate_id")
    finished = (event_sequences(root, "candidate-close", "candidate_id").keys() |
                event_sequences(root, "candidate-promote", "candidate_id").keys())
    for identity in initialized.keys() - finished:
        if (not ID_RE.fullmatch(identity) or
                not (root / "candidates" / identity / "candidate.json").is_file()):
            raise GateError(
                "performance policy amendment cannot audit initialized open "
                f"candidate {identity!r}: missing dossier or invalid identity")
    for dossier, candidate in candidate_values(root):
        identities = {dossier.name, str(candidate.get("candidate_id", ""))}
        registered = identities & initialized.keys()
        if registered and candidate.get("candidate_id") != dossier.name:
            raise GateError(
                "performance policy amendment cannot audit conflicting "
                f"candidate identities in dossier {dossier.name!r}")
        if ((registered and registered <= finished) or
                (not registered and not candidate_is_open(dossier))):
            continue
        scope = candidate_scope(root, candidate)
        unassigned = candidate.get("baseline_id") in {None, ""}
        # Pre-policy research notes have neither a frozen intent nor an
        # initialization event. They cannot observe a formal timing sequence
        # and must not force unrelated, durable tracks to be closed.
        if unassigned and "promotion_intent" not in candidate and not registered:
            continue
        if (scope == scope_sha256 or unassigned or
                (scope is None and (registered or "promotion_intent" in candidate))):
            raise GateError(
                f"performance policy amendment cannot observe open candidate "
                f"{candidate.get('candidate_id')!r}; close it and initialize a "
                "fresh candidate after the amendment")


def verify_verifier_identity(repo: Path, baseline: dict) -> None:
    expected = baseline.get("provenance", {}).get("verifier_sha256")
    actual = verifier_sha256(repo)
    if expected != actual:
        raise GateError(
            "active baseline verifier identity differs; amend governance before promotion")


def reviewed_verifier_identity(repo: Path, value: object, label: str) -> dict:
    if (not isinstance(value, dict) or
            set(value) != {"source_commit", "sha256"} or
            not re.fullmatch(r"[0-9a-f]{40}", str(value.get("source_commit", ""))) or
            not isinstance(value.get("sha256"), dict)):
        raise GateError(f"{label} has no reviewed verifier identity")
    if run_git(repo, "status", "--porcelain=v1", "-uall"):
        raise GateError(f"{label} requires a clean reviewed verifier checkout")
    expected = verifier_sha256(repo)
    if (value["source_commit"] != run_git(repo, "rev-parse", "HEAD") or
            value["sha256"] != expected):
        raise GateError(f"{label} reviewed verifier identity differs from this checkout")
    return {"source_commit": value["source_commit"], "sha256": expected}


def read_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise GateError(f"cannot read benchmark manifest {path}: {error}") from error
    for line in lines:
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise GateError(f"malformed or duplicate manifest key in {path}: {key!r}")
        values[key] = value
    return values


def iso_utc_ns(value: str) -> int:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise GateError("prospective pair event has an invalid UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise GateError("prospective pair event timestamp is not UTC")
    utc = parsed.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = utc - epoch
    return ((delta.days * 86400 + delta.seconds) * 1_000_000_000 +
            delta.microseconds * 1000)


def benchmark_run_utc_ns(run_id: str) -> int:
    match = re.fullmatch(
        r"(\d{8}T\d{6})\.(\d{1,9})Z-[A-Za-z0-9][A-Za-z0-9._-]*", run_id)
    if not match:
        raise GateError("benchmark run_id has no parseable UTC timestamp")
    try:
        base = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(
            tzinfo=timezone.utc)
    except ValueError as error:
        raise GateError("benchmark run_id has an invalid UTC timestamp") from error
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = base - epoch
    nanoseconds = int(match.group(2).ljust(9, "0"))
    return (delta.days * 86400 + delta.seconds) * 1_000_000_000 + nanoseconds


def read_benchmark(path: Path) -> tuple[dict[str, str], dict[str, str], Path]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as error:
        raise GateError(f"cannot read benchmark CSV {path}: {error}") from error
    if len(rows) != 1:
        raise GateError(f"candidate benchmark must contain one row: {path}")
    row = rows[0]
    fnv = str(row.get("gen_token_fnv64", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{16}", fnv):
        raise GateError(f"candidate benchmark has invalid fingerprint: {path}")
    try:
        if int(row.get("gen_tokens", "0")) < 300:
            raise GateError(f"candidate benchmark generated fewer than 300 tokens: {path}")
    except ValueError as error:
        raise GateError(f"candidate benchmark has invalid gen_tokens: {path}") from error
    manifest_path = path.with_suffix(".manifest")
    manifest = read_manifest(manifest_path)
    return row, manifest, manifest_path


def threshold_file(value: dict, baseline_id: str, section: str, directory: Path) -> Path:
    thresholds = value["thresholds"].get(section)
    if not isinstance(thresholds, dict):
        raise GateError(f"baseline has no {section} thresholds")
    payload = {"baseline_id": baseline_id, **thresholds}
    path = directory / f"{section}-thresholds.json"
    atomic_json(path, payload)
    return path


def rerun_json_tool(command: list[str], label: str) -> dict:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise GateError(f"{label} failed independent recomputation: {detail}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise GateError(f"{label} returned invalid JSON") from error


def verify_numerical_evidence(repo: Path, root: Path, summary_path: Path,
                              baseline_id: str, baseline: dict,
                              candidate_model: dict, lane: str,
                              source: dict, toolchain: dict,
                              target_definition: dict,
                              candidate_switches: dict,
                              candidate_binary_sha256: str) -> dict:
    try:
        recorded = json.loads(summary_path.read_text(encoding="utf-8"))
        sources = recorded["sources"]
        reference_dir = Path(sources["reference_dir"]).resolve()
        candidate_dir = Path(sources["candidate_dir"]).resolve()
        reference_manifest = Path(sources["reference_manifest"]).resolve()
        candidate_manifest = Path(sources["candidate_manifest"]).resolve()
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise GateError(f"invalid numerical-envelope summary: {error}") from error
    for path in (reference_dir, candidate_dir, reference_manifest, candidate_manifest):
        if path != root and root not in path.parents:
            raise GateError(f"numerical source escapes canonical research root: {path}")
    if reference_dir == candidate_dir:
        raise GateError("numerical reference and candidate directories must differ")
    definition_changed = lane == "C" and target_definition.get("changed") is True
    threshold_section = "oracle_numerical" if lane == "C" else "numerical"
    with tempfile.TemporaryDirectory() as raw:
        threshold = threshold_file(baseline, baseline_id, threshold_section, Path(raw))
        command = [
            sys.executable, str(repo / "scripts" / "compare-teacher-logits.py"),
            str(reference_dir), str(candidate_dir), "--thresholds", str(threshold),
        ]
        allow_quality_difference = baseline["thresholds"][threshold_section].get(
            "allow_quality_difference", False)
        if recorded.get("allow_quality_difference") is not allow_quality_difference:
            raise GateError("numerical comparison mode differs from the baseline contract")
        if allow_quality_difference:
            command.append("--allow-quality-difference")
        recomputed = rerun_json_tool(command, "numerical envelope")
    ignored = {"thresholds_sha256"}
    if {key: value for key, value in recorded.items() if key not in ignored} != {
            key: value for key, value in recomputed.items() if key not in ignored}:
        raise GateError("numerical-envelope summary does not match its raw logits")
    reference_hashes = [
        {"name": item.get("name"), "sha256": item.get("reference_sha256")}
        for item in recorded.get("sources", {}).get("pairs", [])
    ]
    if definition_changed:
        oracle_identity = read_manifest(reference_manifest)
        approved_generators = {
            item["id"]: item for item in baseline.get("oracle_generators", [])
        }
        generator_id = oracle_identity.get("generator_id")
        approved = approved_generators.get(generator_id)
        if not isinstance(approved, dict):
            raise GateError("Lane C oracle generator is not approved by the predecessor")
        generator_path, runner_path = verify_generator_closure(repo, root, approved)
        file_list_hash = hashlib.sha256("".join(
            f"{item['name']} {item['sha256']}\n" for item in reference_hashes
        ).encode()).hexdigest()
        token_file = Path(oracle_identity.get("token_file", "")).resolve()
        if token_file != root and root not in token_file.parents:
            raise GateError("Lane C oracle token file escapes the research root")
        try:
            token_sha256 = sha256(token_file)
        except OSError as error:
            raise GateError(f"cannot hash Lane C oracle token file: {error}") from error
        if (oracle_identity.get("oracle") != "1" or
                oracle_identity.get("definition_id") != target_definition.get("id") or
                oracle_identity.get("model") != candidate_model["path"] or
                oracle_identity.get("model_size") != str(candidate_model["size"]) or
                oracle_identity.get("model_sample_sha256") != candidate_model["sample_sha256"] or
                oracle_identity.get("generator_closure_sha256") !=
                    approved["closure_sha256"] or
                oracle_identity.get("generator_environment_id") !=
                    approved["environment_id"] or
                not oracle_identity.get("toolchain_id") or
                oracle_identity.get("prefix_tokens") != str(
                    baseline["key"]["workload"]["frontier"]) or
                oracle_identity.get("file_count") != str(len(reference_hashes)) or
                oracle_identity.get("files_sha256") != file_list_hash):
            raise GateError("new Lane C oracle manifest is not a bound canonical producer")
        if (token_sha256 != oracle_identity.get("token_sha256") or
                token_sha256 != baseline["key"]["workload"].get("frozen_token_sha256")):
            raise GateError("Lane C oracle token sequence differs from the frozen workload")
        with tempfile.TemporaryDirectory() as generated_raw:
            generated_dir = Path(generated_raw)
            command = ([str(generator_path)] if runner_path == generator_path else
                       [str(runner_path), str(generator_path)])
            command.extend([
                "--model", candidate_model["path"], "--definition", target_definition["id"],
                "--prefix", str(baseline["key"]["workload"]["frontier"]),
                "--token-file", str(token_file), "--output-dir", str(generated_dir),
            ])
            try:
                result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
            except OSError as error:
                raise GateError(f"cannot execute approved Lane C oracle: {error}") from error
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                raise GateError(f"approved Lane C oracle generator failed: {detail}")
            regenerated = [
                {"name": path.name, "sha256": sha256(path)}
                for path in sorted(generated_dir.glob("decode_*.logits.json"))
            ]
        if regenerated != reference_hashes:
            raise GateError("approved Lane C generator did not reproduce the oracle dumps")
    else:
        reference_section = (baseline["reference"].get("oracle_numerical")
                             if lane == "C" else baseline["reference"].get("numerical"))
        if not isinstance(reference_section, dict) or not isinstance(reference_section.get("files"), list):
            raise GateError(f"lane {lane} baseline has no bound numerical oracle")
        if lane == "C" and target_definition.get("id") != reference_section.get("definition_id"):
            raise GateError("Lane C target definition differs without declaring a new oracle")
        if reference_hashes != reference_section["files"]:
            label = "canonical oracle" if lane == "C" else "predecessor baseline"
            raise GateError(f"numerical reference is not the bound {label} artifact")
        if recorded["sources"].get("reference_manifest_sha256") != reference_section.get("manifest_sha256"):
            raise GateError("numerical reference manifest is not bound by the baseline")
    expected_bits = {"Q4_K": 4, "Q2_K": 2}.get(baseline["key"].get("quantization"))
    try:
        expected_prefix = int(baseline["key"]["workload"]["frontier"])
    except (KeyError, TypeError, ValueError) as error:
        raise GateError("baseline has an invalid numerical frontier") from error
    for item in recorded["sources"]["pairs"]:
        reference_dump = json.loads((reference_dir / item["name"]).read_text(encoding="utf-8"))
        candidate_dump = json.loads((candidate_dir / item["name"]).read_text(encoding="utf-8"))
        expected_reference_source = (
            "ds4-canonical-oracle" if lane == "C"
            else "ds4-bench-frozen-teacher")
        if reference_dump.get("source") != expected_reference_source:
            raise GateError("reference logit dump has an unexpected producer schema")
        if candidate_dump.get("source") != "ds4-bench-frozen-teacher":
            raise GateError("candidate logit dump has an unexpected producer schema")
        for dump in (reference_dump, candidate_dump):
            if expected_bits is not None and dump.get("quant_bits") != expected_bits:
                raise GateError("teacher-logit quantization differs from the baseline model")
            if dump.get("prefix_tokens") != expected_prefix:
                raise GateError("teacher-logit prefix differs from the baseline workload")
            if dump.get("dspark") is not False:
                raise GateError("ordinary numerical promotion cannot use DSpark logits")
        if reference_dump.get("model") != candidate_model["path"]:
            raise GateError("reference teacher logits came from a different model path")
        if candidate_dump.get("model") != candidate_model["path"]:
            raise GateError("candidate teacher logits came from a different model path")
    candidate_identity = read_manifest(candidate_manifest)
    for field, producer in (
            ("quality_launcher_sha256", repo / "run-tp-quality-score.sh"),
            ("worker_supervisor_sha256", repo / "scripts/tp-worker-supervisor.sh")):
        if candidate_identity.get(field) != sha256(producer):
            raise GateError(f"quality candidate {field} differs from the active verifier")
    if (candidate_identity.get("model") != candidate_model["path"] or
            candidate_identity.get("model_size") != str(candidate_model["size"]) or
            candidate_identity.get("model_sample_sha256") != candidate_model["sample_sha256"] or
            candidate_identity.get("source_commit") != source["commit"] or
            candidate_identity.get("source_dirty") != "0" or
            candidate_identity.get("ds4_sha256") != candidate_binary_sha256 or
            candidate_identity.get("toolchain_id") != toolchain["id"] or
            candidate_identity.get("prefix_tokens") != str(expected_prefix) or
            candidate_identity.get("file_count") != str(len(recorded["sources"]["pairs"])) or
            candidate_identity.get("frozen_token_sha256") !=
                baseline["key"]["workload"].get("frozen_token_sha256") or
            candidate_identity.get("dspark") != "0"):
        raise GateError("numerical candidate manifest does not match the ordinary candidate")
    expected_switches = {
        name: arms["candidate"] for name, arms in candidate_switches.items()
    }
    if manifest_switch_values(
            candidate_identity, candidate_switches,
            "numerical candidate manifest") != expected_switches:
        raise GateError("numerical candidate used different initialized switches")
    minimum = baseline["thresholds"][threshold_section]["min_teacher_steps"]
    if (recorded.get("baseline_id") != baseline_id or
            recorded.get("passed") is not True or
            recorded.get("far_margin_inversions") != 0 or
            not isinstance(recorded.get("steps"), int) or
            recorded["steps"] < minimum):
        raise GateError("numerical envelope did not pass the versioned baseline")
    return recorded


def verify_quality_evidence(repo: Path, root: Path, summary_path: Path,
                            baseline_id: str, baseline: dict,
                            candidate_model: dict, source: dict,
                            candidate_switches: dict,
                            candidate_binary_sha256: str,
                            reference_section: dict | None = None,
                            label: str = "predecessor") -> dict:
    try:
        recorded = json.loads(summary_path.read_text(encoding="utf-8"))
        sources = recorded["sources"]
        reference = Path(sources["reference"]).resolve()
        candidate = Path(sources["candidate"]).resolve()
        reference_manifest = Path(sources["reference_manifest"]).resolve()
        candidate_manifest = Path(sources["candidate_manifest"]).resolve()
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise GateError(f"invalid reference-score summary: {error}") from error
    for path in (reference, candidate, reference_manifest, candidate_manifest):
        if path != root and root not in path.parents:
            raise GateError(f"quality source escapes canonical research root: {path}")
    if reference == candidate:
        raise GateError("quality reference and candidate TSVs must differ")
    with tempfile.TemporaryDirectory() as raw:
        threshold = threshold_file(baseline, baseline_id, "quality", Path(raw))
        recomputed = rerun_json_tool([
            sys.executable, str(repo / "scripts" / "compare-quality-scores.py"),
            str(reference), str(candidate), "--thresholds", str(threshold),
            "--require-candidate-status",
        ], "quality score")
    ignored = {"thresholds_sha256"}
    if {key: value for key, value in recorded.items() if key not in ignored} != {
            key: value for key, value in recomputed.items() if key not in ignored}:
        raise GateError("reference-score summary does not match its source TSVs")
    if reference_section is None:
        reference_section = baseline["reference"]["quality"]
    if recorded.get("sources", {}).get("reference_sha256") != reference_section["sha256"]:
        raise GateError(f"quality reference is not the {label} artifact")
    if (recorded.get("sources", {}).get("reference_manifest_sha256") !=
            reference_section["manifest_sha256"]):
        raise GateError(f"quality reference manifest is not the {label} artifact")
    candidate_identity = read_manifest(candidate_manifest)
    if (candidate_identity.get("model") != candidate_model["path"] or
            candidate_identity.get("model_size") != str(candidate_model["size"]) or
            candidate_identity.get("model_sample_sha256") != candidate_model["sample_sha256"] or
            candidate_identity.get("source_commit") != source["commit"] or
            candidate_identity.get("source_dirty") != "0" or
            candidate_identity.get("ds4_sha256") != candidate_binary_sha256 or
            candidate_identity.get("dspark") != "0"):
        raise GateError("quality candidate manifest does not match the ordinary candidate")
    expected_switches = {
        name: arms["candidate"] for name, arms in candidate_switches.items()
    }
    if manifest_switch_values(
            candidate_identity, candidate_switches,
            "quality candidate manifest") != expected_switches:
        raise GateError("quality candidate used different initialized switches")
    if recorded.get("baseline_id") != baseline_id or recorded.get("passed") is not True:
        raise GateError("reference quality comparison did not pass")
    return recorded


def verify_screen_threshold_contract(root: Path, proof: dict,
                                     baseline_id: str, baseline: dict) -> None:
    expected = {"baseline_id": baseline_id,
                **baseline["thresholds"]["numerical"]}
    screens = [proof.get("diverse_screen"), proof.get("long_context_screen")]
    regressions = proof.get("ordinary_regressions")
    if isinstance(regressions, list):
        screens.extend(regressions)
    for screen in screens:
        trajectory = screen.get("trajectory") if isinstance(screen, dict) else None
        if not isinstance(trajectory, dict) or trajectory.get("mode") != "teacher":
            continue
        bound = trajectory.get("thresholds")
        if not isinstance(bound, dict):
            raise GateError("teacher screen has no bound threshold artifact")
        threshold_path = Path(str(bound.get("path", ""))).resolve()
        if (threshold_path != root and root not in threshold_path.parents) or \
                not threshold_path.is_file() or sha256(threshold_path) != bound.get("sha256"):
            raise GateError("teacher screen threshold artifact is invalid")
        try:
            actual = json.loads(threshold_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise GateError("teacher screen threshold artifact is not JSON") from error
        if actual != expected:
            raise GateError("teacher screen thresholds differ from the active baseline")


def environment_without_switches(environment: dict[str, str],
                                 switches: dict) -> dict[str, str]:
    return {name: value for name, value in environment.items()
            if name not in switches and name not in VOLATILE_BENCH_ENV}


def control_anchor_environment(manifest: dict[str, str], switches: dict,
                               label: str) -> dict[str, dict[str, str]]:
    environment = {
        field: environment_without_switches(
            parse_env(manifest[field], f"{label} {field}"), switches)
        for field in ENV_FIELDS
    }
    # Launchers may repeat an explicit trailing assignment in their common
    # defaults. It is redundant only when the same value is explicitly bound
    # in extra_env AND both final rank environments. Never infer defaults or
    # ignore an actual rank setting, even for reporting switches.
    environment["common_env"] = {
        name: value for name, value in environment["common_env"].items()
        if not all(environment[field].get(name) == value
                   for field in (*RANK_ENV_FIELDS, "extra_env"))
    }
    return environment


def verify_control_anchor(proof: dict, baseline: dict,
                          candidate_switches: dict) -> None:
    provider = proof["required_provider"]
    anchor = baseline["reference"]["performance"].get(provider)
    if not isinstance(anchor, dict):
        raise GateError("active baseline has no performance anchor for proof provider")
    pairs = proof["headline_pairs"]
    controls = [pair["control"]["manifest"] for pair in pairs]
    expected = control_anchor_environment(
        anchor["environment"], candidate_switches, "baseline")
    for manifest in controls:
        current = control_anchor_environment(manifest, candidate_switches, "control")
        for field in ENV_FIELDS:
            if current[field] != expected[field]:
                raise GateError(
                    f"control off-path environment differs from baseline in {field}")
    source_is_baseline = (
        proof["source_commits"]["control"] == baseline["key"]["source_commit"])
    binary_is_baseline = all(
        manifest.get("ds4_sha256") == anchor["ds4_sha256"]
        for manifest in controls)
    if source_is_baseline and binary_is_baseline:
        return
    maximum_regression = baseline["thresholds"]["performance"][
        "maximum_control_regression"]
    for metric in ("prefill", "decode"):
        observed = proof["performance"]["metrics"][metric][
            "geometric_mean_control_tps"]
        required = anchor["geometric_mean_tps"][metric] * (1.0 - maximum_regression)
        if observed < required:
            raise GateError(
                f"same-commit control {metric} throughput is below its baseline anchor")


def verify_promotion_evidence(repo: Path, root: Path, proof_path: Path,
                              candidate_id: str, value: dict,
                              baseline: dict) -> tuple[dict, list[dict[str, str]]]:
    environment = os.environ.copy()
    environment["DS4_RESEARCH_ROOT"] = str(root)
    result = subprocess.run(
        [sys.executable, str(repo / "scripts" / "promotion-proof.py"),
         "verify", str(proof_path)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=environment)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise GateError(f"promotion proof failed independent recomputation: {detail}")
    try:
        proof = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise GateError("promotion proof returned invalid JSON") from error
    source = value["source"]
    source_commits = proof.get("source_commits")
    intent = validate_promotion_intent(value.get("promotion_intent"))
    performance_policy = proof.get("performance_policy")
    if (not isinstance(performance_policy, dict) or
            performance_policy.get("contract") != baseline["thresholds"]["performance"] or
            performance_policy.get("target_metrics") != intent["target_metrics"] or
            performance_policy.get("candidate_switches") != intent["candidate_switches"]):
        raise GateError("promotion proof differs from initialized performance policy")
    pairs = proof.get("headline_pairs")
    registered_pairs = headline_pair_state(root, candidate_id, value)
    if (not isinstance(pairs, list) or
            [item.get("order") for item in pairs] !=
                intent["headline_pair_order"][:len(pairs)] or
            [item.get("pair_id") for item in pairs] !=
                [item["pair_id"] for item in registered_pairs] or
            proof.get("required_provider") !=
                baseline["thresholds"]["performance"]["required_provider"]):
        raise GateError(
            "promotion proof differs from its prospective pair/provider contract")
    for index, (pair, registered) in enumerate(zip(pairs, registered_pairs)):
        began = iso_utc_ns(registered["recorded_utc"])
        next_began = (iso_utc_ns(registered_pairs[index + 1]["recorded_utc"])
                      if index + 1 < len(registered_pairs) else None)
        for arm in ("control", "candidate"):
            manifest = pair.get(arm, {}).get("manifest")
            if not isinstance(manifest, dict):
                raise GateError("promotion proof has an invalid headline manifest")
            journaled_run = registered.get("runs", {}).get(arm)
            run_result = (journaled_run.get("result", {})
                          if isinstance(journaled_run, dict) else {})
            result_csv = run_result.get("result_csv", {})
            registered_artifacts = dict(run_result.get("artifacts", {}))
            if isinstance(result_csv, dict) and result_csv.get("exists") is True:
                registered_artifacts["csv"] = {
                    "path": result_csv.get("path"),
                    "sha256": result_csv.get("sha256"),
                }
            if (manifest.get("candidate_id") != candidate_id or
                    not isinstance(journaled_run, dict) or
                    manifest.get("run_id") != journaled_run.get("run_id") or
                    pair.get(arm, {}).get("artifacts") != registered_artifacts or
                    run_result.get("observations") !=
                        SUCCESSFUL_HEADLINE_OBSERVATIONS):
                raise GateError(
                    "headline proof differs from its journaled run/result artifacts")
            run_time = benchmark_run_utc_ns(manifest.get("run_id", ""))
            if run_time <= began or (next_began is not None and run_time >= next_began):
                raise GateError(
                    "headline run falls outside its prospective pair window")
    if (proof.get("passed") is not True or
            proof.get("performance", {}).get("merge_eligible") is not True or
            proof.get("stage") != "promotion" or
            proof.get("candidate_id") != candidate_id or
            proof.get("lane") != value["lane"] or
            proof.get("baseline_id") != value["baseline_id"] or
            proof.get("baseline_fnv64") != baseline["reference"]["fnv64"] or
            not isinstance(source_commits, dict) or
            source_commits.get("candidate") != source["commit"] or
            source_commits.get("control") not in {
                source["commit"], baseline["key"].get("source_commit")}):
        raise GateError("promotion proof identity differs from the candidate dossier")
    if not isinstance(pairs, list) or len(pairs) not in {3, 5, 7, 9}:
        raise GateError("promotion proof has invalid headline timing coverage")
    manifests = [item["candidate"]["manifest"] for item in pairs]
    if not all(isinstance(item, dict) for item in manifests):
        raise GateError("promotion proof has invalid candidate manifests")
    verify_screen_threshold_contract(root, proof, value["baseline_id"], baseline)
    verify_control_anchor(proof, baseline, intent["candidate_switches"])
    return proof, manifests


def check_candidate(repo: Path, root: Path, candidate_id: str) -> tuple[Path, dict, dict]:
    dossier, value = load_candidate(root, candidate_id)
    if (dossier / "CLOSED.json").exists():
        raise GateError("closed candidates cannot be checked or promoted")
    if value.get("schema_version") != 2 or value.get("candidate_id") != candidate_id:
        raise GateError("candidate schema/id mismatch")
    lane = str(value.get("lane", "")).upper()
    if lane not in LANES or value.get("lane") != lane:
        raise GateError("invalid candidate lane")
    source = value.get("source")
    if (not isinstance(source, dict) or
            not re.fullmatch(r"[0-9a-f]{40}", str(source.get("commit", "")))):
        raise GateError("source.commit must be a full Git object id")
    if source.get("dirty") is not False:
        raise GateError("performance promotion requires a clean source worktree snapshot")
    initialization = require_candidate_init(root, candidate_id, value)
    if run_git(repo, "rev-parse", "HEAD") != source["commit"]:
        raise GateError("gate must run from the candidate source commit")
    if run_git(repo, "status", "--porcelain=v1", "-uall"):
        raise GateError("gate must run from a clean candidate worktree")
    _, baseline = load_baseline(root, str(value.get("baseline_id", "")))
    if baseline["schema_version"] != 2:
        raise GateError("legacy baselines are diagnostic-only and cannot sponsor promotion")
    require_active_baseline(root, value["baseline_id"], baseline)
    if initialization["sequence"] <= baseline_authority_sequence(
            root, value["baseline_id"]):
        raise GateError(
            "candidate initialization must follow the active baseline policy; "
            "close and restart the candidate")
    verify_verifier_identity(repo, baseline)

    evidence = value.get("evidence")
    if not isinstance(evidence, list):
        raise GateError("evidence must be a list")
    kinds: list[str] = []
    paths_by_kind: dict[str, list[Path]] = {}
    allowed_kinds = (COMMON_KINDS | REVIEW_KINDS |
                     set().union(*LANE_KINDS.values()) |
                     DSPARK_KINDS | {"quality-anchor-score"})
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise GateError(f"evidence[{index}] must be an object")
        kind = str(item.get("kind", ""))
        expected = str(item.get("sha256", ""))
        if kind not in allowed_kinds or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise GateError(f"evidence[{index}] has an invalid kind or SHA-256")
        path = Path(str(item.get("path", "")))
        if not path.is_absolute():
            path = dossier / path
        path = path.resolve()
        if path != root and root not in path.parents:
            raise GateError(f"evidence escapes canonical root: {path}")
        if not path.is_file() or sha256(path) != expected:
            raise GateError(f"evidence hash mismatch: {path}")
        kinds.append(kind)
        paths_by_kind.setdefault(kind, []).append(path)
    required_kinds = COMMON_KINDS | LANE_KINDS[lane]
    if lane == "C":
        required_kinds |= REVIEW_KINDS
    if value.get("dspark") is True:
        required_kinds |= DSPARK_KINDS
    missing = sorted(required_kinds - set(kinds))
    if missing:
        raise GateError("missing evidence kinds: " + ", ".join(missing))
    if len(paths_by_kind.get("promotion-proof", [])) != 1:
        raise GateError("exactly one typed promotion proof is required")
    if lane == "C":
        verify_review_evidence(root, evidence, "Lane C candidate", value)

    model = value.get("model")
    if (not isinstance(model, dict) or
            not re.fullmatch(r"[0-9a-f]{64}", str(model.get("sample_sha256", ""))) or
            not re.fullmatch(r"[0-9a-f]{64}", str(model.get("sha256", ""))) or
            not isinstance(model.get("size"), int) or model["size"] <= 0 or
            not isinstance(model.get("quantization"), str) or
            not isinstance(model.get("path"), str) or
            not Path(model["path"]).is_absolute()):
        raise GateError("candidate model identity is incomplete")
    try:
        actual_size, actual_sample = sampled_model_sha256(Path(model["path"]))
    except OSError as error:
        raise GateError(f"cannot verify candidate model artifact: {error}") from error
    if (actual_size != model["size"] or actual_sample != model["sample_sha256"] or
            sha256(Path(model["path"])) != model["sha256"]):
        raise GateError("candidate model bytes differ from the dossier identity")
    toolchain = value.get("toolchain")
    workload = value.get("workload")
    transport = value.get("transport")
    if not isinstance(toolchain, dict) or not isinstance(toolchain.get("id"), str) or not toolchain["id"]:
        raise GateError("candidate toolchain.id is required")
    target_definition = value.get("target_definition")
    if not isinstance(target_definition, dict):
        raise GateError("candidate target_definition is missing")
    if lane == "C" and (type(target_definition.get("changed")) is not bool or
                         not isinstance(target_definition.get("id"), str) or
                         not target_definition["id"]):
        raise GateError("Lane C requires an explicit target definition id")
    if (not isinstance(workload, dict) or workload.get("architecture") != "gfx1151" or
            workload.get("tp_degree") != 2 or
            workload.get("decode_mode") != "ordinary-greedy" or
            not isinstance(workload.get("workload_id"), str) or not workload["workload_id"]):
        raise GateError("candidate workload is not the ordinary balanced gfx1151 TP=2 contract")
    candidate_layout = tp_layout_contract(workload, "candidate workload")
    providers = transport.get("providers") if isinstance(transport, dict) else None
    if (not isinstance(providers, list) or not providers or
            len(set(providers)) != len(providers) or
            any(provider not in {"odinlink", "roce-v2"} for provider in providers)):
        raise GateError("candidate transport.providers must name validated RDMA providers")

    baseline_key = baseline["key"]
    if (baseline_key.get("model_sample_sha256") != model["sample_sha256"] or
            baseline_key.get("model_sha256") != model["sha256"] or
            baseline_key.get("model_size") != model["size"] or
            baseline_key.get("quantization") != model["quantization"]):
        raise GateError("candidate model does not match the active baseline")
    if candidate_layout != tp_layout_contract(baseline_key, "active baseline key"):
        raise GateError("candidate TP layout differs from the active baseline")

    proof, benchmark_manifests = verify_promotion_evidence(
        repo, root, paths_by_kind["promotion-proof"][0], candidate_id, value, baseline)
    if proof.get("required_provider") not in providers:
        raise GateError("promotion proof used an undeclared RDMA provider")
    baseline_workload = baseline_key.get("workload")
    if not isinstance(baseline_workload, dict):
        raise GateError("active baseline has no workload contract")
    if ("prefill_batch" in baseline_workload and
            workload.get("prefill_batch") != baseline_workload["prefill_batch"]):
        raise GateError("candidate prefill_batch differs from active baseline")
    for manifest in benchmark_manifests:
        if (manifest.get("model") != model["path"] or
                manifest.get("model_size") != str(model["size"]) or
                manifest.get("model_sample_sha256") != model["sample_sha256"] or
                manifest.get("toolchain_id") != toolchain["id"]):
            raise GateError("headline benchmark model/toolchain differs from dossier")
        for key in workload_manifest_fields(baseline_workload):
            if manifest.get(key) != str(baseline_workload.get(key, "")):
                raise GateError(f"headline benchmark differs from baseline workload: {key}")
        verify_tp_layout_manifest(manifest, candidate_layout,
                                  "headline benchmark layout")
    candidate_binary_hashes = {
        manifest.get("ds4_sha256") for manifest in benchmark_manifests
    }
    if (len(candidate_binary_hashes) != 1 or
            not re.fullmatch(r"[0-9a-f]{64}",
                             str(next(iter(candidate_binary_hashes), "")))):
        raise GateError("headline candidate runs used different binaries")
    candidate_binary_sha256 = next(iter(candidate_binary_hashes))
    candidate_bench_hashes = {
        manifest.get("ds4_bench_tp_sha256") for manifest in benchmark_manifests
    }
    if (len(candidate_bench_hashes) != 1 or
            not re.fullmatch(r"[0-9a-f]{64}",
                             str(next(iter(candidate_bench_hashes), "")))):
        raise GateError(
            "headline candidate runs used different benchmark executables")
    candidate_switches = value["promotion_intent"]["candidate_switches"]
    expected_switches = {
        name: arms["candidate"] for name, arms in candidate_switches.items()
    }
    for manifest in benchmark_manifests:
        if manifest_switch_values(
                manifest, candidate_switches,
                "headline candidate manifest") != expected_switches:
            raise GateError("headline candidate used different initialized switches")
    screen_manifests = []
    for screen_name in ("diverse_screen", "long_context_screen"):
        screen = proof.get(screen_name)
        try:
            pair = screen["pair"]
            screen_manifests.extend(
                (pair["control"]["manifest"], pair["candidate"]["manifest"]))
        except (KeyError, TypeError):
            raise GateError(f"promotion proof has invalid {screen_name}") from None
    for manifest in screen_manifests:
        if (not isinstance(manifest, dict) or
                manifest.get("model") != model["path"] or
                manifest.get("model_size") != str(model["size"]) or
                manifest.get("model_sample_sha256") != model["sample_sha256"] or
                manifest.get("toolchain_id") != toolchain["id"]):
            raise GateError("4K/8K benchmark model/toolchain differs from dossier")
        verify_tp_layout_manifest(manifest, candidate_layout,
                                  "4K/8K benchmark layout")
    candidate_fingerprint = proof["candidate_fnv64"]
    if lane == "A" and candidate_fingerprint != baseline["reference"]["fnv64"]:
        raise GateError("Lane A did not reproduce the active baseline fingerprint")

    numerical_result = quality_result = quality_anchor_result = None
    if lane in {"B", "C"}:
        numerical_paths = paths_by_kind.get("numerical-envelope", [])
        quality_paths = paths_by_kind.get("reference-score", [])
        if len(numerical_paths) != 1 or len(quality_paths) != 1:
            raise GateError("Lane B/C requires one numerical and one quality summary")
        numerical_result = verify_numerical_evidence(
            repo, root, numerical_paths[0], value["baseline_id"], baseline,
            model, lane, source, toolchain, target_definition,
            candidate_switches, candidate_binary_sha256)
        quality_result = verify_quality_evidence(
            repo, root, quality_paths[0], value["baseline_id"], baseline,
            model, source, candidate_switches, candidate_binary_sha256)
        quality_anchor = baseline["reference"].get(
            "quality_anchor", baseline["reference"]["quality"])
        if quality_anchor == baseline["reference"]["quality"]:
            quality_anchor_result = quality_result
        else:
            anchor_paths = paths_by_kind.get("quality-anchor-score", [])
            if len(anchor_paths) != 1:
                raise GateError("Lane B/C requires one quality-anchor summary")
            quality_anchor_result = verify_quality_evidence(
                repo, root, anchor_paths[0], value["baseline_id"], baseline,
                model, source, candidate_switches, candidate_binary_sha256,
                quality_anchor, "immutable anchor")

    invalidation_count = headline_invalidation_count(root, candidate_id)
    prior_candidates = prior_formal_candidates(
        root, candidate_id, value, candidate_binary_sha256)
    print(f"PASS candidate={candidate_id} lane={lane} evidence={len(evidence)} "
          f"headline_invalidations={invalidation_count} "
          f"prior_formal_candidate_count={len(prior_candidates)} "
          "prior_formal_candidates=" + json.dumps(prior_candidates, separators=(",", ":")))
    return dossier, value, {
        "fingerprint": candidate_fingerprint,
        "baseline": baseline,
        "benchmark_manifests": benchmark_manifests,
        "promotion_proof": proof,
        "numerical_result": numerical_result,
        "quality_result": quality_result,
        "quality_anchor_result": quality_anchor_result,
        "target_definition": target_definition,
        "headline_invalidation_count": invalidation_count,
        "prior_formal_candidates": prior_candidates,
        "candidate_binary_sha256": candidate_binary_sha256,
        "candidate_bench_sha256": next(iter(candidate_bench_hashes)),
    }


@governance_mutation(1)
def promote_candidate(repo: Path, root: Path, candidate_id: str) -> None:
    dossier, _ = load_candidate(root, candidate_id)
    if (dossier / "PROMOTED.json").exists():
        raise GateError("candidate was already promoted; promotion is append-only")
    dossier, value, derived = check_candidate(repo, root, candidate_id)
    candidate_file = dossier / "candidate.json"
    new_baseline_id = None
    baseline_path = None
    if value["lane"] in {"B", "C"}:
        model = value["model"]
        workload = value["workload"]
        toolchain = value["toolchain"]
        proof = derived["promotion_proof"]
        provider = proof["required_provider"]
        candidate_manifests = [
            pair["candidate"]["manifest"] for pair in proof["headline_pairs"]
        ]
        environment = normalized_manifest_environment(
            candidate_manifests[0], "promoted candidate manifest")
        record = {
            "schema_version": derived["baseline"]["schema_version"],
            "kind": "ds4-numerical-baseline",
            "key": {
                "model_sample_sha256": model["sample_sha256"],
                "model_sha256": model["sha256"],
                "model_size": model["size"],
                "quantization": model["quantization"],
                "source_commit": value["source"]["commit"],
                "toolchain_id": toolchain["id"],
                "architecture": workload["architecture"],
                "tp_degree": workload["tp_degree"],
                "decode_mode": workload["decode_mode"],
                "workload_id": workload["workload_id"],
                "workload": {
                    key: derived["benchmark_manifests"][0][key]
                    for key in workload_manifest_fields(
                        derived["baseline"]["key"]["workload"])
                },
                "rdma_providers": [provider],
            },
            "reference": {
                "fnv64": derived["fingerprint"],
                "performance": {
                    provider: {
                        "runs": len(proof["headline_pairs"]),
                        "geometric_mean_tps": {
                            metric: proof["performance"]["metrics"][metric][
                                "geometric_mean_candidate_tps"]
                            for metric in ("prefill", "decode")
                        },
                        "environment": environment,
                        "ds4_sha256": candidate_manifests[0]["ds4_sha256"],
                        "ds4_bench_tp_sha256":
                            derived["candidate_bench_sha256"],
                    },
                },
                "numerical": {
                    "files": [
                        {"name": item["name"], "sha256": item["candidate_sha256"]}
                        for item in derived["numerical_result"]["sources"]["pairs"]
                    ],
                    "manifest_sha256": derived["numerical_result"]["sources"]["candidate_manifest_sha256"],
                },
                "quality": {
                    "sha256": derived["quality_result"]["sources"]["candidate_sha256"],
                    "manifest_sha256": derived["quality_result"]["sources"]["candidate_manifest_sha256"],
                },
                "quality_anchor": derived["baseline"]["reference"].get(
                    "quality_anchor", derived["baseline"]["reference"]["quality"]),
            },
            "thresholds": derived["baseline"]["thresholds"],
            "provenance": {
                "candidate_id": candidate_id,
                "lane_origin": value["lane"],
                "replaces": value["baseline_id"],
                "candidate_json_sha256": sha256(candidate_file),
                "calibration": derived["baseline"]["provenance"]["calibration"],
                "verifier_sha256": derived["baseline"]["provenance"][
                    "verifier_sha256"],
                "performance_method": derived["baseline"]["provenance"][
                    "performance_method"],
                "evidence": value["evidence"],
            },
            "oracle_generators": derived["baseline"].get("oracle_generators", []),
        }
        record["key"].update(tp_layout_contract(workload, "candidate workload"))
        record["key"]["workload"]["frozen_token_sha256"] = \
            derived["baseline"]["key"]["workload"]["frozen_token_sha256"]
        record["scope_sha256"] = baseline_scope_sha256(record)
        if value["lane"] == "C" and derived["target_definition"].get("changed") is True:
            record["reference"]["oracle_numerical"] = {
                "definition_id": derived["target_definition"]["id"],
                "files": [
                    {"name": item["name"], "sha256": item["reference_sha256"]}
                    for item in derived["numerical_result"]["sources"]["pairs"]
                ],
                "manifest_sha256": derived["numerical_result"]["sources"]["reference_manifest_sha256"],
            }
        elif derived["baseline"]["reference"].get("oracle_numerical") is not None:
            record["reference"]["oracle_numerical"] = derived["baseline"]["reference"]["oracle_numerical"]
        digest = canonical_sha256(record)
        baseline_path = root / "baselines" / "sha256" / f"{digest}.json"
        if baseline_path.exists():
            raise GateError(f"refusing to overwrite existing baseline: {baseline_path}")
        new_baseline_id = f"sha256:{digest}"
    promoted = {
        "schema_version": 2,
        "candidate_id": candidate_id,
        "lane": value["lane"],
        "source_commit": value["source"]["commit"],
        "candidate_json_sha256": sha256(candidate_file),
        "new_baseline_id": new_baseline_id,
        "headline_invalidation_count": derived["headline_invalidation_count"],
        "prior_formal_candidate_count": len(derived["prior_formal_candidates"]),
        "prior_formal_candidates": derived["prior_formal_candidates"],
        "candidate_binary_sha256": derived["candidate_binary_sha256"],
        "promoted_utc": datetime.now(timezone.utc).isoformat(),
    }
    promoted_path = dossier / "PROMOTED.json"
    try:
        if baseline_path is not None:
            atomic_json(baseline_path, record)
        atomic_json(promoted_path, promoted)
        append_event(root, {
            "type": "candidate-promote", "candidate_id": candidate_id,
            "promoted_json_sha256": sha256(promoted_path),
            "new_baseline_id": new_baseline_id,
            "headline_invalidation_count": derived["headline_invalidation_count"],
            "prior_formal_candidate_count": len(derived["prior_formal_candidates"]),
            "prior_formal_candidates": derived["prior_formal_candidates"],
            "candidate_binary_sha256": derived["candidate_binary_sha256"],
        })
    except Exception:
        promoted_path.unlink(missing_ok=True)
        if baseline_path is not None:
            baseline_path.unlink(missing_ok=True)
        raise
    print(promoted_path)


@governance_mutation(1)
def amend_baseline(repo: Path, root: Path, amendment_path: Path) -> None:
    amendment_path = amendment_path.resolve()
    if amendment_path != root and root not in amendment_path.parents:
        raise GateError("baseline amendment must live in the canonical research root")
    try:
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"invalid baseline amendment: {error}") from error
    required = {
        "schema_version", "kind", "amendment_id", "baseline_id", "rationale",
        "add_oracle_generator", "threshold_updates", "calibration",
        "calibration_sha256", "timing_noise_qualification", "verifier",
        "evidence",
    }
    if (not isinstance(amendment, dict) or set(amendment) != required or
            amendment.get("schema_version") != 2 or
            amendment.get("kind") != "ds4-baseline-amendment" or
            not ID_RE.fullmatch(str(amendment.get("amendment_id", ""))) or
            not isinstance(amendment.get("rationale"), str) or
            not amendment["rationale"].strip()):
        raise GateError("invalid baseline amendment identity or rationale")
    _, predecessor = load_baseline(root, str(amendment.get("baseline_id", "")))
    if predecessor["schema_version"] != 2:
        raise GateError("legacy baselines cannot sponsor production amendments")
    require_active_baseline(root, amendment["baseline_id"], predecessor)

    updates = amendment["threshold_updates"]
    if (not isinstance(updates, dict) or not updates or any(
            key not in {"numerical", "oracle_numerical", "quality", "performance"}
            for key in updates)):
        raise GateError("baseline amendment has unsupported threshold updates")
    performance_only = set(updates) == {"performance"}
    if "performance" in updates and not performance_only:
        raise GateError(
            "performance policy changes must be isolated from numerical/quality amendments")

    record = json.loads(json.dumps(predecessor))
    generator = amendment["add_oracle_generator"]
    performance_method = predecessor["provenance"]["performance_method"]
    if performance_only:
        if (generator is not None or amendment["calibration"] is not None or
                amendment["calibration_sha256"] is not None):
            raise GateError(
                "performance-only amendment must not use oracle or numerical calibration")
        reject_open_scope_candidates(root, baseline_scope_sha256(predecessor))
        proposed_performance = validate_performance_thresholds(
            updates["performance"], "performance")
        performance_method = verify_performance_method_evidence(
            repo, root, proposed_performance,
            amendment["timing_noise_qualification"], "performance amendment",
            baseline=predecessor)
        previous_verifiers = predecessor["provenance"]["verifier_sha256"]
        active_verifiers = verifier_sha256(repo)
        for name in ("numerical", "quality"):
            if active_verifiers[name] != previous_verifiers[name]:
                raise GateError(
                    "performance-only amendment cannot change calibrated "
                    f"{name} comparator")
        record["thresholds"]["performance"] = proposed_performance
        calibration = predecessor["provenance"]["calibration"]
    else:
        if amendment["timing_noise_qualification"] is not None:
            raise GateError(
                "numerical/quality amendment must not replace timing-noise evidence")
        if not re.fullmatch(
                r"[0-9a-f]{64}", str(amendment.get("calibration_sha256", ""))):
            raise GateError("numerical/quality amendment requires calibration SHA-256")

    if generator is not None:
        if (not isinstance(generator, dict) or
                not ID_RE.fullmatch(str(generator.get("id", "")))):
            raise GateError("baseline amendment has an invalid oracle generator")
        if any(item.get("id") == generator["id"]
               for item in record.get("oracle_generators", [])):
            raise GateError("oracle generator id already exists in the baseline lineage")
        verify_generator_closure(repo, root, generator)
        record.setdefault("oracle_generators", []).append(generator)

    previous_thresholds = {}
    for section, thresholds in (() if performance_only else updates.items()):
        if not isinstance(thresholds, dict):
            raise GateError(f"invalid {section} threshold update")
        if section in {"numerical", "oracle_numerical"}:
            validated = validate_numerical_thresholds(thresholds, section)
            if validated.get("schema_version") != 2:
                raise GateError(f"{section} amendment requires threshold schema v2")
            previous = predecessor["thresholds"].get(
                section, record["thresholds"]["numerical"])
        else:
            validated = validate_quality_thresholds(thresholds, "quality")
            if validated.get("schema_version") != 2:
                raise GateError("quality amendment requires threshold schema v2")
            previous = predecessor["thresholds"]["quality"]
        previous_thresholds[section] = previous
        record.setdefault("thresholds", {})[section] = thresholds
    if ("oracle_numerical" in updates and
            "oracle_numerical" not in predecessor["thresholds"] and generator is None):
        raise GateError("first canonical-oracle envelope must adopt its generator")
    if generator is not None and "oracle_numerical" not in record["thresholds"]:
        raise GateError("an adopted oracle generator requires a canonical-oracle envelope")
    if (len(record["reference"]["numerical"]["files"]) <
            record["thresholds"]["numerical"]["min_teacher_steps"]):
        raise GateError("numerical threshold update exceeds the bound reference length")
    if ("oracle_numerical" in record["thresholds"] and
            len(record["reference"]["numerical"]["files"]) <
            record["thresholds"]["oracle_numerical"]["min_teacher_steps"]):
        raise GateError("oracle threshold update exceeds the frozen reference length")

    if not performance_only:
        calibration = evaluate_calibration(
            repo, root, amendment["calibration"],
            baseline_scope_sha256(predecessor), previous_thresholds, updates)
        if amendment["calibration_sha256"] != calibration["calibration_sha256"]:
            raise GateError("baseline amendment calibration digest does not recompute")
    reviewed_verifier = reviewed_verifier_identity(
        repo, amendment["verifier"], "baseline amendment")
    reviews = verify_review_evidence(root, amendment["evidence"],
                                     "baseline amendment", amendment)
    record["reference"].setdefault("quality_anchor", record["reference"]["quality"])
    record["scope_sha256"] = baseline_scope_sha256(record)
    record["provenance"] = {
        "amendment_id": amendment["amendment_id"],
        "amendment_kind": ("performance-policy" if performance_only
                           else "governance-only"),
        "replaces": amendment["baseline_id"],
        "amendment_json_sha256": sha256(amendment_path),
        "calibration": calibration,
        "verifier_source_commit": reviewed_verifier["source_commit"],
        "verifier_sha256": reviewed_verifier["sha256"],
        "performance_method": performance_method,
        "evidence": reviews,
    }
    digest = canonical_sha256(record)
    baseline_path = root / "baselines" / "sha256" / f"{digest}.json"
    if baseline_path.exists():
        raise GateError(f"refusing to overwrite existing baseline: {baseline_path}")
    try:
        atomic_json(baseline_path, record)
        load_baseline(root, f"sha256:{digest}")
        append_event(root, {
            "type": "baseline-amend", "amendment_id": amendment["amendment_id"],
            "replaces": amendment["baseline_id"],
            "new_baseline_id": f"sha256:{digest}",
            "calibration_sha256": (
                None if performance_only else calibration["calibration_sha256"]),
            "performance_policy_sha256": canonical_sha256(
                record["thresholds"]["performance"]),
        })
    except Exception:
        baseline_path.unlink(missing_ok=True)
        raise
    print(f"sha256:{digest}")


def evaluate_calibration_request(repo: Path, root: Path, request_path: Path,
                                 output: Path | None) -> None:
    request_path = request_path.resolve()
    if request_path != root and root not in request_path.parents:
        raise GateError("calibration request must live in the canonical research root")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "kind", "scope_sha256", "previous_thresholds",
        "proposed_thresholds", "calibration",
    }
    if (not isinstance(request, dict) or set(request) != required or
            request.get("schema_version") != 1 or
            request.get("kind") != "ds4-calibration-request" or
            not re.fullmatch(r"[0-9a-f]{64}", str(request.get("scope_sha256", ""))) or
            not isinstance(request.get("proposed_thresholds"), dict)):
        raise GateError("invalid calibration request")
    proposed = request["proposed_thresholds"]
    previous = request["previous_thresholds"]
    if previous is not None and not isinstance(previous, dict):
        raise GateError("previous_thresholds must be an object or null")
    for name, thresholds in proposed.items():
        if name in {"numerical", "oracle_numerical"}:
            validate_numerical_thresholds(thresholds, name)
        elif name == "quality":
            validate_quality_thresholds(thresholds, name)
        else:
            raise GateError(f"unsupported calibration section: {name}")
        if previous is not None and name not in previous:
            raise GateError(f"previous_thresholds is missing {name}")
    result = evaluate_calibration(
        repo, root, request["calibration"], request["scope_sha256"],
        previous, proposed)
    if output is not None:
        output = output.resolve()
        if output != root and root not in output.parents:
            raise GateError("calibration output must live in the canonical research root")
        atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("candidate_id")
    init.add_argument("lane")
    init.add_argument("--target-metric", action="append", choices=("prefill", "decode"),
                      default=[])
    init.add_argument("--switch", action="append", default=[],
                      metavar="NAME=CONTROL,CANDIDATE")
    init.add_argument("--first-order", choices=("AB", "BA"), default="AB")
    for command in ("check", "promote"):
        child = sub.add_parser(command)
        child.add_argument("candidate_id")
    amend = sub.add_parser("amend-baseline")
    amend.add_argument("amendment", type=Path)
    genesis = sub.add_parser("bootstrap-baseline")
    genesis.add_argument("genesis", type=Path)
    register = sub.add_parser("register-control")
    register.add_argument("descriptor", type=Path)
    close = sub.add_parser("close-candidate")
    close.add_argument("candidate_id")
    close.add_argument("--reason", required=True)
    begin_pair = sub.add_parser("begin-pair")
    begin_pair.add_argument("candidate_id")
    record_run = sub.add_parser("record-run")
    record_run.add_argument("candidate_id")
    record_run.add_argument("pair_id")
    record_run.add_argument("order", choices=("AB", "BA"))
    record_run.add_argument("arm", choices=("control", "candidate"))
    record_run.add_argument("run_id")
    record_result = sub.add_parser("record-result")
    record_result.add_argument("candidate_id")
    record_result.add_argument("pair_id")
    record_result.add_argument("run_id")
    record_result.add_argument("--manifest", type=Path, required=True)
    record_result.add_argument("--coordinator-log", type=Path, required=True)
    record_result.add_argument("--coordinator-status", type=Path, required=True)
    record_result.add_argument("--worker-log", type=Path, required=True)
    record_result.add_argument("--worker-status", type=Path, required=True)
    invalidate_pair = sub.add_parser("invalidate-pair")
    invalidate_pair.add_argument("candidate_id")
    invalidate_pair.add_argument("pair_id")
    invalidate_pair.add_argument("--run-id", required=True)
    invalidate_pair.add_argument("--manifest", type=Path, required=True)
    invalidate_pair.add_argument("--coordinator-log", type=Path, required=True)
    invalidate_pair.add_argument("--coordinator-status", type=Path, required=True)
    invalidate_pair.add_argument("--worker-log", type=Path, required=True)
    invalidate_pair.add_argument("--worker-status", type=Path, required=True)
    invalidate_pair.add_argument("--reason", required=True)
    classify_result = sub.add_parser("classify-headline-result")
    classify_result.add_argument("csv", type=Path)
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("request", type=Path)
    calibrate.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        repo, root = roots()
        if args.command == "init":
            init_candidate(repo, root, args.candidate_id, args.lane,
                           args.target_metric, args.switch, args.first_order)
        elif args.command == "check":
            check_candidate(repo, root, args.candidate_id)
        elif args.command == "promote":
            promote_candidate(repo, root, args.candidate_id)
        elif args.command == "amend-baseline":
            amend_baseline(repo, root, args.amendment)
        elif args.command == "bootstrap-baseline":
            bootstrap_baseline(repo, root, args.genesis)
        elif args.command == "register-control":
            print(register_control(repo, root, args.descriptor))
        elif args.command == "close-candidate":
            close_candidate(root, args.candidate_id, args.reason)
        elif args.command == "begin-pair":
            begin_headline_pair(repo, root, args.candidate_id)
        elif args.command == "record-run":
            record_headline_run(
                root, args.candidate_id, args.pair_id, args.order,
                args.arm, args.run_id)
        elif args.command == "record-result":
            record_headline_result(
                root, args.candidate_id, args.pair_id, args.run_id,
                args.manifest, args.coordinator_log, args.coordinator_status,
                args.worker_log, args.worker_status)
        elif args.command == "invalidate-pair":
            invalidate_headline_pair(
                root, args.candidate_id, args.pair_id, args.run_id,
                args.manifest, args.coordinator_log, args.coordinator_status,
                args.worker_log, args.worker_status, args.reason)
        elif args.command == "classify-headline-result":
            csv_path = args.csv.expanduser().resolve()
            if csv_path != root and root not in csv_path.parents:
                raise GateError("headline result CSV escapes DS4_RESEARCH_ROOT")
            print("1" if headline_result_is_complete(csv_path) else "0")
        else:
            evaluate_calibration_request(repo, root, args.request, args.output)
    except (ControlError, GateError, OSError, json.JSONDecodeError,
            subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
