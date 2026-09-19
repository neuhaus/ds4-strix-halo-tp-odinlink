#!/usr/bin/env python3
"""Create and reverify one typed DS4 merge-promotion proof.

The proof deliberately reuses recorded benchmark artifacts for workload,
trajectory, transport, cache, semantic, rollback, and timing decisions. It is
the final merge gate; exploratory runs do not need to satisfy this schema.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ds4_gate_stats import (StatsError, exact_sign_test,
                            paired_log_ratio_interval,
                            paired_log_ratio_repeated_interval)


class ProofError(RuntimeError):
    pass


def strictly_exceeds(value: float, boundary: float) -> bool:
    """Apply a strict performance margin without admitting float-equal ties."""
    return value > boundary and not math.isclose(
        value, boundary, rel_tol=1e-12, abs_tol=1e-12)


DIVERSE_PROMPT_SHA256 = "24d19432acab4d4cd2971d938b3c013fcfad1010ed701218bc7bdc1b630ecfef"
DEEPSEEK_PROMPT_SHA256 = "6dff0f4bc6000881259d96b2126b9c4f86f377efbaaa349e0a49d6da0435d34b"
LONG_CONTEXT_PROMPT = "tests/long_context_security_prompt.txt"
LONG_CONTEXT_PROMPT_SHA256 = "e7c1a2cadf781d274cc26bd251d532fe1b9e632080da97e3eb4684741e7cc308"
REPEATED_STUDENT_CALIBRATION_SHA256 = \
    "ed429ce50d58016e01a8276f2004f5c4777f896f1a23c98eb17f81c87cb5abde"
REQUIRED_REGRESSIONS = {"deepseek-0731-q4", "deepseek-0731-q2"}
RUN_ARTIFACTS = (
    "csv", "manifest", "coordinator_log", "coordinator_status", "worker_log",
    "worker_status",
)
PAIR_ALLOWED_BUILD_FIELDS = {
    "source_commit", "ds4_sha256", "peer_ds4_sha256",
    "ds4_bench_tp_sha256",
}
# SDK comparisons are diagnostic until the migration lifecycle, actual loaded
# runtime and per-model numerical/quality admission are connected to the gate.
# An explicit arm identity is narrower than arbitrary manifest allowances.
SDK_IDENTITY_FIELDS = {
    "toolchain_id", "binary_toolchain_sha256", "binary_toolchain_comment",
    "binary_runpath", "expected_binary_toolchain_sha256",
}
SDK_ARM_FIELDS = SDK_IDENTITY_FIELDS | PAIR_ALLOWED_BUILD_FIELDS
SDK_CELLS = {"headline", "diverse", "long-context", "glm-53-q2",
             "deepseek-0731-q4", "deepseek-0731-q2"}
SDK_LIBRARY_PREFIXES = ("libamdhip64.", "libhsa-runtime64.", "libhipblas.",
                        "libhipblaslt.", "librocblas.")
SWITCH_MANIFEST_FIELDS = {
    "glm5_bf16_wmma_hilo": "DS4_ROCM_GLM5_BF16_WMMA_HILO",
    "glm5_bf16_wmma_qkv_fused": "DS4_ROCM_GLM5_BF16_WMMA_QKV_FUSED",
    "glm5_bf16_qkv_decode_multiptr":
        "DS4_ROCM_GLM5_BF16_QKV_DECODE_MULTIPTR",
    "glm5_bf16_qkv_shared_a_prefill":
        "DS4_ROCM_GLM5_BF16_QKV_SHARED_A_PREFILL",
    "glm5_bf16_kda_six_multiptr":
        "DS4_ROCM_GLM5_BF16_KDA_SIX_MULTIPTR",
    "glm5_bf16_kda_six_prefill": "DS4_ROCM_GLM5_BF16_KDA_SIX_PREFILL",
}
ENV_FIELDS = ("common_env", "worker_env", "coordinator_env", "extra_env")
RANK_ENV_FIELDS = ("worker_env", "coordinator_env")
GLM_IDENTITY_FIELDS = (
    "model", "model_arch", "model_size", "model_sample_sha256", "toolchain_id",
    "tp_weight_layout", "tp_intermediate_size", "tp_intermediate_shards",
    "tp_expert_count", "tp_experts_used", "tp_reduce_op", "tp_reduce_scope",
    "tp_reduce_count", "tp_reduce_width", "tp_reduce_dtype",
)


def parse_env(encoded: str, label: str) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        assignments = shlex.split(encoded)
    except ValueError as error:
        raise ProofError(f"{label} has malformed shell quoting") from error
    for assignment in assignments:
        name, separator, value = assignment.partition("=")
        if (not separator or
                not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or
                name in values):
            raise ProofError(f"{label} has a duplicate or invalid assignment")
        values[name] = value
    return values


def validate_switches(value: object) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        raise ProofError("candidate_switches must be an object")
    result: dict[str, dict[str, str]] = {}
    for name, arms in value.items():
        if (not isinstance(name, str) or
                not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or
                not isinstance(arms, dict) or set(arms) != {"control", "candidate"} or
                any(not isinstance(item, str) or "\n" in item
                    for item in arms.values()) or
                arms["control"] == arms["candidate"]):
            raise ProofError("candidate_switches has an invalid arm declaration")
        result[name] = dict(arms)
    return result


def validate_sdk_contrast(value: object) -> dict:
    if (not isinstance(value, dict) or
            set(value) != {"kind", "control", "candidate", "cell_switches", "runtimes"} or
            value.get("kind") != "sdk-contrast-v1"):
        raise ProofError("SDK contrast has invalid fields")
    for arm in ("control", "candidate"):
        identity = value[arm]
        if (not isinstance(identity, dict) or set(identity) != SDK_ARM_FIELDS or
                any(not isinstance(item, str) or not item or "\n" in item
                    for item in identity.values())):
            raise ProofError(f"SDK {arm} identity is incomplete")
        for field in SDK_ARM_FIELDS:
            if field.endswith("sha256") and not re.fullmatch(
                    r"[0-9a-f]{64}", identity[field]):
                raise ProofError(f"SDK {arm} has invalid {field}")
        if (not re.fullmatch(r"[0-9a-f]{40}", identity["source_commit"]) or
                identity["ds4_sha256"] != identity["peer_ds4_sha256"] or
                identity["binary_toolchain_sha256"] !=
                    identity["expected_binary_toolchain_sha256"] or
                identity["toolchain_id"] !=
                    "elf-comment-sha256:" + identity["binary_toolchain_sha256"]):
            raise ProofError(f"SDK {arm} source/build identity is inconsistent")
    if value["control"]["toolchain_id"] == value["candidate"]["toolchain_id"]:
        raise ProofError("SDK contrast requires distinct compiler identities")
    for field in ("ds4_sha256", "ds4_bench_tp_sha256"):
        if value["control"][field] == value["candidate"][field]:
            raise ProofError("SDK contrast requires distinct inference and benchmark builds")
    runtimes = value["runtimes"]
    if not isinstance(runtimes, dict) or set(runtimes) != {"control", "candidate"}:
        raise ProofError("SDK contrast requires both runtime identities")
    for runtime in runtimes.values():
        if (not isinstance(runtime, dict) or set(runtime) != {"sdk_root", "kernel", "libraries"} or
                not isinstance(runtime["sdk_root"], str) or
                not Path(runtime["sdk_root"]).is_absolute() or
                not isinstance(runtime["kernel"], str) or not runtime["kernel"] or
                not isinstance(runtime["libraries"], dict)):
            raise ProofError("SDK runtime identity is incomplete")
        libraries = runtime["libraries"]
        if len(libraries) != len(SDK_LIBRARY_PREFIXES):
            raise ProofError("SDK runtime requires the complete HIP/HSA/BLAS library set")
        for prefix in SDK_LIBRARY_PREFIXES:
            if sum(Path(path).name.startswith(prefix) for path in libraries) != 1:
                raise ProofError("SDK runtime has missing or duplicate library families")
        for path, digest in libraries.items():
            if (not Path(path).is_absolute() or
                    ".." in Path(path).parts or
                    not Path(path).is_relative_to(runtime["sdk_root"]) or
                    not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                raise ProofError("SDK runtime has an invalid library path/hash")
    if runtimes["control"]["kernel"] != runtimes["candidate"]["kernel"]:
        raise ProofError("SDK comparison must retain one kernel driver release")
    cells = value["cell_switches"]
    if not isinstance(cells, dict) or set(cells) != SDK_CELLS:
        raise ProofError("SDK contrast requires all model/context cell bindings")
    for switches in cells.values():
        validate_switches(switches)
    if any(cells[name] != cells["headline"] for name in ("diverse", "long-context")):
        raise ProofError("SDK GLM Q4 screens must use the headline recipe")
    return value


def verify_sdk_arm(manifest: dict, arm: str, contrast: dict) -> None:
    for field, expected in contrast[arm].items():
        if manifest.get(field) != expected:
            raise ProofError(f"{arm} run differs from frozen SDK identity in {field}")
    if manifest.get("dspark") != "0":
        raise ProofError("SDK migration comparison requires ordinary inference")
    if manifest.get("rocprof_binary", "") or manifest.get("rocprof_sha256", ""):
        raise ProofError("SDK timing comparison must be uninstrumented")
    for field in ENV_FIELDS:
        environment = parse_env(manifest.get(field, ""), f"SDK {arm} {field}")
        if ((field in RANK_ENV_FIELDS and
             environment.get("DS4_GLM5_NATIVE_DRAFT") != "0") or
                environment.get("DS4_GLM5_NATIVE_DRAFT", "0") != "0"):
            raise ProofError("SDK migration comparison requires explicit native MTP off")
        if any(name.startswith("DS4_") and name.endswith("_PROFILE")
               for name in environment):
            raise ProofError("SDK timing comparison has a presence-based profiler")


def verify_sdk_runtime(run: dict, value: object, root: Path, arm: str,
                       contrast: dict) -> None:
    path, bound = artifact(value, root, "live SDK runtime")
    capture = json.loads(path.read_text())
    manifest = run["manifest"]
    runtime = contrast["runtimes"][arm]
    if (not isinstance(capture, dict) or not manifest.get("tag") or
            capture.get("tag") != manifest["tag"] or
            capture.get("source") != manifest.get("source_commit") or
            not isinstance(capture.get("ranks"), dict) or
            set(capture["ranks"]) != set(("coordinator", "worker"))):
        raise ProofError("SDK runtime capture differs from its benchmark identity")
    manifest_bytes = Path(run["artifacts"]["manifest"]["path"]).read_bytes()
    capture_hash = capture.get("manifest_sha256")
    if capture_hash != hashlib.sha256(manifest_bytes).hexdigest():
        # The launcher appends this one field after generation. A loading-time
        # capture must bind every other original byte of the final manifest.
        stripped, count = re.subn(
            rb"(?m)^dump_generated_token_sha256=[0-9a-f]{64}\n", b"", manifest_bytes)
        if count != 1 or capture_hash != hashlib.sha256(stripped).hexdigest():
            raise ProofError("SDK runtime capture manifest hash does not bind this run")
    for rank, record in capture["ranks"].items():
        binary_field = "ds4_bench_tp_sha256" if rank == "coordinator" else "peer_ds4_sha256"
        expected_env = parse_env(manifest[rank + "_env"], "runtime rank environment")
        if (not isinstance(record, dict) or
                record.get("executable_sha256") != manifest[binary_field] or
                record.get("kernel") != runtime["kernel"] or
                record.get("effective_environment") != expected_env or
                record.get("runtime_environment") != expected_env or
                expected_env.get("DS4_BENCH_RUN_ID") != manifest.get("run_id") or
                type(record.get("pid")) is not int or record["pid"] <= 0 or
                not str(record.get("start_ticks", "")).isdigit() or
                type(record.get("observed_unix_ns")) is not int or
                record["observed_unix_ns"] <= 0):
            raise ProofError("SDK runtime rank process/settings binding is invalid")
        libraries = record.get("libraries")
        if (not isinstance(libraries, list) or len(libraries) != len(SDK_LIBRARY_PREFIXES) or
                any(not isinstance(item, dict) or set(item) != {"path", "resolved", "sha256"}
                    for item in libraries)):
            raise ProofError("SDK runtime capture has incomplete mapped libraries")
        if ({item["resolved"]: item["sha256"] for item in libraries} != runtime["libraries"] or
                any(item["path"] != item["resolved"] for item in libraries)):
            raise ProofError("SDK runtime library hashes differ from the frozen arm")
        mappings = record.get("mappings")
        if (not isinstance(mappings, list) or
                any(not isinstance(line, str) or len(line.split(None, 5)) != 6 for line in mappings) or
                {line.split(None, 5)[5] for line in mappings} != set(runtime["libraries"])):
            raise ProofError("SDK runtime mapping evidence is incomplete")
    run["artifacts"]["runtime"] = bound


def validate_performance_policy(value: object) -> dict:
    required = {"contract", "target_metrics", "candidate_switches", "public_claim"}
    if (not isinstance(value, dict) or
            set(value) not in (required, required | {"sdk_contrast"})):
        raise ProofError("performance policy has invalid fields")
    migration = "sdk_contrast" in value
    if migration:
        validate_sdk_contrast(value["sdk_contrast"])
    contract = value["contract"]
    expected = {
        "schema_version", "method", "qualification_pairs", "merge_looks",
        "formal_test", "futility_confidence_level", "minimum_gain",
        "maximum_untargeted_regression", "maximum_control_regression",
        "absolute_floor", "screens", "required_provider",
    }
    if (not isinstance(contract, dict) or set(contract) != expected or
            contract.get("schema_version") != 2 or
            contract.get("method") != "paired-log-ratio-two-tier-v2" or
            contract.get("qualification_pairs") != 3 or
            contract.get("merge_looks") != [5, 7, 9] or
            contract.get("futility_confidence_level") != 0.95 or
            contract.get("required_provider") != "roce-v2"):
        raise ProofError("performance contract has an unsupported sequential design")
    formal_test = contract["formal_test"]
    if not isinstance(formal_test, dict):
        raise ProofError("performance contract formal_test must be an object")
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
            raise ProofError("repeated-Student formal test is not frozen and qualified")
    elif formal_test.get("kind") == "exact-sign-fixed-nine-v1":
        if (set(formal_test) != {
                "kind", "pairs", "minimum_positive", "ties",
                "familywise_alpha"} or
                formal_test.get("pairs") != 9 or
                formal_test.get("minimum_positive") != 8 or
                formal_test.get("ties") != "fail" or
                formal_test.get("familywise_alpha") != 0.05):
            raise ProofError("exact-sign formal test is not the frozen fallback")
    else:
        raise ProofError("performance contract has an unsupported formal test")
    targets = value["target_metrics"]
    if (not isinstance(targets, list) or (not targets and not migration) or
            len(set(targets)) != len(targets) or
            set(targets) - {"prefill", "decode"} or
            type(value["public_claim"]) is not bool):
        raise ProofError("performance target declaration is invalid")
    if migration and (targets or value["public_claim"] or
                      formal_test["kind"] != "exact-sign-fixed-nine-v1"):
        raise ProofError(
            "SDK migration requires guard-only exact-sign timing without a speed claim")
    for name in ("minimum_gain", "maximum_untargeted_regression",
                 "maximum_control_regression"):
        number = contract[name]
        if (not isinstance(number, (int, float)) or
                not math.isfinite(number) or not 0 <= number <= 0.10):
            raise ProofError(f"performance contract {name} is invalid")
    floors = contract["absolute_floor"]
    screens = contract["screens"]
    if (not isinstance(floors, dict) or set(floors) != {"prefill", "decode"} or
            not isinstance(screens, dict) or
            set(screens) != {"max_prefill_regression", "max_decode_regression"}):
        raise ProofError("performance floor or screen contract is invalid")
    for name, number in {**floors, **screens}.items():
        if (not isinstance(number, (int, float)) or
                not math.isfinite(number) or number < 0 or
                (name.startswith("max_") and number > 0.10)):
            raise ProofError(f"performance contract {name} is invalid")
    switches = validate_switches(value["candidate_switches"])
    if migration and switches:
        raise ProofError("SDK migration switches must be bound per cell")
    return {**value, "candidate_switches": switches}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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


def research_root() -> Path:
    raw = os.environ.get("DS4_RESEARCH_ROOT")
    if not raw:
        raise ProofError("DS4_RESEARCH_ROOT is required")
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise ProofError(f"canonical research root is missing: {root}")
    return root


def confined(path: Path, root: Path, label: str) -> Path:
    value = path.expanduser().resolve()
    if value != root and root not in value.parents:
        raise ProofError(f"{label} escapes DS4_RESEARCH_ROOT: {value}")
    return value


def artifact(value: object, root: Path, label: str) -> tuple[Path, dict]:
    if isinstance(value, str):
        path = confined(Path(value), root, label)
        expected = None
    elif isinstance(value, dict) and set(value) == {"path", "sha256"}:
        path = confined(Path(str(value["path"])), root, label)
        expected = str(value["sha256"])
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ProofError(f"{label} has an invalid SHA-256")
    else:
        raise ProofError(f"{label} must be a path or content-addressed artifact")
    if not path.is_file():
        raise ProofError(f"missing {label}: {path}")
    actual = sha256(path)
    if expected is not None and expected != actual:
        raise ProofError(f"{label} hash mismatch: {path}")
    return path, {"path": str(path), "sha256": actual}


def read_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise ProofError(f"{path}: malformed or duplicate manifest key {key!r}")
        values[key] = value
    return values


def committed_source_sha256(repo: Path, commit: str, relative: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "blob", f"{commit}:{relative}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise ProofError(
            f"cannot resolve {relative} from run source commit {commit}")
    return hashlib.sha256(result.stdout).hexdigest()


def read_result(path: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ProofError(f"benchmark CSV must have exactly one row: {path}")
    row = rows[0]
    try:
        result: dict[str, object] = {
            "frontier": int(row["ctx_tokens"]),
            "generated_tokens": int(row["gen_tokens"]),
            "prefill_tps": float(row["prefill_tps"]),
            "decode_tps": float(row["gen_tps"]),
            "steady_decode_tps": float(row["gen_steady_tps"]),
            "fnv64": row["gen_token_fnv64"].lower(),
        }
    except (KeyError, ValueError) as error:
        raise ProofError(f"invalid benchmark CSV {path}: {error}") from error
    for key in ("prefill_tps", "decode_tps", "steady_decode_tps"):
        number = float(result[key])
        if not math.isfinite(number) or number <= 0:
            raise ProofError(f"{path}: {key} must be positive and finite")
    if not re.fullmatch(r"[0-9a-f]{16}", str(result["fnv64"])):
        raise ProofError(f"{path}: invalid token fingerprint")
    return result


def validate_status(path: Path, label: str) -> None:
    values = read_manifest(path)
    if values.get("exit_code") != "0" or values.get("signal") != "0":
        raise ProofError(f"{label} did not exit cleanly: {path}")


def normalize_run(value: object, root: Path, repo: Path, *,
                  lane: str, expected_fnv: str | None) -> dict:
    if not isinstance(value, dict) or set(value) != set(RUN_ARTIFACTS):
        raise ProofError("each run must bind CSV, manifest, both logs, and both statuses")
    paths: dict[str, Path] = {}
    bound: dict[str, dict] = {}
    for name in RUN_ARTIFACTS:
        paths[name], bound[name] = artifact(value[name], root, f"run {name}")
    row = read_result(paths["csv"])
    manifest = read_manifest(paths["manifest"])
    producer_source = manifest.get("ds4_bench_producer_source_sha256", "")
    source_commit = manifest.get("source_commit", "")
    bench_binary = manifest.get("ds4_bench_tp_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ProofError("benchmark source commit identity is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", bench_binary):
        raise ProofError("benchmark executable identity is invalid")
    if (not re.fullmatch(r"[0-9a-f]{64}", producer_source) or
            producer_source != committed_source_sha256(
                repo, source_commit, "ds4_bench.c")):
        raise ProofError(
            "benchmark producer binary does not match its committed ds4_bench.c")
    validate_status(paths["coordinator_status"], "coordinator")
    validate_status(paths["worker_status"], "worker")
    run_id = manifest.get("run_id", "")
    for field in ("common_env", *RANK_ENV_FIELDS):
        if parse_env(manifest.get(field, ""), f"run {field}").get(
                "DS4_BENCH_RUN_ID") != run_id:
            raise ProofError(f"run {field} does not bind its run_id")
    if paths["manifest"] != paths["csv"].with_suffix(".manifest"):
        raise ProofError("run manifest must be adjacent to its CSV")
    if (manifest.get("source_dirty") != "0" or
            manifest.get("candidate") != "1" or
            manifest.get("candidate_lane") != lane or
            manifest.get("dspark") != "0"):
        raise ProofError("promotion run is not a clean ordinary candidate-validation run")
    if (manifest.get("ds4_sha256") != manifest.get("peer_ds4_sha256") or
            not re.fullmatch(r"[0-9a-f]{64}", manifest.get("ds4_sha256", ""))):
        raise ProofError("promotion run used different or unbound rank binaries")
    if (manifest.get("frontier") != str(row["frontier"]) or
            manifest.get("generated_tokens") != str(row["generated_tokens"])):
        raise ProofError("benchmark CSV and manifest workload differ")
    if expected_fnv is not None and row["fnv64"] != expected_fnv:
        raise ProofError(
            f"run fingerprint {row['fnv64']} differs from expected {expected_fnv}")
    profile = manifest.get("rdma_profile")
    if profile not in {"roce-v2", "odinlink"}:
        raise ProofError("promotion run did not use a supported explicit RDMA profile")
    if (profile == "roce-v2" and
            (not re.fullmatch(r"mlx5_[A-Za-z0-9_.-]+",
                              manifest.get("coordinator_rdma_device", "")) or
             not re.fullmatch(r"mlx5_[A-Za-z0-9_.-]+",
                              manifest.get("worker_rdma_device", "")) or
             not re.fullmatch(r"[0-9]+", manifest.get("rdma_gid_index", "")))):
        raise ProofError("RoCE v2 run has incomplete device/GID identity")
    command = [
        str(repo / "scripts" / "check-ds4-bench-result.sh"),
        str(paths["csv"]), str(paths["coordinator_log"]),
        str(paths["worker_log"]), expected_fnv or "",
        str(row["generated_tokens"]), "1", profile,
        manifest.get("coordinator_rdma_device", ""),
        manifest.get("rdma_gid_index", ""),
        manifest.get("worker_rdma_device", ""),
        manifest.get("run_id", ""),
    ]
    checked = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    if checked.returncode != 0:
        detail = checked.stderr.strip() or checked.stdout.strip()
        raise ProofError(f"run transport/cache/semantic validation failed: {detail}")
    return {"artifacts": bound, "manifest": manifest, "result": row}


def compare_manifests(repo: Path, control: Path, candidate: Path,
                      allowed_fields: list[str], allowed_env: list[str]) -> None:
    command = [sys.executable, str(repo / "scripts" / "compare-bench-manifests.py"),
               str(control), str(candidate), "--allow-field", "pair_arm"]
    for name in allowed_fields:
        command.extend(["--allow-field", name])
    for name in allowed_env:
        command.extend(["--allow-env", name])
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ProofError(f"paired manifests differ outside registered switches: {detail}")


def normalize_pair(value: object, root: Path, repo: Path, *, lane: str,
                   control_fnv: str, source_commits: dict[str, str],
                   candidate_switches: dict[str, dict[str, str]],
                   expected_candidate_id: str | None = None,
                   sdk_contrast: dict | None = None) -> dict:
    required = {"pair_id", "order", "control", "candidate",
                "allowed_fields", "allowed_env"}
    if not isinstance(value, dict) or set(value) != required:
        raise ProofError("timing pair has invalid fields")
    pair_id = str(value["pair_id"])
    order = str(value["order"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", pair_id):
        raise ProofError("timing pair has an invalid pair_id")
    if order not in {"AB", "BA"}:
        raise ProofError("timing pair order must be AB or BA")
    allowed_fields = value["allowed_fields"]
    allowed_env = value["allowed_env"]
    if (not isinstance(allowed_fields, list) or
            not isinstance(allowed_env, list) or
            any(not isinstance(item, str) or not item for item in
                [*allowed_fields, *allowed_env])):
        raise ProofError("paired manifest allowances must be string lists")
    if len(set(allowed_fields)) != len(allowed_fields) or len(set(allowed_env)) != len(allowed_env):
        raise ProofError("paired manifest allowances must not contain duplicates")
    switch_fields = {
        field for field, switch in SWITCH_MANIFEST_FIELDS.items()
        if switch in candidate_switches
    }
    permitted_fields = PAIR_ALLOWED_BUILD_FIELDS | switch_fields
    if sdk_contrast is not None:
        validate_sdk_contrast(sdk_contrast)
        permitted_fields |= SDK_IDENTITY_FIELDS
    unexpected = set(allowed_fields) - permitted_fields
    if unexpected:
        raise ProofError(
            "paired manifest field is not an approved build or declared-switch "
            "field: " + ", ".join(sorted(unexpected)))
    if set(allowed_env) != set(candidate_switches):
        raise ProofError(
            "paired environment allowances must exactly match initialized switches")
    arms = {arm: value[arm] for arm in ("control", "candidate")}
    if sdk_contrast is not None:
        for arm, run in arms.items():
            if not isinstance(run, dict) or set(run) != set(RUN_ARTIFACTS) | {"runtime"}:
                raise ProofError("SDK run requires live runtime evidence from both ranks")
        arms = {arm: {key: run[key] for key in RUN_ARTIFACTS} for arm, run in arms.items()}
    control = normalize_run(
        arms["control"], root, repo, lane=lane, expected_fnv=control_fnv)
    candidate = normalize_run(
        arms["candidate"], root, repo, lane=lane,
        expected_fnv=control_fnv if lane == "A" else None)
    for arm_name, run in (("control", control), ("candidate", candidate)):
        manifest = run["manifest"]
        if sdk_contrast is not None:
            verify_sdk_arm(manifest, arm_name, sdk_contrast)
            verify_sdk_runtime(run, value[arm_name]["runtime"], root, arm_name, sdk_contrast)
        if (expected_candidate_id is not None and
                manifest.get("candidate_id") != expected_candidate_id):
            raise ProofError(
                "headline run manifest does not match the proof candidate_id")
        if (manifest.get("pair_id") != pair_id or
                manifest.get("pair_order") != order or
                manifest.get("pair_arm") != arm_name):
            raise ProofError("run manifest does not match its registered pair identity")
        if manifest.get("source_commit") != source_commits[arm_name]:
            raise ProofError(
                f"{arm_name} run does not match its registered source commit")
        for field in ENV_FIELDS:
            if field not in manifest:
                raise ProofError(f"{arm_name} run is missing {field}")
        extra = parse_env(manifest["extra_env"], f"{arm_name} extra_env")
        for switch, arms in candidate_switches.items():
            expected = arms[arm_name]
            if extra.get(switch) != expected:
                raise ProofError(
                    f"{arm_name} extra_env does not match initialized {switch}")
            for field in RANK_ENV_FIELDS:
                effective = parse_env(
                    manifest[field], f"{arm_name} {field}")
                if effective.get(switch) != expected:
                    raise ProofError(
                        f"{arm_name} {field} does not match initialized {switch}")
        for field, switch in SWITCH_MANIFEST_FIELDS.items():
            if field in allowed_fields and manifest.get(field) != \
                    candidate_switches[switch][arm_name]:
                raise ProofError(
                    f"{arm_name} manifest mirror {field} does not match {switch}")
    control_id = control["manifest"].get("run_id", "")
    candidate_run_id = candidate["manifest"].get("run_id", "")
    if not control_id or not candidate_run_id or control_id == candidate_run_id:
        raise ProofError("paired run IDs must be distinct")
    if ((order == "AB" and not control_id < candidate_run_id) or
            (order == "BA" and not candidate_run_id < control_id)):
        raise ProofError("run IDs contradict the registered AB/BA order")
    compare_manifests(
        repo,
        Path(control["artifacts"]["manifest"]["path"]),
        Path(candidate["artifacts"]["manifest"]["path"]),
        allowed_fields, allowed_env)
    return {
        "pair_id": pair_id,
        "order": order,
        "allowed_fields": allowed_fields,
        "allowed_env": allowed_env,
        "control": control,
        "candidate": candidate,
    }


def normalize_trajectory(value: object, root: Path, repo: Path,
                         control: dict, candidate: dict,
                         candidate_switches: dict[str, dict[str, str]]) -> dict:
    if not isinstance(value, dict) or value.get("mode") not in {"exact", "teacher"}:
        raise ProofError("screen trajectory mode must be exact or teacher")
    if value["mode"] == "exact":
        if set(value) != {"mode"}:
            raise ProofError("exact trajectory has unexpected fields")
        if control["result"]["fnv64"] != candidate["result"]["fnv64"]:
            raise ProofError("screen exact trajectory fingerprint changed")
        return {"mode": "exact", "fnv64": control["result"]["fnv64"]}
    if set(value) != {"mode", "summary", "thresholds"}:
        raise ProofError("teacher trajectory has invalid fields")
    summary_path, summary_ref = artifact(value["summary"], root, "teacher summary")
    threshold_path, threshold_ref = artifact(value["thresholds"], root, "teacher thresholds")
    try:
        recorded = json.loads(summary_path.read_text(encoding="utf-8"))
        sources = recorded["sources"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ProofError(f"invalid teacher summary: {error}") from error
    command = [
        sys.executable, str(repo / "scripts" / "compare-teacher-logits.py"),
        str(sources["reference_dir"]), str(sources["candidate_dir"]),
        "--thresholds", str(threshold_path),
    ]
    if recorded.get("allow_quality_difference") is True:
        command.append("--allow-quality-difference")
    rerun = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    if rerun.returncode != 0:
        detail = rerun.stderr.strip() or rerun.stdout.strip()
        raise ProofError(f"teacher trajectory failed recomputation: {detail}")
    recomputed = json.loads(rerun.stdout)
    ignored = {"thresholds_sha256"}
    if ({key: item for key, item in recorded.items() if key not in ignored} !=
            {key: item for key, item in recomputed.items() if key not in ignored}):
        raise ProofError("teacher trajectory summary differs from raw logits")
    frontier = int(control["result"]["frontier"])
    for role, arm_name, run in (("reference_manifest", "control", control),
                                ("candidate_manifest", "candidate", candidate)):
        manifest = read_manifest(confined(Path(sources[role]), root, role))
        if manifest.get("prefix_tokens") != str(frontier):
            raise ProofError("teacher trajectory prefix differs from screen frontier")
        run_manifest = run["manifest"]
        for key in ("source_commit", "model", "model_size",
                    "model_sample_sha256", "prompt_sha256", "toolchain_id",
                    "rdma_profile", "dspark", "ds4_sha256"):
            if manifest.get(key) != run_manifest.get(key):
                raise ProofError(
                    f"teacher trajectory differs from its screen run in {key}")
        for field in (*RANK_ENV_FIELDS, "extra_env"):
            if field not in manifest:
                raise ProofError(f"teacher trajectory manifest is missing {field}")
            effective = parse_env(manifest[field], f"teacher {role} {field}")
            for switch, arms in candidate_switches.items():
                if effective.get(switch) != arms[arm_name]:
                    raise ProofError(
                        f"teacher trajectory {role} differs in initialized {switch}")
    reference_manifest = read_manifest(confined(
        Path(sources["reference_manifest"]), root, "reference_manifest"))
    candidate_manifest = read_manifest(confined(
        Path(sources["candidate_manifest"]), root, "candidate_manifest"))
    if (not reference_manifest.get("frozen_token_sha256") or
            reference_manifest.get("frozen_token_sha256") !=
            candidate_manifest.get("frozen_token_sha256")):
        raise ProofError("teacher trajectory does not share one frozen token stream")
    return {"mode": "teacher", "summary": summary_ref,
            "thresholds": threshold_ref}


def normalize_screen(name: str, value: object, root: Path, repo: Path, *,
                     lane: str, source_commits: dict[str, str],
                     candidate_switches: dict[str, dict[str, str]],
                     screen_limits: dict[str, float],
                     frontier: int | None = None,
                     minimum_frontier: int | None = None,
                     prompt_sha256: str | None = None,
                     sdk_contrast: dict | None = None) -> dict:
    required = {"pair", "trajectory", "control_fnv64", "max_prefill_regression",
                "max_decode_regression"}
    if not isinstance(value, dict) or set(value) != required:
        raise ProofError(f"{name} screen has invalid fields")
    if (sdk_contrast is not None and
            (not isinstance(value.get("trajectory"), dict) or
             value["trajectory"].get("mode") != "exact")):
        raise ProofError("SDK screen is missing model-specific numerical/quality admission")
    control_fnv = str(value["control_fnv64"]).lower()
    if not re.fullmatch(r"[0-9a-f]{16}", control_fnv):
        raise ProofError(f"{name} screen has an invalid control fingerprint")
    pair = normalize_pair(value["pair"], root, repo, lane=lane,
                          control_fnv=control_fnv,
                          source_commits=source_commits,
                          candidate_switches=candidate_switches,
                          sdk_contrast=sdk_contrast)
    control = pair["control"]
    candidate = pair["candidate"]
    actual_frontier = int(control["result"]["frontier"])
    if frontier is not None and actual_frontier != frontier:
        raise ProofError(f"{name} screen must use frontier {frontier}")
    if minimum_frontier is not None and actual_frontier < minimum_frontier:
        raise ProofError(f"{name} screen must use at least {minimum_frontier} tokens")
    if int(candidate["result"]["frontier"]) != actual_frontier:
        raise ProofError(f"{name} screen frontiers differ")
    if (control["result"]["generated_tokens"] != 300 or
            candidate["result"]["generated_tokens"] != 300):
        raise ProofError(f"{name} screen must generate 300 tokens")
    if prompt_sha256 is not None and (
            control["manifest"].get("prompt_sha256") != prompt_sha256 or
            candidate["manifest"].get("prompt_sha256") != prompt_sha256):
        raise ProofError(f"{name} screen used the wrong frozen prompt")
    limits = {}
    for metric in ("prefill", "decode"):
        key = f"max_{metric}_regression"
        limit = value[key]
        if limit != screen_limits[key]:
            raise ProofError(f"{name}.{key} differs from the frozen contract")
        control_metric = float(control["result"][f"{metric}_tps"])
        candidate_metric = float(candidate["result"][f"{metric}_tps"])
        change = candidate_metric / control_metric - 1.0
        limits[metric] = {"change": change, "maximum_regression": limit,
                          "passed": change >= -limit}
    trajectory = normalize_trajectory(
        value["trajectory"], root, repo, control, candidate,
        candidate_switches)
    if not all(item["passed"] for item in limits.values()):
        raise ProofError(f"{name} screen exceeded its registered regression margin")
    return {"control_fnv64": control_fnv, "pair": pair,
            "trajectory": trajectory, "metrics": limits}


def performance_decision(pairs: list[dict], policy: dict) -> dict:
    policy = validate_performance_policy(policy)
    contract = policy["contract"]
    targets = policy["target_metrics"]
    minimum_gain = contract["minimum_gain"]
    max_regression = contract["maximum_untargeted_regression"]
    floors = contract["absolute_floor"]
    public_claim = policy["public_claim"]
    formal_test = contract["formal_test"]
    count = len(pairs)
    if count not in {contract["qualification_pairs"], *contract["merge_looks"]}:
        raise ProofError("headline performance requires 3, 5, 7, or 9 matched pairs")
    if public_claim and count < 5:
        raise ProofError("README/headline claims require at least five matched pairs")
    orders = [item["order"] for item in pairs]
    if (set(orders) != {"AB", "BA"} or
            abs(orders.count("AB") - orders.count("BA")) > 1 or
            any(left == right for left, right in zip(orders, orders[1:]))):
        raise ProofError("headline timing pairs must strictly alternate AB/BA")
    result = {}
    passed = True
    qualification = count == contract["qualification_pairs"]
    for metric in ("prefill", "decode"):
        controls = [float(item["control"]["result"][f"{metric}_tps"])
                    for item in pairs]
        candidates = [float(item["candidate"]["result"][f"{metric}_tps"])
                      for item in pairs]
        try:
            diagnostic = paired_log_ratio_interval(
                controls, candidates,
                confidence_level=contract["futility_confidence_level"])
            interval = {
                key: diagnostic[key] for key in (
                    "pairs", "geometric_mean_change", "median_change",
                    "log_standard_error")
            }
            interval["diagnostic_fixed_sample_t95"] = {
                key: diagnostic[key] for key in (
                    "confidence_level", "one_sided_lower", "one_sided_upper")
            }
            if not qualification and formal_test["kind"] == "repeated-student-v1":
                interval = paired_log_ratio_repeated_interval(
                    controls, candidates, boundary=formal_test["boundary"])
        except StatsError as error:
            raise ProofError(str(error)) from error
        required = minimum_gain if metric in targets else -max_regression
        changes = [candidate / control - 1.0
                   for control, candidate in zip(controls, candidates)]
        interval["required_lower_bound"] = required
        interval["paired_changes"] = changes
        interval["geometric_mean_control_tps"] = math.exp(
            sum(math.log(item) for item in controls) / len(controls))
        interval["geometric_mean_candidate_tps"] = math.exp(
            sum(math.log(item) for item in candidates) / len(candidates))
        interval["mean_candidate_tps"] = sum(candidates) / len(candidates)
        interval["absolute_floor"] = floors[metric]
        interval["formal_test"] = None if qualification else formal_test["kind"]
        interval["qualification_consistent_direction"] = bool(
            all(strictly_exceeds(change, required) for change in changes))
        if qualification:
            endpoint_passed = interval["qualification_consistent_direction"]
        elif formal_test["kind"] == "repeated-student-v1":
            endpoint_passed = strictly_exceeds(
                interval["one_sided_lower"], required)
        elif count == formal_test["pairs"]:
            sign_result = exact_sign_test(changes, required)
            interval["sign_test"] = sign_result
            endpoint_passed = (
                sign_result["positive_adjusted_effects"] >=
                formal_test["minimum_positive"])
        else:
            endpoint_passed = False
        interval["passed"] = bool(
            endpoint_passed and interval["mean_candidate_tps"] >= floors[metric])
        interval["futility_diagnostic"] = bool(
            diagnostic["one_sided_upper"] < required)
        passed = passed and interval["passed"]
        result[metric] = interval
    if passed and qualification:
        decision = "consistent-direction"
    elif passed:
        decision = "pass"
    elif count < 9:
        decision = "needs-two-more-pairs"
    else:
        decision = "not-demonstrated"
    return {
        "pairs": count,
        "order_counts": {item: orders.count(item) for item in ("AB", "BA")},
        "method": contract["method"],
        "formal_test": formal_test,
        "familywise_confidence_level": None if qualification else 0.95,
        "metrics": result,
        "decision": decision,
        "merge_eligible": bool(passed and not qualification and
                               "sdk_contrast" not in policy),
        "passed": passed,
    }


def calculate(spec: object, root: Path, repo: Path) -> dict:
    required = {"schema_version", "kind", "stage", "candidate_id", "lane",
                "baseline_id", "baseline_fnv64", "source_commits",
                "required_provider", "performance", "headline_pairs",
                "diverse_screen", "long_context_screen",
                "ordinary_regressions"}
    if not isinstance(spec, dict) or set(spec) != required:
        raise ProofError("promotion-proof spec has invalid fields")
    if spec.get("schema_version") != 2 or spec.get("kind") != "ds4-promotion-proof-spec":
        raise ProofError("unsupported promotion-proof spec")
    candidate_id = str(spec["candidate_id"])
    lane = str(spec["lane"])
    stage = str(spec["stage"])
    baseline_fnv = str(spec["baseline_fnv64"]).lower()
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", candidate_id) or
            lane not in {"A", "B", "C"} or
            stage not in {"qualification", "promotion", "sdk-diagnostic"} or
            not re.fullmatch(r"sha256:[0-9a-f]{64}", str(spec["baseline_id"])) or
            not re.fullmatch(r"[0-9a-f]{16}", baseline_fnv) or
            spec["required_provider"] not in {"roce-v2", "odinlink"}):
        raise ProofError("promotion-proof identity is invalid")
    source_commits = spec["source_commits"]
    if (not isinstance(source_commits, dict) or
            set(source_commits) != {"control", "candidate"} or
            any(not re.fullmatch(r"[0-9a-f]{40}", str(value))
                for value in source_commits.values())):
        raise ProofError("promotion-proof source commits are invalid")
    for role, commit in source_commits.items():
        available = subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if available.returncode != 0:
            raise ProofError(f"{role} source commit is unavailable in the repository")
    producer_sources = {
        role: committed_source_sha256(repo, commit, "ds4_bench.c")
        for role, commit in source_commits.items()
    }
    if len(set(producer_sources.values())) != 1:
        raise ProofError(
            "paired control and candidate benchmark producer source differs")
    performance_policy = validate_performance_policy(spec["performance"])
    sdk_contrast = performance_policy.get("sdk_contrast")
    if (sdk_contrast is not None) != (stage == "sdk-diagnostic"):
        raise ProofError(
            "SDK contrast is diagnostic-only pending migration admission; "
            "it cannot qualify or promote a candidate")
    if spec["required_provider"] != performance_policy["contract"]["required_provider"]:
        raise ProofError("proof provider differs from the frozen performance contract")
    formal_test = performance_policy["contract"]["formal_test"]
    if formal_test["kind"] == "repeated-student-v1":
        calibration_path = repo / "scripts" / \
            "promotion-boundary-repeated-student-v1.json"
        if (not calibration_path.is_file() or
                sha256(calibration_path) != REPEATED_STUDENT_CALIBRATION_SHA256):
            raise ProofError("versioned repeated-Student calibration changed")
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if (calibration.get("passed") is not True or
                calibration.get("boundary") != formal_test["boundary"] or
                calibration.get("familywise_alpha") !=
                    formal_test["familywise_alpha"] or
                calibration.get("design", {}).get("looks") !=
                    performance_policy["contract"]["merge_looks"]):
            raise ProofError("repeated-Student calibration is internally inconsistent")
    candidate_switches = (performance_policy["candidate_switches"]
                          if sdk_contrast is None else
                          sdk_contrast["cell_switches"]["headline"])
    screen_limits = performance_policy["contract"]["screens"]
    long_prompt = repo / LONG_CONTEXT_PROMPT
    if not long_prompt.is_file() or sha256(long_prompt) != LONG_CONTEXT_PROMPT_SHA256:
        raise ProofError("versioned long-context prompt identity changed")
    raw_pairs = spec["headline_pairs"]
    if not isinstance(raw_pairs, list):
        raise ProofError("headline_pairs must be a list")
    pairs = [normalize_pair(item, root, repo, lane=lane,
                            control_fnv=baseline_fnv,
                            source_commits=source_commits,
                            candidate_switches=candidate_switches,
                            expected_candidate_id=candidate_id,
                            sdk_contrast=sdk_contrast)
             for item in raw_pairs]
    pair_ids = [item["pair_id"] for item in pairs]
    if len(set(pair_ids)) != len(pair_ids):
        raise ProofError("headline pair IDs are not distinct")
    for pair in pairs:
        for run in (pair["control"], pair["candidate"]):
            manifest = run["manifest"]
            if (manifest.get("baseline_id") != spec["baseline_id"] or
                    manifest.get("rdma_profile") != spec["required_provider"]):
                raise ProofError("headline run differs from proof source/baseline/provider")
    candidate_fingerprints = {
        item["candidate"]["result"]["fnv64"] for item in pairs}
    if len(candidate_fingerprints) != 1:
        raise ProofError("headline candidate runs do not have one deterministic fingerprint")
    performance = performance_decision(pairs, performance_policy)
    if ((stage == "qualification" and len(pairs) != 3) or
            (stage in {"promotion", "sdk-diagnostic"} and len(pairs) not in {5, 7, 9})):
        raise ProofError(
            "qualification requires three pairs; promotion requires 5, 7, or 9")
    diverse = normalize_screen(
        "diverse", spec["diverse_screen"], root, repo, lane=lane,
        source_commits=source_commits,
        candidate_switches=candidate_switches, screen_limits=screen_limits,
        frontier=4096,
        prompt_sha256=DIVERSE_PROMPT_SHA256, sdk_contrast=sdk_contrast)
    if stage == "qualification":
        if spec["long_context_screen"] is not None or spec["ordinary_regressions"] != []:
            raise ProofError(
                "qualification proof must defer long-context and DeepSeek screens")
        long_context = None
    else:
        long_context = normalize_screen(
            "long-context", spec["long_context_screen"], root, repo, lane=lane,
            source_commits=source_commits,
            candidate_switches=candidate_switches, screen_limits=screen_limits,
            minimum_frontier=8192,
            prompt_sha256=LONG_CONTEXT_PROMPT_SHA256, sdk_contrast=sdk_contrast)
    named_glm_screens = [("diverse", diverse)]
    if long_context is not None:
        named_glm_screens.append(("long-context", long_context))
    for screen_name, screen in named_glm_screens:
        for run in (screen["pair"]["control"], screen["pair"]["candidate"]):
            manifest = run["manifest"]
            if (manifest.get("baseline_id") != spec["baseline_id"] or
                    manifest.get("rdma_profile") != spec["required_provider"]):
                raise ProofError(
                    f"{screen_name} run differs from proof source/baseline/provider")
    glm_runs = [run for pair in pairs for run in (pair["control"], pair["candidate"])]
    for _, screen in named_glm_screens:
        glm_runs.extend((screen["pair"]["control"], screen["pair"]["candidate"]))
    identity_fields = tuple(key for key in GLM_IDENTITY_FIELDS
                            if sdk_contrast is None or key != "toolchain_id")
    reference_identity = {
        key: glm_runs[0]["manifest"].get(key) for key in identity_fields
    }
    for required_field in ("model", "model_size", "model_sample_sha256", "toolchain_id"):
        if not glm_runs[0]["manifest"].get(required_field):
            raise ProofError(f"GLM promotion run is missing {required_field}")
    for run in glm_runs[1:]:
        identity = {key: run["manifest"].get(key) for key in identity_fields}
        if identity != reference_identity:
            raise ProofError("4K/8K/headline GLM runs differ in model, toolchain, or layout")
    regressions = spec["ordinary_regressions"]
    required_regressions = (REQUIRED_REGRESSIONS if sdk_contrast is None else
                            REQUIRED_REGRESSIONS | {"glm-53-q2"})
    if (not isinstance(regressions, list) or
            (stage in {"promotion", "sdk-diagnostic"} and
             {str(item.get("name")) for item in regressions
              if isinstance(item, dict)} != required_regressions) or
            len(regressions) != (0 if stage == "qualification" else len(required_regressions))):
        raise ProofError("promotion regressions must cover each required model exactly once")
    normalized_regressions = []
    regression_model_hashes = set()
    for item in regressions:
        if set(item) != {"name", "baseline_fnv64", "model", "screen"}:
            raise ProofError("ordinary regression has invalid fields")
        regression_fnv = str(item["baseline_fnv64"]).lower()
        model = item["model"]
        if (not re.fullmatch(r"[0-9a-f]{16}", regression_fnv) or
                not isinstance(model, dict) or
                set(model) != {"path", "size", "sample_sha256", "sha256",
                              "quantization"} or
                not isinstance(model["path"], str) or
                not Path(model["path"]).is_absolute() or
                not isinstance(model["size"], int) or model["size"] <= 0 or
                not re.fullmatch(r"[0-9a-f]{64}", str(model["sample_sha256"])) or
                not re.fullmatch(r"[0-9a-f]{64}", str(model["sha256"])) or
                model["quantization"] not in {"Q4_K", "Q2_K"}):
            raise ProofError("ordinary regression model/fingerprint identity is invalid")
        model_path = Path(model["path"]).resolve()
        if sdk_contrast is not None:
            expected_quant = "Q4_K" if item["name"] == "deepseek-0731-q4" else "Q2_K"
            if model["quantization"] != expected_quant or model["sha256"] in regression_model_hashes:
                raise ProofError("SDK regression cells require distinct models and the named quantization")
            regression_model_hashes.add(model["sha256"])
        try:
            actual_size, actual_sample = sampled_model_sha256(model_path)
        except OSError as error:
            raise ProofError(f"cannot verify ordinary regression model: {error}") from error
        if (str(model_path) != model["path"] or actual_size != model["size"] or
                actual_sample != model["sample_sha256"] or
                sha256(model_path) != model["sha256"]):
            raise ProofError("ordinary regression model bytes differ from the proof")
        normalized = normalize_screen(
            str(item["name"]), item["screen"], root, repo, lane=lane,
            source_commits=source_commits,
            candidate_switches=(candidate_switches if sdk_contrast is None else
                                sdk_contrast["cell_switches"][item["name"]]),
            screen_limits=screen_limits,
            frontier=2048, prompt_sha256=DEEPSEEK_PROMPT_SHA256,
            sdk_contrast=sdk_contrast)
        if normalized["control_fnv64"] != regression_fnv:
            raise ProofError("ordinary regression screen fingerprint is inconsistent")
        for run in (normalized["pair"]["control"],
                    normalized["pair"]["candidate"]):
            manifest = run["manifest"]
            if sdk_contrast is not None:
                expected_arch = "glm5-next" if item["name"] == "glm-53-q2" else "deepseek4"
                if (reference_identity.get("model_arch") != "glm5-next" or
                        manifest.get("model_arch") != expected_arch):
                    raise ProofError("SDK regression cell has the wrong model architecture")
            if (manifest.get("rdma_profile") != spec["required_provider"] or
                    manifest.get("model") != model["path"] or
                    manifest.get("model_size") != str(model["size"]) or
                    manifest.get("model_sample_sha256") != model["sample_sha256"]):
                raise ProofError(
                    "ordinary regression run differs from proof source/provider/model")
        normalized_regressions.append({
            "name": item["name"], "baseline_fnv64": regression_fnv,
            "model": model, **normalized})
    all_runs = [run for pair in pairs for run in (pair["control"], pair["candidate"])]
    final_screens = [diverse, *normalized_regressions]
    if long_context is not None:
        final_screens.append(long_context)
    for screen in final_screens:
        pair = screen["pair"]
        all_runs.extend((pair["control"], pair["candidate"]))
    all_pairs = [*pairs, *(item["pair"] for item in final_screens)]
    for arm in ("control", "candidate"):
        for field, label in (
                ("ds4_sha256", "binary"),
                ("ds4_bench_tp_sha256", "benchmark executable")):
            identities = {pair[arm]["manifest"].get(field) for pair in all_pairs}
            if len(identities) != 1:
                raise ProofError(
                    f"{arm} runs did not use one {label} across all proof cells")
    run_ids = [run["manifest"].get("run_id") for run in all_runs]
    if len(set(run_ids)) != len(run_ids):
        raise ProofError("promotion proof reuses a process run in multiple slots")
    passed = performance["passed"]
    return {
        "schema_version": 2,
        "kind": "ds4-promotion-proof",
        "stage": stage,
        "candidate_id": candidate_id,
        "lane": lane,
        "baseline_id": spec["baseline_id"],
        "baseline_fnv64": baseline_fnv,
        "source_commits": source_commits,
        "required_provider": spec["required_provider"],
        "candidate_fnv64": next(iter(candidate_fingerprints)),
        "performance_policy": performance_policy,
        "performance": performance,
        "headline_pairs": pairs,
        "diverse_screen": diverse,
        "long_context_screen": long_context,
        "ordinary_regressions": normalized_regressions,
        **({"admission": "none-diagnostic-only"} if sdk_contrast is not None else {}),
        "passed": passed,
    }


def denormalize_run(run: dict) -> dict:
    names = (*RUN_ARTIFACTS, "runtime") if "runtime" in run["artifacts"] else RUN_ARTIFACTS
    return {name: run["artifacts"][name] for name in names}


def proof_to_spec(proof: dict) -> dict:
    def pair_spec(pair: dict) -> dict:
        return {"pair_id": pair["pair_id"], "order": pair["order"],
                "allowed_fields": pair["allowed_fields"],
                "allowed_env": pair["allowed_env"],
                "control": denormalize_run(pair["control"]),
                "candidate": denormalize_run(pair["candidate"])}

    def trajectory_spec(value: dict) -> dict:
        if value["mode"] == "exact":
            return {"mode": "exact"}
        return {"mode": "teacher", "summary": value["summary"],
                "thresholds": value["thresholds"]}

    def screen_spec(screen: dict) -> dict:
        return {"pair": pair_spec(screen["pair"]),
                "control_fnv64": screen["control_fnv64"],
                "trajectory": trajectory_spec(screen["trajectory"]),
                "max_prefill_regression": screen["metrics"]["prefill"]["maximum_regression"],
                "max_decode_regression": screen["metrics"]["decode"]["maximum_regression"]}

    return {
        "schema_version": 2, "kind": "ds4-promotion-proof-spec",
        "stage": proof["stage"],
        "candidate_id": proof["candidate_id"], "lane": proof["lane"],
        "baseline_id": proof["baseline_id"],
        "baseline_fnv64": proof["baseline_fnv64"],
        "source_commits": proof["source_commits"],
        "required_provider": proof["required_provider"],
        "performance": proof["performance_policy"],
        "headline_pairs": [pair_spec(item) for item in proof["headline_pairs"]],
        "diverse_screen": screen_spec(proof["diverse_screen"]),
        "long_context_screen": (
            None if proof["long_context_screen"] is None
            else screen_spec(proof["long_context_screen"])),
        "ordinary_regressions": [
            {"name": item["name"],
             "baseline_fnv64": item["baseline_fnv64"],
             "model": item["model"],
             "screen": screen_spec(item)}
            for item in proof["ordinary_regressions"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--spec", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("proof", type=Path)
    args = parser.parse_args()
    try:
        root = research_root()
        repo = Path(__file__).resolve().parents[1]
        if args.command == "create":
            spec_path = confined(args.spec, root, "promotion-proof spec")
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            result = calculate(spec, root, repo)
            result["spec_sha256"] = canonical_sha256(proof_to_spec(result))
            result["created_utc"] = datetime.now(timezone.utc).isoformat()
            output = confined(args.output, root, "promotion-proof output")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
        else:
            proof_path = confined(args.proof, root, "promotion proof")
            recorded = json.loads(proof_path.read_text(encoding="utf-8"))
            spec = proof_to_spec(recorded)
            if recorded.get("spec_sha256") != canonical_sha256(spec):
                raise ProofError("promotion proof spec digest is invalid")
            result = calculate(spec, root, repo)
            ignored = {"created_utc", "spec_sha256"}
            if ({key: value for key, value in recorded.items() if key not in ignored} !=
                    result):
                raise ProofError("promotion proof differs from recomputed source artifacts")
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["passed"] is not True:
            raise ProofError(
                f"performance decision is {result['performance']['decision']}")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError,
            ProofError) as error:
        print(f"promotion-proof: FAIL {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
