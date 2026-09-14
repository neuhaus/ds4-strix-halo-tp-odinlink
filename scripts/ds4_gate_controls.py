#!/usr/bin/env python3
"""Content-addressed, candidate-blind calibration controls for Gate v2."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path


class ControlError(RuntimeError):
    pass


ROLES = {"self-repeat", "positive", "holdout", "negative"}
FAILURE_REASONS = {
    "far-margin", "hard-safety", "numerical-distribution",
    "non-finite", "quality-nll", "quality-api",
}
CALIBRATION_FIELDS = {"self_repeat", "positive", "holdout", "negative"}
THRESHOLD_SECTIONS = {"numerical", "oracle_numerical", "quality"}
VERIFIER_FILES = {
    "candidate_gate": "scripts/candidate-gate.py",
    "gate_controls": "scripts/ds4_gate_controls.py",
    "numerical": "scripts/compare-teacher-logits.py",
    "quality": "scripts/compare-quality-scores.py",
    "statistics": "scripts/ds4_gate_stats.py",
    "promotion_proof": "scripts/promotion-proof.py",
    "manifest_comparison": "scripts/compare-bench-manifests.py",
    "benchmark_result": "scripts/check-ds4-bench-result.sh",
    "benchmark_producer": "ds4_bench.c",
    "glm5_prefill_proof": "scripts/check-glm5-prefill-proof.sh",
    "rdma_logs": "scripts/check-tp-rdma-logs.sh",
    "benchmark_launcher": "run-tp-ds4-bench.sh",
    "quality_launcher": "run-tp-quality-score.sh",
    "worker_supervisor": "scripts/tp-worker-supervisor.sh",
    "research_root": "scripts/ds4-research-root.sh",
    "gguf_tensor_types": "scripts/gguf_tensor_types.py",
    "performance_design": "scripts/promotion-boundary-repeated-student-v1.json",
    "performance_calibrator": "scripts/calibrate-promotion-boundary.py",
}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verifier_sha256(repo: Path) -> dict[str, str]:
    return {name: sha256(repo / relative)
            for name, relative in VERIFIER_FILES.items()}


@contextmanager
def governance_lock(root: Path):
    directory = root / "gate-governance"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "mutation.lock").open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def governance_mutation(root_position: int):
    """Serialize a complete validate/write/journal governance transaction."""
    def decorate(function):
        @wraps(function)
        def locked(*args, **kwargs):
            root = kwargs.get("root")
            if root is None:
                root = args[root_position]
            with governance_lock(Path(root)):
                return function(*args, **kwargs)
        return locked
    return decorate


def confined(path: Path, root: Path, label: str) -> Path:
    value = path.expanduser().resolve()
    if value != root and root not in value.parents:
        raise ControlError(f"{label} escapes DS4_RESEARCH_ROOT: {value}")
    return value


def file_ref(path: Path, root: Path, label: str) -> dict[str, str]:
    value = confined(path, root, label)
    if not value.is_file():
        raise ControlError(f"missing {label}: {value}")
    return {"path": str(value), "sha256": sha256(value)}


def verify_ref(value: object, root: Path, label: str) -> Path:
    if (not isinstance(value, dict) or set(value) != {"path", "sha256"} or
            not re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", "")))):
        raise ControlError(f"{label} has an invalid artifact binding")
    path = confined(Path(str(value["path"])), root, label)
    if not path.is_file() or sha256(path) != value["sha256"]:
        raise ControlError(f"{label} hash mismatch: {path}")
    return path


def read_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise ControlError(f"invalid manifest line in {path}")
        values[key] = value
    return values


def snapshot_logits(directory: Path, root: Path, label: str) -> dict:
    directory = confined(directory, root, label)
    if not directory.is_dir():
        raise ControlError(f"missing {label}: {directory}")
    files = sorted(directory.glob("decode_*.logits.json"))
    if len(files) < 300:
        raise ControlError(f"{label} must contain at least 300 teacher positions")
    return {
        "path": str(directory),
        "manifest": file_ref(directory / "manifest", root, f"{label} manifest"),
        "files": [file_ref(path, root, f"{label} logit") for path in files],
    }


def verify_logits(value: object, root: Path, label: str) -> Path:
    if (not isinstance(value, dict) or set(value) != {"path", "manifest", "files"} or
            not isinstance(value["files"], list) or len(value["files"]) < 300):
        raise ControlError(f"{label} has an invalid logit snapshot")
    directory = confined(Path(str(value["path"])), root, label)
    manifest = verify_ref(value["manifest"], root, f"{label} manifest")
    if manifest != directory / "manifest":
        raise ControlError(f"{label} manifest is not inside its directory")
    expected_names = []
    for item in value["files"]:
        path = verify_ref(item, root, f"{label} logit")
        if path.parent != directory or not re.fullmatch(
                r"decode_[0-9]{6}\.logits\.json", path.name):
            raise ControlError(f"{label} has an invalid logit path")
        expected_names.append(path.name)
    actual_names = [path.name for path in sorted(directory.glob("decode_*.logits.json"))]
    if expected_names != actual_names or len(set(expected_names)) != len(expected_names):
        raise ControlError(f"{label} file set changed")
    return directory


def normalize_comparisons(value: object, root: Path,
                          source_commit: str) -> dict:
    if not isinstance(value, dict) or not value or set(value) - {"numerical", "quality"}:
        raise ControlError("control comparisons must contain numerical and/or quality")
    result = {}
    if "numerical" in value:
        item = value["numerical"]
        if (not isinstance(item, dict) or
                set(item) != {"reference_dir", "candidate_dir",
                              "allow_quality_difference"} or
                type(item["allow_quality_difference"]) is not bool):
            raise ControlError("numerical control has invalid fields")
        reference = snapshot_logits(Path(str(item["reference_dir"])), root,
                                    "control numerical reference")
        candidate = snapshot_logits(Path(str(item["candidate_dir"])), root,
                                    "control numerical candidate")
        if reference["path"] == candidate["path"]:
            raise ControlError("numerical control arms must use distinct directories")
        candidate_manifest = read_manifest(Path(candidate["manifest"]["path"]))
        reference_manifest = read_manifest(Path(reference["manifest"]["path"]))
        if (candidate_manifest.get("source_commit") != source_commit or
                candidate_manifest.get("source_dirty") != "0"):
            raise ControlError("numerical control candidate source is not clean and bound")
        identity_fields = (
            "model", "model_size", "model_sample_sha256", "prefix_tokens",
            "frozen_token_sha256", "quant_bits", "dspark",
        )
        for field in identity_fields:
            if (field in reference_manifest and field in candidate_manifest and
                    reference_manifest[field] != candidate_manifest[field]):
                raise ControlError(
                    f"numerical control differs in identity field {field}")
        result["numerical"] = {
            "reference": reference, "candidate": candidate,
            "allow_quality_difference": item["allow_quality_difference"],
        }
    if "quality" in value:
        item = value["quality"]
        if not isinstance(item, dict) or set(item) != {"reference", "candidate"}:
            raise ControlError("quality control has invalid fields")
        reference_path = confined(Path(str(item["reference"])), root,
                                  "control quality reference")
        candidate_path = confined(Path(str(item["candidate"])), root,
                                  "control quality candidate")
        if reference_path == candidate_path:
            raise ControlError("quality control arms must use distinct tables")
        reference = file_ref(reference_path, root, "control quality reference")
        candidate = file_ref(candidate_path, root, "control quality candidate")
        reference_manifest = file_ref(reference_path.with_suffix(".manifest"), root,
                                      "control quality reference manifest")
        candidate_manifest = file_ref(candidate_path.with_suffix(".manifest"), root,
                                      "control quality candidate manifest")
        identity = read_manifest(Path(candidate_manifest["path"]))
        reference_identity = read_manifest(Path(reference_manifest["path"]))
        if (identity.get("source_commit") != source_commit or
                identity.get("source_dirty") != "0"):
            raise ControlError("quality control candidate source is not clean and bound")
        for field in ("model", "model_size", "model_sample_sha256",
                      "prefix_tokens", "frozen_token_sha256", "dspark"):
            if (field in reference_identity and field in identity and
                    reference_identity[field] != identity[field]):
                raise ControlError(
                    f"quality control differs in identity field {field}")
        result["quality"] = {
            "reference": reference, "reference_manifest": reference_manifest,
            "candidate": candidate, "candidate_manifest": candidate_manifest,
        }
    return result


def artifact_hashes(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "sha256" and isinstance(item, str) and re.fullmatch(
                    r"[0-9a-f]{64}", item):
                result.add(item)
            else:
                result.update(artifact_hashes(item))
    elif isinstance(value, list):
        for item in value:
            result.update(artifact_hashes(item))
    return result


def journal_entries(root: Path) -> list[dict]:
    path = root / "gate-governance" / "events.jsonl"
    if not path.exists():
        return []
    entries = []
    previous = "0" * 64
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as error:
            raise ControlError(f"invalid governance journal line {index + 1}") from error
        unsigned = {key: value for key, value in entry.items()
                    if key != "entry_sha256"}
        if (entry.get("schema_version") != 1 or entry.get("sequence") != index or
                entry.get("previous_sha256") != previous or
                entry.get("entry_sha256") != canonical_sha256(unsigned)):
            raise ControlError(f"governance journal chain is invalid at line {index + 1}")
        previous = entry["entry_sha256"]
        entries.append(entry)
    return entries


def append_event(root: Path, event: dict) -> dict:
    directory = root / "gate-governance"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "events.jsonl"
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        lines = stream.read().splitlines()
        previous = "0" * 64
        for index, line in enumerate(lines):
            item = json.loads(line)
            unsigned = {key: value for key, value in item.items()
                        if key != "entry_sha256"}
            if (item.get("schema_version") != 1 or
                    item.get("sequence") != index or
                    item.get("previous_sha256") != previous or
                    item.get("entry_sha256") != canonical_sha256(unsigned)):
                raise ControlError("governance journal is not appendable")
            previous = item["entry_sha256"]
        entry = {
            "schema_version": 1, "sequence": len(lines),
            "previous_sha256": previous,
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
        }
        entry["entry_sha256"] = canonical_sha256(entry)
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return entry


def candidate_values(root: Path) -> list[tuple[Path, dict]]:
    result = []
    for path in sorted((root / "candidates").glob("*/candidate.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ControlError(f"cannot audit candidate {path}: {error}") from error
        result.append((path.parent, value))
    return result


def candidate_is_open(dossier: Path) -> bool:
    return not (dossier / "PROMOTED.json").exists() and not (
        dossier / "CLOSED.json").exists()


def event_sequences(root: Path, event_type: str, identity_key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for entry in journal_entries(root):
        event = entry.get("event")
        if not isinstance(event, dict) or event.get("type") != event_type:
            continue
        identity = event.get(identity_key)
        if not isinstance(identity, str) or not identity or identity in result:
            raise ControlError(
                f"governance journal has duplicate or invalid {event_type} identity")
        result[identity] = int(entry["sequence"])
    return result


def candidate_scope(root: Path, candidate: dict) -> str | None:
    baseline_id = candidate.get("baseline_id")
    if not isinstance(baseline_id, str):
        return None
    match = re.fullmatch(r"sha256:([0-9a-f]{64})", baseline_id)
    if not match:
        return None
    path = root / "baselines" / "sha256" / f"{match.group(1)}.json"
    if not path.is_file():
        return None
    try:
        baseline = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ControlError(f"cannot resolve open candidate baseline {path}: {error}") from error
    scope = baseline.get("scope_sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", str(scope or "")):
        raise ControlError(f"open candidate baseline has no valid scope digest: {path}")
    return str(scope)


@governance_mutation(1)
def register_control(repo: Path, root: Path, descriptor_path: Path) -> str:
    descriptor_path = confined(descriptor_path, root, "control descriptor")
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ControlError(f"invalid control descriptor: {error}") from error
    required = {"schema_version", "kind", "control_id", "scope_sha256",
                "role", "source_commit", "comparisons", "expected_failures",
                "notes"}
    if (not isinstance(descriptor, dict) or set(descriptor) != required or
            descriptor.get("schema_version") != 2 or
            descriptor.get("kind") != "ds4-gate-control" or
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}",
                             str(descriptor.get("control_id", ""))) or
            not re.fullmatch(r"[0-9a-f]{64}", str(descriptor.get("scope_sha256", ""))) or
            descriptor.get("role") not in ROLES or
            not re.fullmatch(r"[0-9a-f]{40}", str(descriptor.get("source_commit", ""))) or
            not isinstance(descriptor.get("notes"), str)):
        raise ControlError("control descriptor identity is invalid")
    if subprocess.run(["git", "-C", str(repo), "cat-file", "-e",
                       f"{descriptor['source_commit']}^{{commit}}"],
                      stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL).returncode != 0:
        raise ControlError("control source commit is not available in this repository")
    failures = descriptor["expected_failures"]
    if (not isinstance(failures, dict) or set(failures) - {"numerical", "quality"} or
            any(reason not in FAILURE_REASONS for reason in failures.values()) or
            (descriptor["role"] == "negative") != bool(failures)):
        raise ControlError("control expected_failures do not match its role")
    comparisons = normalize_comparisons(
        descriptor["comparisons"], root, descriptor["source_commit"])
    if set(failures) - set(comparisons):
        raise ControlError("negative failure reason has no corresponding comparison")
    candidate_hashes: set[str] = set()
    for dossier, candidate in candidate_values(root):
        if not candidate_is_open(dossier):
            continue
        source = candidate.get("source")
        if (isinstance(source, dict) and
                source.get("commit") == descriptor["source_commit"]):
            raise ControlError("control reuses an unpromoted candidate source commit")
        candidate_hashes.update(artifact_hashes(candidate))
        for evidence in candidate.get("evidence", []):
            if not isinstance(evidence, dict):
                continue
            path = Path(str(evidence.get("path", "")))
            if not path.is_absolute():
                path = dossier / path
            try:
                if path.is_file():
                    candidate_hashes.update(artifact_hashes(json.loads(
                        path.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError):
                continue
    if artifact_hashes(comparisons) & candidate_hashes:
        raise ControlError("control reuses artifacts from an unpromoted candidate")
    record = {
        **{key: descriptor[key] for key in required - {"comparisons"}},
        "comparisons": comparisons,
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "descriptor_sha256": sha256(descriptor_path),
    }
    digest = canonical_sha256(record)
    destination = root / "controls" / "sha256" / f"{digest}.json"
    if destination.exists():
        raise ControlError(f"control is already registered: sha256:{digest}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, destination)
    try:
        append_event(root, {
            "type": "control-register", "control_id": f"sha256:{digest}",
            "scope_sha256": descriptor["scope_sha256"],
        })
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return f"sha256:{digest}"


def load_control(root: Path, control_id: str) -> dict:
    match = re.fullmatch(r"sha256:([0-9a-f]{64})", control_id)
    if not match:
        raise ControlError("calibration control id must be content-addressed")
    path = root / "controls" / "sha256" / f"{match.group(1)}.json"
    if not path.is_file():
        raise ControlError(f"missing calibration control: {control_id}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if canonical_sha256(value) != match.group(1):
        raise ControlError(f"calibration control digest mismatch: {control_id}")
    for kind, comparison in value["comparisons"].items():
        if kind == "numerical":
            verify_logits(comparison["reference"], root, "control numerical reference")
            verify_logits(comparison["candidate"], root, "control numerical candidate")
        else:
            for name, artifact_value in comparison.items():
                verify_ref(artifact_value, root, f"control quality {name}")
    return value


def run_comparison(repo: Path, root: Path, control: dict, section: str,
                   thresholds: dict) -> tuple[bool, dict | None, str]:
    comparison = control["comparisons"].get(section)
    if comparison is None:
        raise ControlError(
            f"control {control['control_id']} has no {section} comparison")
    with tempfile.TemporaryDirectory(prefix="gate-control-", dir=root) as temporary:
        threshold_path = Path(temporary) / "thresholds.json"
        threshold_path.write_text(json.dumps(
            {"baseline_id": "CONTROL", **thresholds},
            indent=2, sort_keys=True) + "\n")
        if section == "numerical":
            command = [
                sys.executable, str(repo / "scripts" / "compare-teacher-logits.py"),
                comparison["reference"]["path"], comparison["candidate"]["path"],
                "--thresholds", str(threshold_path),
            ]
            if comparison["allow_quality_difference"]:
                command.append("--allow-quality-difference")
        else:
            command = [
                sys.executable, str(repo / "scripts" / "compare-quality-scores.py"),
                comparison["reference"]["path"], comparison["candidate"]["path"],
                "--thresholds", str(threshold_path),
            ]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    parsed = None
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError:
        pass
    passed = bool(result.returncode == 0 and isinstance(parsed, dict) and
                  parsed.get("passed") is True)
    return passed, parsed, result.stderr.strip() or result.stdout.strip()


def failure_matches(reason: str, result: dict | None, detail: str) -> bool:
    if reason == "non-finite":
        return result is None and bool(re.search(r"NaN|Inf|non-finite", detail, re.I))
    if not isinstance(result, dict):
        return False
    if reason == "far-margin":
        return int(result.get("far_margin_inversions", 0)) > 0
    if reason == "hard-safety":
        return int(result.get("hard_safety_breaches", 0)) > 0
    if reason == "numerical-distribution":
        aggregate = result.get("aggregate_gate")
        return (isinstance(aggregate, dict) and aggregate.get("passed") is False and
                not result.get("far_margin_inversions") and
                not result.get("hard_safety_breaches"))
    if reason == "quality-nll":
        return result.get("nll_screen_passed") is False
    if reason == "quality-api":
        return result.get("api_screen_passed") is False
    return False


def _numerical_widened(old: dict, new: dict) -> bool:
    if old.get("schema_version") != 2 or new.get("schema_version") != 2:
        raise ControlError("calibration amendments require numerical schema v2")
    if (new["min_teacher_steps"] < old["min_teacher_steps"] or
            (not old["allow_quality_difference"] and new["allow_quality_difference"])):
        raise ControlError("numerical amendment weakens an immutable coverage/mode rule")
    if new["safety"] != old["safety"]:
        for key, value in new["safety"].items():
            if value > old["safety"][key]:
                raise ControlError(f"numerical amendment widens hard safety {key}")
    old_d, new_d = old["distribution"], new["distribution"]
    for key in ("bootstrap_method", "bootstrap_seed", "cluster_mode", "block_size"):
        if new_d[key] != old_d[key]:
            raise ControlError(f"numerical amendment changes metric meaning {key}")
    if (new_d["bootstrap_resamples"] < old_d["bootstrap_resamples"] or
            new_d["min_clusters"] < old_d["min_clusters"] or
            new["decision"]["confidence_level"] < old["decision"]["confidence_level"]):
        raise ControlError("numerical amendment weakens statistical coverage")
    widened = (
        new["decision"]["e_bound"] > old["decision"]["e_bound"] or
        new["decision"]["max_near_tie_cluster_rate_upper"] >
            old["decision"]["max_near_tie_cluster_rate_upper"] or
        any(new_d[key] > old_d[key] for key in (
            "max_mean_kl_upper", "max_mean_tvd_upper",
            "max_mean_teacher_nll_delta_upper",
            "max_soft_exceedance_cluster_rate_upper")) or
        new_d["min_same_top1_cluster_rate_lower"] <
            old_d["min_same_top1_cluster_rate_lower"] or
        any(new_d["soft_limits"][key] > old_d["soft_limits"][key]
            for key in old_d["soft_limits"])
    )
    return widened


def _quality_widened(old: dict, new: dict) -> bool:
    if old.get("schema_version") != 2 or new.get("schema_version") != 2:
        raise ControlError("calibration amendments require quality schema v2")
    if (new["min_cases"] < old["min_cases"] or
            new["min_target_tokens"] < old["min_target_tokens"]):
        raise ControlError("quality amendment weakens coverage")
    for key in ("method", "seed"):
        if new["bootstrap"][key] != old["bootstrap"][key]:
            raise ControlError(f"quality amendment changes metric meaning {key}")
    if (new["bootstrap"]["resamples"] < old["bootstrap"]["resamples"] or
            new["bootstrap"]["nll_confidence_level"] <
                old["bootstrap"]["nll_confidence_level"] or
            new["bootstrap"]["api_confidence_level"] <
                old["bootstrap"]["api_confidence_level"]):
        raise ControlError("quality amendment weakens statistical coverage")
    if (old["api"]["required"] and not new["api"]["required"] or
            new["api"]["min_cases"] < old["api"]["min_cases"]):
        raise ControlError("quality amendment weakens API coverage")
    if new["nll"]["max_case_delta"] > old["nll"]["max_case_delta"]:
        raise ControlError("quality amendment widens hard per-case NLL safety")
    return bool(
        new["nll"]["max_delta_upper"] > old["nll"]["max_delta_upper"] or
        new["api"]["min_top1_delta_lower"] < old["api"]["min_top1_delta_lower"] or
        new["api"]["min_pair_delta_lower"] < old["api"]["min_pair_delta_lower"])


def evaluate_calibration(repo: Path, root: Path, calibration: object,
                         scope_sha256: str, previous: dict | None,
                         proposed: dict) -> dict:
    if not isinstance(calibration, dict) or set(calibration) != CALIBRATION_FIELDS:
        raise ControlError("calibration must name self-repeat, positive, holdout, and negative controls")
    minimums = {"self_repeat": 1, "positive": 2, "holdout": 1, "negative": 2}
    ids = []
    records: dict[str, list[tuple[str, dict]]] = {key: [] for key in CALIBRATION_FIELDS}
    for field, minimum in minimums.items():
        values = calibration[field]
        if not isinstance(values, list) or len(values) < minimum:
            raise ControlError(f"calibration requires at least {minimum} {field} controls")
        for control_id in values:
            if not isinstance(control_id, str):
                raise ControlError("calibration control IDs must be strings")
            control = load_control(root, control_id)
            expected_role = field.replace("_", "-")
            if control.get("role") != expected_role:
                raise ControlError(f"{control_id} is not a {expected_role} control")
            if control.get("scope_sha256") != scope_sha256:
                raise ControlError(f"{control_id} belongs to a different baseline scope")
            ids.append(control_id)
            records[field].append((control_id, control))
    if len(set(ids)) != len(ids):
        raise ControlError("calibration reuses a control in multiple roles")
    registered = event_sequences(root, "control-register", "control_id")
    if any(control_id not in registered for control_id in ids):
        raise ControlError("calibration control is missing its registration event")
    candidate_inits = event_sequences(root, "candidate-init", "candidate_id")
    latest_control = max(registered[control_id] for control_id in ids)
    selected_controls = [load_control(root, control_id) for control_id in ids]
    selected_source_commits = {
        str(control["source_commit"]) for control in selected_controls
    }
    selected_hashes = set().union(*(
        artifact_hashes(control.get("comparisons", {}))
        for control in selected_controls
    ))
    for dossier, candidate in candidate_values(root):
        if not candidate_is_open(dossier) or candidate_scope(root, candidate) != scope_sha256:
            continue
        candidate_id = candidate.get("candidate_id")
        if candidate_id not in candidate_inits:
            raise ControlError(
                f"open candidate {candidate_id!r} has no auditable initialization event")
        if candidate_inits[candidate_id] <= latest_control:
            raise ControlError(
                f"calibration controls must predate open candidate {candidate_id}; "
                "close and restart that candidate after governance changes")
        candidate_source = candidate.get("source", {}).get("commit")
        if candidate_source in selected_source_commits:
            raise ControlError(
                f"calibration controls reuse the open candidate {candidate_id} source commit")
        candidate_hashes = artifact_hashes(candidate)
        for evidence in candidate.get("evidence", []):
            if not isinstance(evidence, dict):
                continue
            evidence_path = Path(str(evidence.get("path", "")))
            if not evidence_path.is_absolute():
                evidence_path = dossier / evidence_path
            if evidence_path.is_file():
                try:
                    candidate_hashes.update(artifact_hashes(json.loads(
                        evidence_path.read_text(encoding="utf-8"))))
                except (OSError, json.JSONDecodeError):
                    pass
        if selected_hashes & candidate_hashes:
            raise ControlError(
                f"calibration controls reuse artifacts from open candidate {candidate_id}")
    legal_commits = {control["source_commit"]
                     for field in ("positive", "holdout")
                     for _, control in records[field]}
    if len(legal_commits) < 2:
        raise ControlError("positive and holdout controls require two source commits")

    updated_sections = sorted(proposed)
    if not updated_sections or set(updated_sections) - THRESHOLD_SECTIONS:
        raise ControlError("calibration has no supported threshold section")
    widened = {}
    for section in updated_sections:
        if previous is None:
            widened[section] = False
        elif section in {"numerical", "oracle_numerical"}:
            widened[section] = _numerical_widened(previous[section], proposed[section])
        else:
            widened[section] = _quality_widened(previous[section], proposed[section])
        for field in ("self_repeat", "positive", "holdout"):
            comparison_section = (
                "numerical" if section == "oracle_numerical" else section)
            if any(comparison_section not in control["comparisons"]
                   for _, control in records[field]):
                raise ControlError(f"all {field} controls must cover {section}")
        comparison_section = "numerical" if section == "oracle_numerical" else section
        if not any(comparison_section in control["expected_failures"]
                   for _, control in records["negative"]):
            raise ControlError(f"negative controls do not exercise {section}")

    evaluations = []
    widening_witness = {section: False for section in updated_sections}
    for field in ("self_repeat", "positive", "holdout", "negative"):
        for control_id, control in records[field]:
            for section in updated_sections:
                comparison_section = (
                    "numerical" if section == "oracle_numerical" else section)
                if comparison_section not in control["comparisons"]:
                    continue
                new_pass, new_result, detail = run_comparison(
                    repo, root, control, comparison_section, proposed[section])
                old_pass = None
                if previous is not None:
                    old_pass, _, _ = run_comparison(
                        repo, root, control, comparison_section, previous[section])
                if field == "negative":
                    reason = control["expected_failures"].get(comparison_section)
                    if not reason or new_pass or not failure_matches(
                            reason, new_result, detail):
                        raise ControlError(
                            f"negative {control_id} did not fail {section} for {reason}")
                elif not new_pass:
                    raise ControlError(f"{field} {control_id} failed proposed {section} thresholds")
                if field in {"positive", "holdout"} and old_pass is False and new_pass:
                    widening_witness[section] = True
                evaluations.append({
                    "control_id": control_id, "role": control["role"],
                    "section": section, "old_pass": old_pass,
                    "proposed_pass": new_pass,
                    "expected_failure": control["expected_failures"].get(
                        comparison_section),
                })
    for section, is_widened in widened.items():
        if is_widened and not widening_witness[section]:
            raise ControlError(
                f"{section} widening has no registered legal control that needs it")
    payload = {
        "schema_version": 1, "scope_sha256": scope_sha256,
        "control_ids": calibration, "widened": widened,
        "widening_witness": widening_witness,
        "evaluations": evaluations,
        "comparator_sha256": verifier_sha256(repo),
    }
    payload["calibration_sha256"] = canonical_sha256(payload)
    return payload
