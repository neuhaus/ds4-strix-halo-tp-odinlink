#!/usr/bin/env python3
"""Exercise Gate v2 calibration governance independently of GPU inference."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import ds4_gate_controls as controls  # noqa: E402


def record_locked_transaction(root: str, label: str) -> None:
    path = Path(root)
    with controls.governance_lock(path):
        trace = path / "transaction-trace.txt"
        with trace.open("a", encoding="utf-8") as stream:
            stream.write(f"{label}-start\n")
            stream.flush()
            time.sleep(0.15)
            stream.write(f"{label}-end\n")
            stream.flush()


def numerical_thresholds(*, wide: bool) -> dict:
    limit = 0.10 if wide else 0.001
    return {
        "schema_version": 2,
        "min_teacher_steps": 300,
        "allow_quality_difference": False,
        "decision": {
            "e_bound": limit,
            "confidence_level": 0.95,
            "max_near_tie_cluster_rate_upper": 0.25 if wide else 0.01,
        },
        "distribution": {
            "bootstrap_method": "bca",
            "bootstrap_resamples": 1000,
            "bootstrap_seed": 17,
            "cluster_mode": "case-or-contiguous-block",
            "block_size": 30,
            "min_clusters": 5,
            "max_mean_kl_upper": limit,
            "max_mean_tvd_upper": limit,
            "max_mean_teacher_nll_delta_upper": limit,
            "min_same_top1_cluster_rate_lower": 0.70,
            "soft_limits": {
                "centered_p99_abs": limit,
                "centered_nrms": limit,
                "kl": limit,
                "tvd": limit,
            },
            "max_soft_exceedance_cluster_rate_upper": 0.25 if wide else 0.01,
        },
        "safety": {
            "max_centered_abs": 1.0,
            "max_centered_nrms": 1.0,
            "max_kl": 1.0,
            "max_tvd": 1.0,
            "max_abs_teacher_nll_delta": 1.0,
        },
    }


def quality_thresholds(*, wide: bool) -> dict:
    margin = 0.02 if wide else 0.0
    return {
        "schema_version": 2,
        "min_cases": 100,
        "min_target_tokens": 2289,
        "bootstrap": {
            "method": "bca",
            "resamples": 1000,
            "seed": 23,
            "nll_confidence_level": 0.95,
            "api_confidence_level": 0.95,
        },
        "nll": {"max_delta_upper": margin, "max_case_delta": 0.1},
        "api": {
            "required": True,
            "min_cases": 100,
            "min_top1_delta_lower": -margin,
            "min_pair_delta_lower": -margin,
        },
    }


def assert_raises(fragment: str, callback) -> None:
    try:
        callback()
    except controls.ControlError as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError(f"expected ControlError containing {fragment!r}")


def test_registration_and_tampering(root: Path) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    reference = root / "register-reference"
    candidate = root / "register-candidate"
    reference.mkdir()
    candidate.mkdir()
    for directory in (reference, candidate):
        for index in range(300):
            (directory / f"decode_{index:06d}.logits.json").write_text("{}\n")
        (directory / "manifest").write_text(
            f"source_commit={commit}\nsource_dirty=0\n")
    descriptor = root / "register-control.json"
    descriptor.write_text(json.dumps({
        "schema_version": 2,
        "kind": "ds4-gate-control",
        "control_id": "registration-fixture",
        "scope_sha256": "a" * 64,
        "role": "positive",
        "source_commit": commit,
        "comparisons": {"numerical": {
            "reference_dir": str(reference),
            "candidate_dir": str(candidate),
            "allow_quality_difference": False,
        }},
        "expected_failures": {},
        "notes": "registration fixture",
    }, indent=2, sort_keys=True) + "\n")

    dossier = root / "candidates" / "open-fixture"
    dossier.mkdir(parents=True)
    (dossier / "candidate.json").write_text(json.dumps({
        "candidate_id": "open-fixture", "source": {"commit": commit},
        "evidence": [],
    }))
    assert_raises("unpromoted candidate source commit", lambda: controls.register_control(
        REPO, root, descriptor))
    (dossier / "CLOSED.json").write_text("{}\n")
    control_id = controls.register_control(REPO, root, descriptor)
    record = controls.load_control(root, control_id)
    assert record["control_id"] == "registration-fixture"
    first = candidate / "decode_000000.logits.json"
    first.write_text("tampered\n")
    assert_raises("hash mismatch", lambda: controls.load_control(root, control_id))


def test_calibration_rules(root: Path) -> None:
    scope = "b" * 64
    commit_a = "1" * 40
    commit_b = "2" * 40
    records = {
        "self": {
            "role": "self-repeat", "source_commit": commit_a,
            "scope_sha256": scope,
            "comparisons": {"numerical": {}, "quality": {}},
            "expected_failures": {},
        },
        "positive-a": {
            "role": "positive", "source_commit": commit_a,
            "scope_sha256": scope,
            "comparisons": {"numerical": {}, "quality": {}},
            "expected_failures": {}, "needs_widening": True,
        },
        "positive-b": {
            "role": "positive", "source_commit": commit_b,
            "scope_sha256": scope,
            "comparisons": {"numerical": {}, "quality": {}},
            "expected_failures": {},
        },
        "holdout": {
            "role": "holdout", "source_commit": commit_b,
            "scope_sha256": scope,
            "comparisons": {"numerical": {}, "quality": {}},
            "expected_failures": {},
        },
        "negative-numerical": {
            "role": "negative", "source_commit": commit_a,
            "scope_sha256": scope,
            "comparisons": {"numerical": {}},
            "expected_failures": {"numerical": "far-margin"},
        },
        "negative-quality": {
            "role": "negative", "source_commit": commit_b,
            "scope_sha256": scope,
            "comparisons": {"quality": {}},
            "expected_failures": {"quality": "quality-nll"},
        },
    }
    calibration = {
        "self_repeat": ["self"],
        "positive": ["positive-a", "positive-b"],
        "holdout": ["holdout"],
        "negative": ["negative-numerical", "negative-quality"],
    }
    for control_id in records:
        controls.append_event(root, {
            "type": "control-register", "control_id": control_id,
            "scope_sha256": scope,
        })

    original_load = controls.load_control
    original_run = controls.run_comparison

    def fake_load(_root: Path, control_id: str) -> dict:
        return records[control_id]

    def fake_run(_repo: Path, _root: Path, control: dict, section: str,
                 thresholds: dict) -> tuple[bool, dict, str]:
        if control["role"] == "negative":
            if section == "numerical":
                return False, {"far_margin_inversions": 1,
                               "hard_safety_breaches": 0}, ""
            return False, {"nll_screen_passed": False,
                           "api_screen_passed": True}, ""
        old_contract = (
            thresholds["decision"]["e_bound"] < 0.01
            if section == "numerical"
            else thresholds["nll"]["max_delta_upper"] == 0.0)
        passed = not (control.get("needs_widening") and old_contract)
        return passed, {"passed": passed}, ""

    controls.load_control = fake_load
    controls.run_comparison = fake_run
    try:
        previous = {
            "numerical": numerical_thresholds(wide=False),
            "quality": quality_thresholds(wide=False),
        }
        proposed = {
            "numerical": numerical_thresholds(wide=True),
            "quality": quality_thresholds(wide=True),
        }
        result = controls.evaluate_calibration(
            REPO, root, calibration, scope, previous, proposed)
        assert result["widened"] == {"numerical": True, "quality": True}
        assert result["widening_witness"] == {"numerical": True, "quality": True}
        assert len(result["evaluations"]) == 10

        records["positive-a"].pop("needs_widening")
        assert_raises("has no registered legal control", lambda:
                      controls.evaluate_calibration(
                          REPO, root, calibration, scope, previous, proposed))
        records["positive-a"]["needs_widening"] = True

        unsafe = deepcopy(proposed)
        unsafe["numerical"]["safety"]["max_kl"] = 2.0
        assert_raises("widens hard safety", lambda: controls.evaluate_calibration(
            REPO, root, calibration, scope, previous, unsafe))

        unsafe_quality = deepcopy(proposed)
        unsafe_quality["quality"]["nll"]["max_case_delta"] = 0.5
        assert_raises("hard per-case NLL safety", lambda:
                      controls.evaluate_calibration(
                          REPO, root, calibration, scope, previous,
                          unsafe_quality))

        baseline_id = "sha256:" + "c" * 64
        baseline_dir = root / "baselines" / "sha256"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        (baseline_dir / f"{baseline_id[7:]}.json").write_text(json.dumps({
            "scope_sha256": scope,
        }))
        candidate_dir = root / "candidates" / "late-candidate"
        candidate_dir.mkdir(parents=True)
        (candidate_dir / "candidate.json").write_text(json.dumps({
            "candidate_id": "late-candidate", "baseline_id": baseline_id,
        }))
        controls.append_event(root, {
            "type": "candidate-init", "candidate_id": "late-candidate",
        })
        records["late-holdout"] = deepcopy(records["holdout"])
        controls.append_event(root, {
            "type": "control-register", "control_id": "late-holdout",
            "scope_sha256": scope,
        })
        late = deepcopy(calibration)
        late["holdout"] = ["late-holdout"]
        assert_raises("must predate open candidate", lambda:
                      controls.evaluate_calibration(
                          REPO, root, late, scope, previous, proposed))
        (candidate_dir / "CLOSED.json").write_text("{}\n")
        closed_result = controls.evaluate_calibration(
            REPO, root, late, scope, previous, proposed)
        assert closed_result["widening_witness"]["numerical"] is True
    finally:
        controls.load_control = original_load
        controls.run_comparison = original_run


def test_governance_transactions_are_process_serialized(root: Path) -> None:
    context = multiprocessing.get_context("fork")
    workers = [context.Process(
        target=record_locked_transaction, args=(str(root), label))
               for label in ("a", "b")]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)
        assert worker.exitcode == 0
    lines = (root / "transaction-trace.txt").read_text().splitlines()
    assert lines in (["a-start", "a-end", "b-start", "b-end"],
                     ["b-start", "b-end", "a-start", "a-end"]), lines


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        test_registration_and_tampering(Path(temporary))
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        test_calibration_rules(root)
        journal = root / "gate-governance" / "events.jsonl"
        lines = journal.read_text().splitlines()
        first = json.loads(lines[0])
        first["event"]["scope_sha256"] = "f" * 64
        lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
        journal.write_text("\n".join(lines) + "\n")
        assert_raises("journal chain is invalid", lambda: controls.journal_entries(root))
    with tempfile.TemporaryDirectory() as temporary:
        test_governance_transactions_are_process_serialized(Path(temporary))
    print("test_ds4_gate_controls: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
