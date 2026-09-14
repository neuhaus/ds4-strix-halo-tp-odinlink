#!/usr/bin/env python3
"""End-to-end Gate v2 candidate promotion fixture."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_baseline_genesis as genesis_fixture  # noqa: E402
import test_promotion_proof as proof_fixture  # noqa: E402


GATE = REPO / "scripts" / "candidate-gate.py"
PROOF = REPO / "scripts" / "promotion-proof.py"
GATE_SPEC = importlib.util.spec_from_file_location(
    "candidate_gate_fixture", REPO / "scripts" / "candidate-gate.py")
assert GATE_SPEC and GATE_SPEC.loader
GATE_MODULE = importlib.util.module_from_spec(GATE_SPEC)
GATE_SPEC.loader.exec_module(GATE_MODULE)


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["DS4_RESEARCH_ROOT"] = str(root)
    return subprocess.run(
        [sys.executable, str(GATE), *arguments], cwd=REPO, env=environment,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def current_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def advance_source_commit(label: str) -> None:
    subprocess.run([
        "git", "-C", str(REPO), "-c", "user.name=DS4-Gate-Test",
        "-c", "user.email=gate-test.invalid", "commit", "--allow-empty", "-qm", label,
    ], check=True)


def read_manifest(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            result[key] = value
    return result


def write_manifest(path: Path, value: dict[str, str]) -> None:
    path.write_text("".join(f"{key}={item}\n" for key, item in value.items()))


def rebind_run_id(run: dict[str, str], run_id: str) -> dict[str, str]:
    manifest_path = Path(run["manifest"])
    fields = read_manifest(manifest_path)
    fields["run_id"] = run_id
    for field in ("common_env", "worker_env", "coordinator_env"):
        environment = {}
        for assignment in shlex.split(fields.get(field, "")):
            name, separator, value = assignment.partition("=")
            assert separator
            environment[name] = value
        environment["DS4_BENCH_RUN_ID"] = run_id
        fields[field] = shlex.join(
            f"{name}={value}" for name, value in sorted(environment.items()))
    write_manifest(manifest_path, fields)
    for log_name in ("coordinator_log", "worker_log"):
        log_path = Path(run[log_name])
        lines = [line for line in log_path.read_text().splitlines()
                 if not line.startswith("ds4-tp: benchmark run_id=")]
        lines.insert(1, f"ds4-tp: benchmark run_id={run_id}")
        log_path.write_text("\n".join(lines) + "\n")
    return fields


def rewrite_run_rates(run: dict[str, str], prefill: float, decode: float) -> None:
    path = Path(run["csv"])
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        row = next(reader)
    assert fieldnames is not None
    row["prefill_tps"] = str(prefill)
    row["gen_tps"] = str(decode)
    row["gen_steady_tps"] = str(decode)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def prospective_run_id(registered: dict, offset: int, name: str) -> str:
    began = datetime.fromisoformat(registered["recorded_utc"]).astimezone(timezone.utc)
    run = began + timedelta(microseconds=offset)
    return run.strftime("%Y%m%dT%H%M%S.%f") + f"000Z-{name}"


def record_result(root: Path, candidate_id: str, pair_id: str,
                  run_id: str, artifact: dict[str, str]) -> None:
    recorded = run(
        root, "record-result", candidate_id, pair_id, run_id,
        "--manifest", artifact["manifest"],
        "--coordinator-log", artifact["coordinator_log"],
        "--coordinator-status", artifact["coordinator_status"],
        "--worker-log", artifact["worker_log"],
        "--worker-status", artifact["worker_status"])
    assert recorded.returncode == 0, recorded.stderr


def make_headline_run(root: Path, candidate_id: str, baseline_id: str,
                      baseline: dict, registered: dict, arm: str,
                      offset: int, name: str) -> tuple[dict[str, str], str]:
    key = baseline["key"]
    workload = key["workload"]
    intent = json.loads(
        (root / "candidates" / candidate_id / "candidate.json").read_text()
    )["promotion_intent"]
    speed = 205.0 if arm == "control" else 246.0
    artifact = proof_fixture.make_run(
        root, name, registered["pair_id"], registered["order"], arm,
        frontier=int(workload["frontier"]), prefill=speed,
        decode=(20.0 if arm == "control" else 24.0),
        prompt_sha=workload["prompt_sha256"], model=root / "model.gguf",
        model_sample=key["model_sample_sha256"], model_size=key["model_size"])
    run_id = prospective_run_id(registered, offset, name)
    manifest_path = Path(artifact["manifest"])
    manifest = rebind_run_id(artifact, run_id)
    switch_effective = [
        f"{switch}={arms[arm]}" for switch, arms in
        sorted(intent["candidate_switches"].items())
    ]
    effective = shlex.join([
        f"DS4_BENCH_RUN_ID={run_id}", *switch_effective])
    manifest.update({
        "source_commit": current_head(), "source_dirty": "0",
        "candidate": "1", "candidate_lane": "A",
        "candidate_id": candidate_id, "baseline_id": baseline_id,
        "pair_id": registered["pair_id"], "pair_order": registered["order"],
        "pair_arm": arm, "worker_env": effective,
        "coordinator_env": effective,
        "extra_env": shlex.join(switch_effective),
        "toolchain_id": key["toolchain_id"],
        "ds4_sha256": baseline["reference"]["performance"]["roce-v2"][
            "ds4_sha256"],
        "peer_ds4_sha256": baseline["reference"]["performance"]["roce-v2"][
            "ds4_sha256"],
        "model": str(root / "model.gguf"), "model_size": str(key["model_size"]),
        "model_sample_sha256": key["model_sample_sha256"],
        "tp_weight_layout": "q4k-ffn-intermediate",
        "tp_intermediate_size": "2048",
        "tp_intermediate_shards": "1024/1024",
        "tp_expert_count": "288", "tp_experts_used": "8",
        "tp_reduce_op": "sum", "tp_reduce_scope": "all-ranks",
        "tp_reduce_count": "42", "tp_reduce_width": "4096",
        "tp_reduce_dtype": "f32",
    })
    for field in ("prompt_sha256", "frontier", "generated_tokens",
                  "context", "prefill_chunk", "dspark"):
        manifest[field] = str(workload[field])
    write_manifest(manifest_path, manifest)
    return artifact, run_id


def record_pair_runs(root: Path, candidate_id: str, registered: dict,
                     baseline_id: str, baseline: dict) -> None:
    arms = (("control", "candidate") if registered["order"] == "AB"
            else ("candidate", "control"))
    registered["runs"] = {}
    registered["artifacts"] = {}
    for offset, arm in enumerate(arms, start=1):
        artifact, run_id = make_headline_run(
            root, candidate_id, baseline_id, baseline, registered, arm, offset,
            f"{registered['pair_id']}-{arm}")
        recorded = run(
            root, "record-run", candidate_id, registered["pair_id"],
            registered["order"], arm, run_id)
        assert recorded.returncode == 0, recorded.stderr
        record_result(root, candidate_id, registered["pair_id"], run_id, artifact)
        registered["runs"][arm] = run_id
        registered["artifacts"][arm] = artifact


def invalidate_failed_attempt(root: Path, candidate_id: str, baseline_id: str,
                              registered: dict,
                              name: str) -> subprocess.CompletedProcess[str]:
    arm = "control" if registered["order"] == "AB" else "candidate"
    failed = proof_fixture.make_run(
        root, name, registered["pair_id"], registered["order"], arm,
        frontier=2048, prefill=100, decode=10)
    csv_path = Path(failed["csv"])
    csv_path.write_text(csv_path.read_text().splitlines()[0] + "\n")
    coordinator_log = Path(failed["coordinator_log"])
    coordinator_log.write_text(coordinator_log.read_text().replace(
        "ds4-bench-launcher: headline_csv_complete=1",
        "ds4-bench-launcher: headline_csv_complete=0"))
    manifest_path = Path(failed["manifest"])
    fields = read_manifest(manifest_path)
    fields.update({
        "source_commit": current_head(), "source_dirty": "0",
        "candidate": "1", "candidate_lane": "A",
        "candidate_id": candidate_id, "baseline_id": baseline_id,
    })
    write_manifest(manifest_path, fields)
    run_id = prospective_run_id(registered, 1, name)
    rebind_run_id(failed, run_id)
    recorded = run(
        root, "record-run", candidate_id, registered["pair_id"],
        registered["order"], arm, run_id)
    assert recorded.returncode == 0, recorded.stderr
    genesis_fixture.write_manifest(
        Path(failed["worker_status"]), {"exit_code": 1, "signal": 0})
    genesis_fixture.write_manifest(
        Path(failed["coordinator_status"]), {"exit_code": 1, "signal": 0})
    coordinator_log.write_text("\n".join(
        line for line in coordinator_log.read_text().splitlines()
        if line != "ds4-bench: headline_csv_complete=1") + "\n")
    record_result(root, candidate_id, registered["pair_id"], run_id, failed)
    invalidated = run(
        root, "invalidate-pair", candidate_id, registered["pair_id"],
        "--run-id", run_id, "--manifest", failed["manifest"],
        "--coordinator-log", failed["coordinator_log"],
        "--coordinator-status", failed["coordinator_status"],
        "--worker-log", failed["worker_log"],
        "--worker-status", failed["worker_status"],
        "--reason", "injected worker failure")
    return invalidated


def bind_amendment_reviews(root: Path, path: Path) -> None:
    value = json.loads(path.read_text())
    reviewable = deepcopy(value)
    reviewable["evidence"] = []
    payload_sha256 = canonical(reviewable)
    reviews = []
    for reviewer in ("fable", "grok"):
        review = root / f"{reviewer}-{value['amendment_id']}.txt"
        review.write_text(
            "DS4-REVIEW-SCHEMA: 1\n"
            f"REVIEWER: {reviewer}\n"
            f"REVIEWED-PAYLOAD-SHA256: {payload_sha256}\n"
            "VERDICT: GO\n")
        reviews.append({
            "kind": f"{reviewer}-review", "path": str(review),
            "sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
        })
    value["evidence"] = reviews
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def iter_pair_runs(spec: dict):
    for pair in spec["headline_pairs"]:
        yield pair
    yield spec["diverse_screen"]["pair"]
    yield spec["long_context_screen"]["pair"]
    for regression in spec["ordinary_regressions"]:
        yield regression["screen"]["pair"]


def build_lane_a_proof(root: Path, baseline_id: str, baseline: dict,
                       candidate_id: str, registered_pairs: list[dict], *,
                       manifest_candidate_id: str | None = None) -> Path:
    spec_path = proof_fixture.build_spec(root, pair_count=5)
    spec = json.loads(spec_path.read_text())
    source = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    candidate = json.loads(
        (root / "candidates" / candidate_id / "candidate.json").read_text())
    intent = candidate["promotion_intent"]
    model = root / "model.gguf"
    key = baseline["key"]
    spec.update({
        "candidate_id": candidate_id,
        "baseline_id": baseline_id,
        "baseline_fnv64": baseline["reference"]["fnv64"],
        "source_commits": {"control": source, "candidate": source},
    })
    spec["performance"]["contract"] = baseline["thresholds"]["performance"]
    spec["performance"]["target_metrics"] = intent["target_metrics"]
    spec["performance"]["candidate_switches"] = intent["candidate_switches"]
    if len(registered_pairs) != len(spec["headline_pairs"]):
        raise AssertionError("registered pair count differs from the proof fixture")
    for pair in spec["headline_pairs"]:
        rewrite_run_rates(pair["control"], 205.0, 20.0)
        rewrite_run_rates(pair["candidate"], 246.0, 24.0)
    for pair, registered in zip(spec["headline_pairs"], registered_pairs):
        pair["pair_id"] = registered["pair_id"]
        pair["order"] = registered["order"]
        for arm in ("control", "candidate"):
            if manifest_candidate_id is None:
                pair[arm] = registered["artifacts"][arm]
            else:
                manifest_path = Path(pair[arm]["manifest"])
                manifest = rebind_run_id(pair[arm], registered["runs"][arm])
                manifest["pair_id"] = registered["pair_id"]
                manifest["pair_order"] = registered["order"]
                write_manifest(manifest_path, manifest)
    glm_pairs = {
        id(pair) for pair in [*spec["headline_pairs"],
                              spec["diverse_screen"]["pair"],
                              spec["long_context_screen"]["pair"]]
    }
    for pair in iter_pair_runs(spec):
        pair["allowed_env"] = sorted(intent["candidate_switches"])
        if pair in spec["headline_pairs"] and manifest_candidate_id is None:
            continue
        for arm in ("control", "candidate"):
            manifest_path = Path(pair[arm]["manifest"])
            manifest = read_manifest(manifest_path)
            switch_effective = [
                f"{name}={arms[arm]}" for name, arms in
                sorted(intent["candidate_switches"].items())
            ]
            effective = shlex.join([
                f"DS4_BENCH_RUN_ID={manifest['run_id']}", *switch_effective])
            manifest["source_commit"] = source
            manifest["baseline_id"] = baseline_id
            manifest["candidate_lane"] = "A"
            manifest["candidate_id"] = (candidate_id if manifest_candidate_id is None
                                        else manifest_candidate_id)
            manifest["worker_env"] = effective
            manifest["coordinator_env"] = effective
            manifest["extra_env"] = shlex.join(switch_effective)
            manifest["toolchain_id"] = key["toolchain_id"]
            baseline_binary = baseline["reference"]["performance"]["roce-v2"][
                "ds4_sha256"]
            manifest["ds4_sha256"] = baseline_binary
            manifest["peer_ds4_sha256"] = baseline_binary
            if id(pair) in glm_pairs:
                manifest.update({
                    "model": str(model),
                    "model_size": str(key["model_size"]),
                    "model_sample_sha256": key["model_sample_sha256"],
                    "tp_weight_layout": "q4k-ffn-intermediate",
                    "tp_intermediate_size": "2048",
                    "tp_intermediate_shards": "1024/1024",
                    "tp_expert_count": "288",
                    "tp_experts_used": "8",
                    "tp_reduce_op": "sum",
                    "tp_reduce_scope": "all-ranks",
                    "tp_reduce_count": "42",
                    "tp_reduce_width": "4096",
                    "tp_reduce_dtype": "f32",
                })
            if pair in spec["headline_pairs"]:
                workload = key["workload"]
                for field in ("prompt_sha256", "frontier", "generated_tokens",
                              "context", "prefill_chunk", "dspark"):
                    manifest[field] = str(workload[field])
            write_manifest(manifest_path, manifest)
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    proof = root / "lane-a-promotion-proof.json"
    environment = os.environ.copy()
    environment["DS4_RESEARCH_ROOT"] = str(root)
    result = subprocess.run(
        [sys.executable, str(PROOF), "create", "--spec", str(spec_path),
         "--output", str(proof)], cwd=REPO, env=environment,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return proof


def configure_candidate(root: Path, candidate_id: str, baseline_id: str,
                        baseline: dict, proof: Path | None) -> Path:
    dossier = root / "candidates" / candidate_id
    path = dossier / "candidate.json"
    value = json.loads(path.read_text())
    model = root / "model.gguf"
    value["baseline_id"] = baseline_id
    value["model"] = {
        "path": str(model),
        "sample_sha256": baseline["key"]["model_sample_sha256"],
        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "size": model.stat().st_size,
        "quantization": "Q4_K",
    }
    value["toolchain"] = {"id": baseline["key"]["toolchain_id"]}
    value["transport"] = {"providers": ["roce-v2"]}
    value["evidence"] = [] if proof is None else [{
        "kind": "promotion-proof", "path": str(proof),
        "sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
    }]
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def main() -> int:
    assert not subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain=v1", "-uall"],
        text=True), "candidate integration test requires its clean wrapper clone"
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        genesis = genesis_fixture.build_fixture(root, providers=("roce-v2",))
        created = genesis_fixture.run_gate(root, genesis)
        assert created.returncode == 0, created.stderr
        baseline_id = created.stdout.strip()
        baseline_path = root / "baselines" / "sha256" / f"{baseline_id[7:]}.json"
        baseline = json.loads(baseline_path.read_text())

        try:
            GATE_MODULE.manifest_switch_values({
                "worker_env": "DS4_FEATURE=1",
                "coordinator_env": "DS4_FEATURE=0",
            }, {"DS4_FEATURE": {"control": "0", "candidate": "1"}},
                "mismatched quality fixture")
        except GATE_MODULE.GateError as error:
            assert "ranks disagree" in str(error)
        else:
            raise AssertionError("teacher/quality switch mismatch was accepted")

        wrong_thresholds = root / "wrong-screen-thresholds.json"
        wrong_thresholds.write_text(json.dumps({
            "baseline_id": baseline_id, "schema_version": 2,
        }) + "\n")
        wrong_proof = {
            "diverse_screen": {"trajectory": {
                "mode": "teacher", "thresholds": {
                    "path": str(wrong_thresholds),
                    "sha256": hashlib.sha256(
                        wrong_thresholds.read_bytes()).hexdigest(),
                }}},
            "long_context_screen": {"trajectory": {"mode": "exact"}},
            "ordinary_regressions": [],
        }
        try:
            GATE_MODULE.verify_screen_threshold_contract(
                root, wrong_proof, baseline_id, baseline)
        except GATE_MODULE.GateError as error:
            assert "differ from the active baseline" in str(error)
        else:
            raise AssertionError("mismatched screen thresholds were accepted")

        required_verifiers = {
            "candidate_gate", "gate_controls", "benchmark_launcher",
            "benchmark_producer", "quality_launcher", "worker_supervisor", "research_root",
            "gguf_tensor_types", "glm5_prefill_proof",
        }
        assert required_verifiers <= set(
            baseline["provenance"]["verifier_sha256"])
        for verifier in ["statistics", *sorted(required_verifiers)]:
            stale_verifiers = deepcopy(baseline)
            stale_verifiers["provenance"]["verifier_sha256"][verifier] = "0" * 64
            try:
                GATE_MODULE.verify_verifier_identity(REPO, stale_verifiers)
            except GATE_MODULE.GateError as error:
                assert "verifier identity differs" in str(error)
            else:
                raise AssertionError(f"stale {verifier} identity was accepted")

        dirty_marker = REPO / ".candidate-dirty-init-fixture"
        dirty_marker.write_text("dirty\n")
        dirty_init = run(root, "init", "dirty-init", "A",
                         "--switch", "DS4_DIRTY=0,1")
        dirty_marker.unlink()
        assert dirty_init.returncode != 0
        assert "requires a clean source checkout" in dirty_init.stderr

        selection_id = "prospective-pair-selection"
        initialized = run(root, "init", selection_id, "A",
                          "--switch", "DS4_SELECTION=0,1")
        assert initialized.returncode == 0, initialized.stderr
        configure_candidate(root, selection_id, baseline_id, baseline, None)
        first = run(root, "begin-pair", selection_id)
        assert first.returncode == 0, first.stderr
        first_pair = json.loads(first.stdout)
        failure_run = proof_fixture.make_run(
            root, "selection-failed-attempt", first_pair["pair_id"],
            first_pair["order"], "control", frontier=2048,
            prefill=100, decode=10)
        failure_manifest = Path(failure_run["manifest"])
        failure_fields = read_manifest(failure_manifest)
        failure_fields.update({
            "source_commit": subprocess.check_output(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            text=True).strip(),
            "source_dirty": "0", "candidate": "1", "candidate_lane": "A",
            "candidate_id": selection_id, "baseline_id": baseline_id,
            "run_id": "20200101T000000.000000000Z-predates-begin",
        })
        write_manifest(failure_manifest, failure_fields)
        predates = run(
            root, "record-run", selection_id, first_pair["pair_id"],
            first_pair["order"], "control", failure_fields["run_id"])
        assert predates.returncode != 0
        assert "predates its prospective pair-begin" in predates.stderr
        failure_fields = rebind_run_id(
            failure_run,
            prospective_run_id(first_pair, 1, "selection-failed-attempt"))
        wrong_order = "BA" if first_pair["order"] == "AB" else "AB"
        mislabeled = run(
            root, "record-run", selection_id, first_pair["pair_id"], wrong_order,
            "control", failure_fields["run_id"])
        assert mislabeled.returncode != 0
        assert "order differs" in mislabeled.stderr
        recorded_failure = run(
            root, "record-run", selection_id, first_pair["pair_id"],
            first_pair["order"], "control", failure_fields["run_id"])
        assert recorded_failure.returncode == 0, recorded_failure.stderr
        duplicate_attempt = run(
            root, "record-run", selection_id, first_pair["pair_id"],
            first_pair["order"], "control",
            prospective_run_id(first_pair, 2, "selection-duplicate-attempt"))
        assert duplicate_attempt.returncode != 0
        assert "duplicate or contradicts AB/BA order" in duplicate_attempt.stderr
        reused_run = run(
            root, "record-run", selection_id, first_pair["pair_id"],
            first_pair["order"], "candidate", failure_fields["run_id"])
        assert reused_run.returncode != 0
        assert "already journaled" in reused_run.stderr
        premature_second = run(
            root, "record-run", selection_id, first_pair["pair_id"],
            first_pair["order"], "candidate",
            prospective_run_id(first_pair, 2, "premature-second-arm"))
        assert premature_second.returncode != 0
        assert "preceding headline arm must bind its result" in \
            premature_second.stderr
        record_result(
            root, selection_id, first_pair["pair_id"],
            failure_fields["run_id"], failure_run)
        invalidation_args = (
            "invalidate-pair", selection_id, first_pair["pair_id"],
            "--run-id", failure_fields["run_id"],
            "--manifest", failure_run["manifest"],
            "--coordinator-log", failure_run["coordinator_log"],
            "--coordinator-status", failure_run["coordinator_status"],
            "--worker-log", failure_run["worker_log"],
            "--worker-status", failure_run["worker_status"],
        )
        wrong_run = run(
            root, "invalidate-pair", selection_id, first_pair["pair_id"],
            "--run-id", prospective_run_id(first_pair, 2, "not-journaled"),
            "--manifest", failure_run["manifest"],
            "--coordinator-log", failure_run["coordinator_log"],
            "--coordinator-status", failure_run["coordinator_status"],
            "--worker-log", failure_run["worker_log"],
            "--worker-status", failure_run["worker_status"],
            "--reason", "wrong run")
        assert wrong_run.returncode != 0
        assert "do not identify the active attempt" in wrong_run.stderr
        rejected_invalidation = run(
            root, *invalidation_args, "--reason", "not failed")
        assert rejected_invalidation.returncode != 0
        assert "complete timing result cannot be invalidated" in \
            rejected_invalidation.stderr
        failure_csv = Path(failure_run["csv"])
        complete_bytes = failure_csv.read_bytes()
        failure_csv.write_text(failure_csv.read_text().splitlines()[0] + "\n")
        selection_value = json.loads(
            (root / "candidates" / selection_id / "candidate.json").read_text())
        try:
            GATE_MODULE.headline_pair_state(root, selection_id, selection_value)
        except GATE_MODULE.GateError as error:
            assert "headline result CSV is not immutable" in str(error)
        else:
            raise AssertionError("truncated journaled result was accepted during replay")
        failure_csv.write_bytes(complete_bytes)
        closed_selection = run(
            root, "close-candidate", selection_id,
            "--reason", "prospective identity fixture complete")
        assert closed_selection.returncode == 0, closed_selection.stderr
        restarted = run(root, "init", "restarted-pair-selection", "A",
                        "--switch", "DS4_SELECTION=0,1")
        assert restarted.returncode != 0
        assert "every new formal hypothesis requires a new commit" in restarted.stderr
        cosmetic_restart = run(root, "init", "cosmetic-pair-selection", "A",
                               "--switch", "DS4_NOOP=0,1")
        assert cosmetic_restart.returncode != 0
        assert "every new formal hypothesis requires a new commit" in \
            cosmetic_restart.stderr

        status_probe = proof_fixture.make_run(
            root, "invalidation-status-probe", "probe-pair", "AB", "control",
            frontier=2048, prefill=100, decode=10)
        Path(status_probe["csv"]).write_text(
            Path(status_probe["csv"]).read_text().splitlines()[0] + "\n")
        status_log = Path(status_probe["coordinator_log"])
        status_log.write_text(status_log.read_text().replace(
            "ds4-bench-launcher: headline_csv_complete=1",
            "ds4-bench-launcher: headline_csv_complete=0"))
        status_paths = {
            name: Path(status_probe[name]) for name in (
                "manifest", "coordinator_log", "coordinator_status",
                "worker_log", "worker_status")
        }
        genesis_fixture.write_manifest(
            status_paths["worker_status"], {"exit_code": 1, "signal": 0})
        try:
            GATE_MODULE.verify_invalidation_process_failure(
                status_paths, "producer race")
        except GATE_MODULE.GateError as error:
            assert "attested complete timing result" in str(error)
        else:
            raise AssertionError("producer-attested complete result was invalidatable")
        status_log.write_text("\n".join(
            line for line in status_log.read_text().splitlines()
            if line != "ds4-bench: headline_csv_complete=1") + "\n")
        genesis_fixture.write_manifest(
            status_paths["worker_status"], {"exit_code": 143, "signal": 15})
        try:
            GATE_MODULE.verify_invalidation_process_failure(
                status_paths, "operator termination")
        except GATE_MODULE.GateError as error:
            assert "requires a nonsignal worker" in str(error)
        else:
            raise AssertionError("unattested worker signal was invalidatable")
        genesis_fixture.write_manifest(
            status_paths["coordinator_status"], {"exit_code": 137, "signal": 9})
        genesis_fixture.write_manifest(
            status_paths["worker_status"], {"exit_code": 1, "signal": 0})
        try:
            GATE_MODULE.verify_invalidation_process_failure(
                status_paths, "coordinator termination")
        except GATE_MODULE.GateError as error:
            assert "requires a nonsignal coordinator" in str(error)
        else:
            raise AssertionError("coordinator signal was invalidatable")
        genesis_fixture.write_manifest(
            status_paths["coordinator_status"], {"exit_code": 1, "signal": 0})
        genesis_fixture.write_manifest(
            status_paths["worker_status"], {"exit_code": 143, "signal": 15})
        status_log.write_text(status_log.read_text().replace(
            "worker_terminated_by_launcher=0", "worker_terminated_by_launcher=1"))
        GATE_MODULE.verify_invalidation_process_failure(
            status_paths, "launcher cleanup")

        advance_source_commit("test: seal producer completion before invalidation")
        producer_race_id = "producer-completion-race"
        initialized = run(root, "init", producer_race_id, "A", "--switch",
                          "DS4_PRODUCER_RACE=0,1")
        assert initialized.returncode == 0, initialized.stderr
        configure_candidate(root, producer_race_id, baseline_id, baseline, None)
        begun = run(root, "begin-pair", producer_race_id)
        assert begun.returncode == 0, begun.stderr
        producer_pair = json.loads(begun.stdout)
        producer_arm = ("control" if producer_pair["order"] == "AB"
                        else "candidate")
        producer_run = proof_fixture.make_run(
            root, "producer-completion-race", producer_pair["pair_id"],
            producer_pair["order"], producer_arm, frontier=2048,
            prefill=100, decode=10)
        producer_csv = Path(producer_run["csv"])
        producer_csv.write_text(producer_csv.read_text().splitlines()[0] + "\n")
        producer_log = Path(producer_run["coordinator_log"])
        producer_log.write_text(producer_log.read_text().replace(
            "ds4-bench-launcher: headline_csv_complete=1",
            "ds4-bench-launcher: headline_csv_complete=0"))
        producer_manifest = Path(producer_run["manifest"])
        producer_fields = read_manifest(producer_manifest)
        producer_fields.update({
            "source_commit": current_head(), "source_dirty": "0",
            "candidate": "1", "candidate_lane": "A",
            "candidate_id": producer_race_id, "baseline_id": baseline_id,
        })
        write_manifest(producer_manifest, producer_fields)
        producer_run_id = prospective_run_id(
            producer_pair, 1, "producer-completion-race")
        rebind_run_id(producer_run, producer_run_id)
        recorded = run(
            root, "record-run", producer_race_id, producer_pair["pair_id"],
            producer_pair["order"], producer_arm, producer_run_id)
        assert recorded.returncode == 0, recorded.stderr
        genesis_fixture.write_manifest(
            Path(producer_run["coordinator_status"]),
            {"exit_code": 1, "signal": 0})
        record_result(
            root, producer_race_id, producer_pair["pair_id"],
            producer_run_id, producer_run)
        producer_race_invalidation = run(
            root, "invalidate-pair", producer_race_id, producer_pair["pair_id"],
            "--run-id", producer_run_id,
            "--manifest", producer_run["manifest"],
            "--coordinator-log", producer_run["coordinator_log"],
            "--coordinator-status", producer_run["coordinator_status"],
            "--worker-log", producer_run["worker_log"],
            "--worker-status", producer_run["worker_status"],
            "--reason", "CSV disappeared after producer completion")
        assert producer_race_invalidation.returncode != 0
        assert "attested complete timing result" in \
            producer_race_invalidation.stderr
        closed_producer_race = run(
            root, "close-candidate", producer_race_id,
            "--reason", "producer completion race fixture complete")
        assert closed_producer_race.returncode == 0, closed_producer_race.stderr

        advance_source_commit("test: isolate selection-bypass hypothesis")
        selection_bypass_id = "selection-bypass-proof"
        initialized = run(root, "init", selection_bypass_id, "A",
                          "--switch", "DS4_SELECTION=0,1")
        assert initialized.returncode == 0, initialized.stderr
        configure_candidate(
            root, selection_bypass_id, baseline_id, baseline, None)
        registered_selection = []
        for _ in range(7):
            begun = run(root, "begin-pair", selection_bypass_id)
            assert begun.returncode == 0, begun.stderr
            registered = json.loads(begun.stdout)
            record_pair_runs(
                root, selection_bypass_id, registered, baseline_id, baseline)
            registered_selection.append(registered)
        selected_proof = build_lane_a_proof(
            root, baseline_id, baseline, selection_bypass_id,
            registered_selection[:5])
        configure_candidate(
            root, selection_bypass_id, baseline_id, baseline, selected_proof)
        selected = run(root, "check", selection_bypass_id)
        assert selected.returncode != 0
        assert "prospective pair/provider contract" in selected.stderr
        closed_selection_bypass = run(
            root, "close-candidate", selection_bypass_id,
            "--reason", "selection-bypass fixture complete")
        assert closed_selection_bypass.returncode == 0, \
            closed_selection_bypass.stderr

        advance_source_commit("test: bind absent invalidation result")
        absent_id = "absent-result-binding"
        initialized = run(root, "init", absent_id, "A", "--switch",
                          "DS4_ABSENT=0,1")
        assert initialized.returncode == 0, initialized.stderr
        configure_candidate(root, absent_id, baseline_id, baseline, None)
        begun = run(root, "begin-pair", absent_id)
        assert begun.returncode == 0, begun.stderr
        absent_pair = json.loads(begun.stdout)
        absent_arm = "control" if absent_pair["order"] == "AB" else "candidate"
        absent_run = proof_fixture.make_run(
            root, "absent-result", absent_pair["pair_id"], absent_pair["order"],
            absent_arm, frontier=2048, prefill=100, decode=10)
        absent_csv = Path(absent_run["csv"])
        absent_csv.unlink()
        absent_log = Path(absent_run["coordinator_log"])
        absent_log.write_text(absent_log.read_text().replace(
            "ds4-bench-launcher: headline_csv_complete=1",
            "ds4-bench-launcher: headline_csv_complete=0"))
        absent_log.write_text("\n".join(
            line for line in absent_log.read_text().splitlines()
            if line != "ds4-bench: headline_csv_complete=1") + "\n")
        absent_manifest = Path(absent_run["manifest"])
        absent_fields = read_manifest(absent_manifest)
        absent_fields.update({
            "source_commit": current_head(), "source_dirty": "0",
            "candidate": "1", "candidate_lane": "A",
            "candidate_id": absent_id, "baseline_id": baseline_id,
        })
        write_manifest(absent_manifest, absent_fields)
        absent_run_id = prospective_run_id(absent_pair, 1, "absent-result")
        rebind_run_id(absent_run, absent_run_id)
        recorded = run(root, "record-run", absent_id, absent_pair["pair_id"],
                       absent_pair["order"], absent_arm, absent_run_id)
        assert recorded.returncode == 0, recorded.stderr
        genesis_fixture.write_manifest(
            Path(absent_run["coordinator_status"]), {"exit_code": 1, "signal": 0})
        genesis_fixture.write_manifest(
            Path(absent_run["worker_status"]), {"exit_code": 143, "signal": 15})
        absent_log.write_text(absent_log.read_text().replace(
            "worker_terminated_by_launcher=0", "worker_terminated_by_launcher=1"))
        record_result(
            root, absent_id, absent_pair["pair_id"], absent_run_id, absent_run)
        absent_second_arm = "candidate" if absent_arm == "control" else "control"
        blocked_second = run(
            root, "record-run", absent_id, absent_pair["pair_id"],
            absent_pair["order"], absent_second_arm,
            prospective_run_id(absent_pair, 2, "after-failed-first-arm"))
        assert blocked_second.returncode != 0
        assert "failed first arm must be invalidated" in blocked_second.stderr
        absent_invalidation = run(
            root, "invalidate-pair", absent_id, absent_pair["pair_id"],
            "--run-id", absent_run_id, "--manifest", absent_run["manifest"],
            "--coordinator-log", absent_run["coordinator_log"],
            "--coordinator-status", absent_run["coordinator_status"],
            "--worker-log", absent_run["worker_log"],
            "--worker-status", absent_run["worker_status"],
            "--reason", "coordinator failed before writing CSV")
        assert absent_invalidation.returncode == 0, absent_invalidation.stderr
        absent_event = [
            entry["event"] for entry in GATE_MODULE.journal_entries(root)
            if entry.get("event", {}).get("type") == "headline-pair-invalidate" and
            entry["event"].get("candidate_id") == absent_id
        ][-1]
        assert absent_event["result_csv"] == {
            "path": str(absent_csv.resolve()), "exists": False,
        }
        absent_value = json.loads(
            (root / "candidates" / absent_id / "candidate.json").read_text())
        GATE_MODULE.headline_pair_state(root, absent_id, absent_value)
        absent_csv.write_text("created after invalidation\n")
        try:
            GATE_MODULE.headline_pair_state(root, absent_id, absent_value)
        except GATE_MODULE.GateError as error:
            assert "result CSV absence changed" in str(error)
        else:
            raise AssertionError("created invalidation CSV was accepted during replay")
        absent_csv.unlink()
        closed_absent = run(root, "close-candidate", absent_id,
                            "--reason", "absent-result replay fixture complete")
        assert closed_absent.returncode == 0, closed_absent.stderr

        advance_source_commit("test: reject kill after first BA arm")
        abort_id = "abort-after-first-ba-arm"
        initialized = run(root, "init", abort_id, "A", "--first-order", "BA",
                          "--switch", "DS4_ABORT=0,1")
        assert initialized.returncode == 0, initialized.stderr
        configure_candidate(root, abort_id, baseline_id, baseline, None)
        begun = run(root, "begin-pair", abort_id)
        assert begun.returncode == 0, begun.stderr
        abort_pair = json.loads(begun.stdout)
        assert abort_pair["order"] == "BA"
        visible_run = proof_fixture.make_run(
            root, "visible-candidate-arm", abort_pair["pair_id"], "BA",
            "candidate", frontier=2048, prefill=80, decode=8)
        visible_manifest = Path(visible_run["manifest"])
        visible_fields = read_manifest(visible_manifest)
        visible_fields.update({
            "source_commit": current_head(), "source_dirty": "0",
            "candidate": "1", "candidate_lane": "A",
            "candidate_id": abort_id, "baseline_id": baseline_id,
        })
        write_manifest(visible_manifest, visible_fields)
        first_run_id = prospective_run_id(abort_pair, 1, "visible-candidate-arm")
        rebind_run_id(visible_run, first_run_id)
        assert run(root, "record-run", abort_id, abort_pair["pair_id"], "BA",
                   "candidate", first_run_id).returncode == 0
        record_result(
            root, abort_id, abort_pair["pair_id"], first_run_id, visible_run)
        killed_run = proof_fixture.make_run(
            root, "killed-control-after-candidate", abort_pair["pair_id"], "BA",
            "control", frontier=2048, prefill=100, decode=10)
        killed_manifest = Path(killed_run["manifest"])
        killed_fields = read_manifest(killed_manifest)
        killed_fields.update({
            "source_commit": current_head(), "source_dirty": "0",
            "candidate": "1", "candidate_lane": "A",
            "candidate_id": abort_id, "baseline_id": baseline_id,
        })
        write_manifest(killed_manifest, killed_fields)
        killed_csv = Path(killed_run["csv"])
        killed_csv.write_text(killed_csv.read_text().splitlines()[0] + "\n")
        killed_log = Path(killed_run["coordinator_log"])
        killed_log.write_text(killed_log.read_text().replace(
            "ds4-bench-launcher: headline_csv_complete=1",
            "ds4-bench-launcher: headline_csv_complete=0"))
        killed_log.write_text("\n".join(
            line for line in killed_log.read_text().splitlines()
            if line != "ds4-bench: headline_csv_complete=1") + "\n")
        killed_run_id = prospective_run_id(
            abort_pair, 2, "killed-control-after-candidate")
        rebind_run_id(killed_run, killed_run_id)
        recorded = run(root, "record-run", abort_id, abort_pair["pair_id"], "BA",
                       "control", killed_run_id)
        assert recorded.returncode == 0, recorded.stderr
        genesis_fixture.write_manifest(
            Path(killed_run["coordinator_status"]), {"exit_code": 1, "signal": 0})
        genesis_fixture.write_manifest(
            Path(killed_run["worker_status"]), {"exit_code": 1, "signal": 0})
        record_result(
            root, abort_id, abort_pair["pair_id"], killed_run_id, killed_run)
        killed_invalidation = run(
            root, "invalidate-pair", abort_id, abort_pair["pair_id"],
            "--run-id", killed_run_id, "--manifest", killed_run["manifest"],
            "--coordinator-log", killed_run["coordinator_log"],
            "--coordinator-status", killed_run["coordinator_status"],
            "--worker-log", killed_run["worker_log"],
            "--worker-status", killed_run["worker_status"],
            "--reason", "killed after observing first arm")
        assert killed_invalidation.returncode != 0
        assert "only a first-arm pre-result failure" in \
            killed_invalidation.stderr
        blocked_after_second_failure = run(root, "begin-pair", abort_id)
        assert blocked_after_second_failure.returncode != 0
        assert "failed completed pair closes this candidate" in \
            blocked_after_second_failure.stderr
        closed_abort = run(root, "close-candidate", abort_id,
                           "--reason", "operator-kill adversarial fixture complete")
        assert closed_abort.returncode == 0, closed_abort.stderr

        advance_source_commit("test: isolate retry-cap hypothesis")
        retry_id = "bounded-pair-retries"
        initialized = run(root, "init", retry_id, "A",
                          "--switch", "DS4_RETRY=0,1")
        assert initialized.returncode == 0, initialized.stderr
        configure_candidate(root, retry_id, baseline_id, baseline, None)
        for attempt in range(1, 3):
            begun = run(root, "begin-pair", retry_id)
            assert begun.returncode == 0, begun.stderr
            registered = json.loads(begun.stdout)
            invalidated = invalidate_failed_attempt(
                root, retry_id, baseline_id, registered,
                f"retry-cap-attempt-{attempt}")
            assert invalidated.returncode == 0, invalidated.stderr
        exhausted_pair = run(root, "begin-pair", retry_id)
        assert exhausted_pair.returncode != 0
        assert "exhausted its two permitted attempts" in exhausted_pair.stderr
        closed_retry = run(root, "close-candidate", retry_id,
                           "--reason", "retry cap fixture complete")
        assert closed_retry.returncode == 0, closed_retry.stderr

        advance_source_commit("test: isolate global-invalidation-cap hypothesis")
        global_id = "bounded-candidate-invalidations"
        initialized = run(root, "init", global_id, "A",
                          "--switch", "DS4_GLOBAL_RETRY=0,1")
        assert initialized.returncode == 0, initialized.stderr
        configure_candidate(root, global_id, baseline_id, baseline, None)
        for pair_index in range(1, 3):
            begun = run(root, "begin-pair", global_id)
            assert begun.returncode == 0, begun.stderr
            registered = json.loads(begun.stdout)
            invalidated = invalidate_failed_attempt(
                root, global_id, baseline_id, registered,
                f"global-cap-pair-{pair_index}-failed")
            assert invalidated.returncode == 0, invalidated.stderr
            replacement = run(root, "begin-pair", global_id)
            assert replacement.returncode == 0, replacement.stderr
            record_pair_runs(
                root, global_id, json.loads(replacement.stdout),
                baseline_id, baseline)
        third = run(root, "begin-pair", global_id)
        assert third.returncode == 0, third.stderr
        blocked_invalidation = invalidate_failed_attempt(
            root, global_id, baseline_id, json.loads(third.stdout),
            "global-cap-third-failure")
        assert blocked_invalidation.returncode != 0
        assert "exhausted its two permitted headline invalidations" in \
            blocked_invalidation.stderr
        closed_global = run(root, "close-candidate", global_id,
                            "--reason", "global invalidation cap fixture complete")
        assert closed_global.returncode == 0, closed_global.stderr

        advance_source_commit("test: isolate unjournaled-run hypothesis")
        unjournaled_id = "unjournaled-run-attempt"
        initialized = run(root, "init", unjournaled_id, "A",
                          "--switch", "DS4_UNJOURNALED=0,1")
        assert initialized.returncode == 0, initialized.stderr
        configure_candidate(root, unjournaled_id, baseline_id, baseline, None)
        begun = run(root, "begin-pair", unjournaled_id)
        assert begun.returncode == 0, begun.stderr
        unjournaled_next = run(root, "begin-pair", unjournaled_id)
        assert unjournaled_next.returncode != 0
        assert "both journaled arms must bind their results" in \
            unjournaled_next.stderr
        closed_unjournaled = run(
            root, "close-candidate", unjournaled_id,
            "--reason", "unjournaled arm fixture complete")
        assert closed_unjournaled.returncode == 0, closed_unjournaled.stderr

        advance_source_commit("test: record prior feature hypothesis")
        prior_feature_id = "prior-lane-a-feature"
        initialized = run(root, "init", prior_feature_id, "A",
                          "--switch", "DS4_FEATURE=0,1")
        assert initialized.returncode == 0, initialized.stderr
        configure_candidate(root, prior_feature_id, baseline_id, baseline, None)
        closed_prior = run(root, "close-candidate", prior_feature_id,
                           "--reason", "prior formal feature attempt")
        assert closed_prior.returncode == 0, closed_prior.stderr

        advance_source_commit("test: isolate lane-a integration hypothesis")
        candidate_id = "lane-a-integration"
        initialized = run(root, "init", candidate_id, "A",
                          "--switch", "DS4_FEATURE=0,1")
        assert initialized.returncode == 0, initialized.stderr
        configure_candidate(root, candidate_id, baseline_id, baseline, None)
        registered_pairs = []
        for _ in range(5):
            begun = run(root, "begin-pair", candidate_id)
            assert begun.returncode == 0, begun.stderr
            registered = json.loads(begun.stdout)
            record_pair_runs(
                root, candidate_id, registered, baseline_id, baseline)
            registered_pairs.append(registered)
        mismatched_pairs = deepcopy(registered_pairs)
        mismatched_pairs[0]["runs"]["candidate"] += "-mismatch"
        mismatched_proof = build_lane_a_proof(
            root, baseline_id, baseline, candidate_id, mismatched_pairs,
            manifest_candidate_id=candidate_id)
        configure_candidate(
            root, candidate_id, baseline_id, baseline, mismatched_proof)
        mismatched = run(root, "check", candidate_id)
        assert mismatched.returncode != 0
        assert "differs from its journaled run/result artifacts" in mismatched.stderr
        try:
            build_lane_a_proof(
                root, baseline_id, baseline, candidate_id, registered_pairs,
                manifest_candidate_id="")
        except AssertionError as error:
            assert "does not match the proof candidate_id" in str(error)
        else:
            raise AssertionError("promotion proof accepted an empty candidate identity")
        proof = build_lane_a_proof(
            root, baseline_id, baseline, candidate_id, registered_pairs)
        configure_candidate(root, candidate_id, baseline_id, baseline, proof)
        checked = run(root, "check", candidate_id)
        assert checked.returncode == 0, checked.stderr
        prior_ids = [
            selection_id, producer_race_id, selection_bypass_id, absent_id,
            abort_id, retry_id, global_id, unjournaled_id, prior_feature_id,
        ]
        assert f"prior_formal_candidate_count={len(prior_ids)}" in checked.stdout
        assert f'"candidate_id":"{prior_feature_id}"' in checked.stdout
        promoted = run(root, "promote", candidate_id)
        assert promoted.returncode == 0, promoted.stderr
        promoted_record = json.loads(Path(promoted.stdout.splitlines()[-1]).read_text())
        assert promoted_record["schema_version"] == 2
        assert promoted_record["headline_invalidation_count"] == 0
        assert promoted_record["prior_formal_candidate_count"] == len(prior_ids)
        prior_records = promoted_record["prior_formal_candidates"]
        assert [item["candidate_id"] for item in prior_records] == prior_ids
        for item in prior_records:
            assert item["status"] == "closed"
            assert item["candidate_binary_sha256"] is None
            assert item["same_candidate_binary"] is None
            assert re.fullmatch(r"[0-9a-f]{40}", item["source_commit"])
            assert item["switch_contrast_match"] is (
                item["candidate_id"] == prior_feature_id)
        assert re.fullmatch(r"[0-9a-f]{64}",
                            promoted_record["candidate_binary_sha256"])
        duplicate = run(root, "promote", candidate_id)
        assert duplicate.returncode != 0
        assert "already promoted" in duplicate.stderr

        advance_source_commit("test: isolate abandoned hypothesis")
        closed_id = "abandoned-research-direction"
        assert run(root, "init", closed_id, "B", "--switch",
                   "DS4_ABANDONED=0,1").returncode == 0
        closed = run(root, "close-candidate", closed_id,
                     "--reason", "hypothesis disproved")
        assert closed.returncode == 0, closed.stderr
        rejected = run(root, "check", closed_id)
        assert rejected.returncode != 0
        assert "closed candidates" in rejected.stderr

        performance = deepcopy(baseline["thresholds"]["performance"])
        performance["minimum_gain"] = 0.005
        genesis_value = json.loads(genesis.read_text())
        amendment = {
            "schema_version": 2,
            "kind": "ds4-baseline-amendment",
            "amendment_id": "prospective-performance-policy",
            "baseline_id": baseline_id,
            "rationale": "Prospective practical-margin update for future candidates.",
            "add_oracle_generator": None,
            "threshold_updates": {"performance": performance},
            "calibration": None,
            "calibration_sha256": None,
            "timing_noise_qualification": genesis_value["artifacts"][
                "timing_noise_qualification"],
            "verifier": {
                "source_commit": current_head(),
                "sha256": GATE_MODULE.verifier_sha256(REPO),
            },
            "evidence": [],
        }
        amendment_path = root / "performance-amendment.json"
        amendment_path.write_text(json.dumps(
            amendment, indent=2, sort_keys=True) + "\n")
        bind_amendment_reviews(root, amendment_path)
        amended = run(root, "amend-baseline", str(amendment_path))
        assert amended.returncode == 0, amended.stderr
        amended_id = amended.stdout.strip()
        amended_path = root / "baselines" / "sha256" / f"{amended_id[7:]}.json"
        amended_record = json.loads(amended_path.read_text())
        assert amended_record["thresholds"]["performance"] == performance
        assert amended_record["provenance"]["amendment_kind"] == "performance-policy"
        assert amended_record["provenance"]["calibration"] == \
            baseline["provenance"]["calibration"]

        advance_source_commit("test: isolate policy observer hypothesis")
        observer_id = "open-policy-observer"
        assert run(root, "init", observer_id, "A", "--switch",
                   "DS4_OBSERVER=0,1").returncode == 0
        observer_path = root / "candidates" / observer_id / "candidate.json"
        observer = json.loads(observer_path.read_text())
        observer["baseline_id"] = amended_id
        observer_path.write_text(json.dumps(observer, indent=2, sort_keys=True) + "\n")
        blocked = deepcopy(amendment)
        blocked["amendment_id"] = "candidate-influenced-policy"
        blocked["baseline_id"] = amended_id
        blocked["threshold_updates"]["performance"]["minimum_gain"] = 0.0
        blocked["evidence"] = []
        blocked_path = root / "blocked-performance-amendment.json"
        blocked_path.write_text(json.dumps(blocked, indent=2, sort_keys=True) + "\n")
        bind_amendment_reviews(root, blocked_path)
        blocked_result = run(root, "amend-baseline", str(blocked_path))
        assert blocked_result.returncode != 0
        assert "cannot observe open candidate" in blocked_result.stderr

        successor = json.loads(json.dumps(baseline))
        successor["provenance"] = {
            "amendment_id": "fixture-successor",
            "amendment_kind": "governance-only",
            "replaces": baseline_id,
            "evidence": [],
        }
        successor_digest = canonical(successor)
        (baseline_path.parent / f"{successor_digest}.json").write_text(
            json.dumps(successor, indent=2, sort_keys=True) + "\n")
        advance_source_commit("test: isolate stale-baseline hypothesis")
        stale_id = "stale-baseline-candidate"
        assert run(root, "init", stale_id, "A", "--switch",
                   "DS4_STALE=0,1").returncode == 0
        stale_path = root / "candidates" / stale_id / "candidate.json"
        stale = json.loads(stale_path.read_text())
        stale["baseline_id"] = baseline_id
        stale_path.write_text(json.dumps(stale, indent=2, sort_keys=True) + "\n")
        stale_check = run(root, "check", stale_id)
        assert stale_check.returncode != 0
        assert "unjournaled record" in stale_check.stderr

    print("test_candidate_gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
