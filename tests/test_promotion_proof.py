#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "scripts" / "promotion-proof.py"
BASELINE_ID = "sha256:" + "a" * 64
FNV = "1234567890abcdef"
DEEPSEEK_PROMPT_SHA256 = \
    "6dff0f4bc6000881259d96b2126b9c4f86f377efbaaa349e0a49d6da0435d34b"
SOURCE = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
CONTROL_SOURCE = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD^"], text=True).strip()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def committed_source_digest(commit: str) -> str:
    return digest_bytes(subprocess.check_output(
        ["git", "-C", str(REPO), "show", f"{commit}:ds4_bench.c"]))


SOURCE_PRODUCER_SHA256 = committed_source_digest(SOURCE)
CONTROL_PRODUCER_SHA256 = committed_source_digest(CONTROL_SOURCE)


def sample_digest(path: Path) -> str:
    size = path.stat().st_size
    offsets = (0, max(0, size // 2 - 4 * 1024 * 1024),
               max(0, size - 8 * 1024 * 1024))
    result = hashlib.sha256(f"{size}\n".encode())
    with path.open("rb") as stream:
        for offset in offsets:
            stream.seek(offset)
            result.update(stream.read(8 * 1024 * 1024))
    return result.hexdigest()


def manifest(path: Path, values: dict[str, object]) -> None:
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()))


def make_run(root: Path, name: str, pair_id: str, order: str, arm: str, *,
             frontier: int, prefill: float, decode: float, fnv: str = FNV,
             prompt_sha: str = "f" * 64, model: Path | None = None,
             model_sample: str = "b" * 64, model_size: int = 10) -> dict[str, str]:
    csv_path = root / f"{name}.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, lineterminator="\n", fieldnames=[
            "ctx_tokens", "prefill_tokens", "prefill_tps", "gen_tokens",
            "gen_tps", "gen_first_ms", "gen_steady_tokens",
            "gen_steady_tps", "kvcache_bytes", "gen_cycles",
            "gen_token_fnv64",
        ])
        writer.writeheader()
        writer.writerow({
            "ctx_tokens": frontier, "prefill_tokens": frontier,
            "prefill_tps": prefill, "gen_tokens": 300, "gen_tps": decode,
            "gen_first_ms": 50, "gen_steady_tokens": 299,
            "gen_steady_tps": decode, "kvcache_bytes": 0,
            "gen_cycles": 300, "gen_token_fnv64": fnv,
        })
    manifest_path = csv_path.with_suffix(".manifest")
    time_prefix = "20260914T120000.00000000"
    arm_rank = 0 if (order == "AB" and arm == "control") or (
        order == "BA" and arm == "candidate") else 1
    run_id = f"{time_prefix}{arm_rank}Z-{name}"
    feature = 1 if arm == "candidate" else 0
    manifest(manifest_path, {
        "tag": name, "run_id": run_id, "source_commit": SOURCE,
        "source_dirty": 0, "bench_config_sha256": "c" * 64,
        "ds4_sha256": "d" * 64, "peer_ds4_sha256": "d" * 64,
        "ds4_bench_producer_source_sha256": SOURCE_PRODUCER_SHA256,
        "model": str(model or (root / "glm.gguf")), "model_arch": "glm5-next",
        "model_size": model_size,
        "model_sample_sha256": model_sample, "prompt_sha256": prompt_sha,
        "frontier": frontier, "generated_tokens": 300,
        "context": frontier + 512, "prefill_chunk": 2048,
        "prefill_batch": 256, "rdma_profile": "roce-v2",
        "coordinator_rdma_device": "mlx5_0",
        "worker_rdma_device": "mlx5_1", "rdma_gid_index": 3,
        "candidate": 1, "candidate_id": "fixture", "candidate_lane": "A",
        "baseline_id": BASELINE_ID,
        "expected_fnv64": FNV, "pair_id": pair_id, "pair_order": order,
        "pair_arm": arm, "dspark": 0,
        "worker_env": f"DS4_BENCH_RUN_ID={run_id} DS4_FEATURE={feature}",
        "coordinator_env": f"DS4_BENCH_RUN_ID={run_id} DS4_FEATURE={feature}",
        "common_env": f"DS4_BENCH_RUN_ID={run_id}",
        "extra_env": f"DS4_FEATURE={feature}",
        "toolchain_id": "rocm-fixture",
    })
    coordinator = root / f"coordinator-{name}.log"
    worker = root / f"worker-{name}.log"
    common = (
        f"ds4-tp: benchmark run_id={run_id}\n"
        "ds4-tp: rdma GID index 3 (RoCE v2)\n"
        "ds4-tp: mlx5 queue pair uses RC\n"
        "ds4-tp: mlx5 registered host slab as 3 MRs\n"
        "ds4: memory promotion: expanded_weight_cache_bytes=0\n"
        "ds4-bench: semantic suite passed cases=2\n"
        "ds4-tp: transport proof requested=rdma active=rdma "
        "payload_fallback_calls=0 failed=0\n"
    )
    coordinator.write_text(
        "ds4-tp: worker connected, transport=rdma\n"
        "ds4-tp: rdma device mlx5_0 (port state 4)\n" + common +
        "ds4-bench: headline_csv_complete=1\n"
        "ds4-bench-launcher: headline_csv_complete=1\n"
        "ds4-bench-launcher: worker_terminated_by_launcher=0\n")
    worker.write_text(
        "ds4-tp: leader connected, transport=rdma\n"
        "ds4-tp: rdma device mlx5_1 (port state 4)\n" + common)
    coordinator_status = root / f"coordinator-{name}.status"
    worker_status = root / f"worker-{name}.status"
    manifest(coordinator_status, {"exit_code": 0, "signal": 0})
    manifest(worker_status, {"exit_code": 0, "signal": 0})
    return {
        "csv": str(csv_path), "manifest": str(manifest_path),
        "coordinator_log": str(coordinator),
        "coordinator_status": str(coordinator_status),
        "worker_log": str(worker), "worker_status": str(worker_status),
    }


def make_pair(root: Path, name: str, order: str, *, frontier: int,
              prompt_sha: str = "f" * 64, fnv: str = FNV,
              model: Path | None = None, model_sample: str = "b" * 64,
              model_size: int = 10, speedup: float = 1.2) -> dict:
    control = make_run(root, f"{name}-0", name, order, "control",
                       frontier=frontier, prefill=100, decode=10, fnv=fnv,
                       prompt_sha=prompt_sha, model=model,
                       model_sample=model_sample, model_size=model_size)
    candidate = make_run(root, f"{name}-1", name, order, "candidate",
                         frontier=frontier, prefill=100 * speedup,
                         decode=10 * speedup, fnv=fnv, prompt_sha=prompt_sha,
                         model=model, model_sample=model_sample,
                         model_size=model_size)
    return {"pair_id": name, "order": order, "control": control,
            "candidate": candidate, "allowed_fields": [],
            "allowed_env": ["DS4_FEATURE"]}


def screen(pair: dict, control_fnv: str = FNV) -> dict:
    return {"pair": pair, "control_fnv64": control_fnv,
            "trajectory": {"mode": "exact"},
            "max_prefill_regression": 0.05, "max_decode_regression": 0.03}


def performance_contract() -> dict:
    return {
        "schema_version": 2,
        "method": "paired-log-ratio-two-tier-v2",
        "qualification_pairs": 3,
        "merge_looks": [5, 7, 9],
        "formal_test": {
            "kind": "repeated-student-v1",
            "boundary": 2.52,
            "familywise_alpha": 0.05,
            "design_calibration_sha256":
                "ed429ce50d58016e01a8276f2004f5c4777f896f1a23c98eb17f81c87cb5abde",
            "timing_noise_qualification_sha256": "e" * 64,
        },
        "futility_confidence_level": 0.95,
        "minimum_gain": 0.01,
        "maximum_untargeted_regression": 0.02,
        "maximum_control_regression": 0.05,
        "required_provider": "roce-v2",
        "absolute_floor": {"prefill": 0.0, "decode": 0.0},
        "screens": {
            "max_prefill_regression": 0.05,
            "max_decode_regression": 0.03,
        },
    }


def build_spec(root: Path, pair_count: int = 3,
               stage: str | None = None) -> Path:
    stage = stage or ("qualification" if pair_count == 3 else "promotion")
    (root / "glm.gguf").write_bytes(b"glm-model")
    q4 = root / "deepseek-q4.gguf"
    q2 = root / "deepseek-q2.gguf"
    q4.write_bytes(b"deepseek-q4")
    q2.write_bytes(b"deepseek-q2")
    q4_sample = sample_digest(q4)
    q2_sample = sample_digest(q2)
    pairs = [make_pair(root, f"headline-{index}",
                       "AB" if index % 2 else "BA", frontier=2048)
             for index in range(1, pair_count + 1)]
    diverse = make_pair(
        root, "diverse", "BA", frontier=4096,
        prompt_sha="24d19432acab4d4cd2971d938b3c013fcfad1010ed701218bc7bdc1b630ecfef")
    long_context = make_pair(root, "long", "AB", frontier=8192)
    for arm in ("control", "candidate"):
        values = dict(line.split("=", 1) for line in
                      Path(long_context[arm]["manifest"]).read_text().splitlines())
        values["prompt_sha256"] = \
            "e7c1a2cadf781d274cc26bd251d532fe1b9e632080da97e3eb4684741e7cc308"
        manifest(Path(long_context[arm]["manifest"]), values)
    q4_pair = make_pair(root, "q4", "BA", frontier=2048,
                        prompt_sha="6dff0f4bc6000881259d96b2126b9c4f86f377efbaaa349e0a49d6da0435d34b",
                        fnv="4444444444444444", model=q4,
                        model_sample=q4_sample, model_size=q4.stat().st_size,
                        speedup=1.0)
    q2_pair = make_pair(root, "q2", "AB", frontier=2048,
                        prompt_sha="6dff0f4bc6000881259d96b2126b9c4f86f377efbaaa349e0a49d6da0435d34b",
                        fnv="2222222222222222", model=q2,
                        model_sample=q2_sample, model_size=q2.stat().st_size,
                        speedup=1.0)
    value = {
        "schema_version": 2, "kind": "ds4-promotion-proof-spec",
        "stage": stage,
        "candidate_id": "fixture", "lane": "A", "baseline_id": BASELINE_ID,
        "baseline_fnv64": FNV,
        "source_commits": {"control": SOURCE, "candidate": SOURCE},
        "required_provider": "roce-v2",
        "performance": {
            "contract": performance_contract(),
            "target_metrics": ["prefill", "decode"],
            "candidate_switches": {
                "DS4_FEATURE": {"control": "0", "candidate": "1"},
            },
            "public_claim": False,
        },
        "headline_pairs": pairs, "diverse_screen": screen(diverse),
        "long_context_screen": (
            None if stage == "qualification" else screen(long_context)),
        "ordinary_regressions": [] if stage == "qualification" else [
            {"name": "deepseek-0731-q4", "baseline_fnv64": "4444444444444444",
             "model": {"path": str(q4), "size": q4.stat().st_size,
                       "sample_sha256": q4_sample,
                       "sha256": digest_bytes(q4.read_bytes()),
                       "quantization": "Q4_K"},
             "screen": screen(q4_pair, "4444444444444444")},
            {"name": "deepseek-0731-q2", "baseline_fnv64": "2222222222222222",
             "model": {"path": str(q2), "size": q2.stat().st_size,
                       "sample_sha256": q2_sample,
                       "sha256": digest_bytes(q2.read_bytes()),
                       "quantization": "Q2_K"},
             "screen": screen(q2_pair, "2222222222222222")},
        ],
    }
    path = root / "spec.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["DS4_RESEARCH_ROOT"] = str(root)
    return subprocess.run([sys.executable, str(TOOL), *args], cwd=REPO,
                          env=environment, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)


def set_pair_speedup(pair: dict, speedup: float) -> None:
    control_path = Path(pair["control"]["csv"])
    candidate_path = Path(pair["candidate"]["csv"])
    with control_path.open(newline="") as stream:
        control = next(csv.DictReader(stream))
    with candidate_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        candidate = next(reader)
    assert fieldnames is not None
    candidate["prefill_tps"] = str(float(control["prefill_tps"]) * speedup)
    candidate["gen_tps"] = str(float(control["gen_tps"]) * speedup)
    candidate["gen_steady_tps"] = candidate["gen_tps"]
    with candidate_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerow(candidate)


def create(root: Path, spec: Path, name: str = "proof.json") -> tuple[
        subprocess.CompletedProcess[str], Path]:
    proof = root / name
    return run(root, "create", "--spec", str(spec), "--output", str(proof)), proof


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec = build_spec(root)
        proof = root / "proof.json"
        created = run(root, "create", "--spec", str(spec), "--output", str(proof))
        assert created.returncode == 0, created.stderr
        checked = run(root, "verify", str(proof))
        assert checked.returncode == 0, checked.stderr
        value = json.loads(proof.read_text())
        assert value["performance"]["decision"] == "consistent-direction"
        assert value["performance"]["pairs"] == 3
        assert value["performance"]["merge_eligible"] is False
        assert value["stage"] == "qualification"
        assert value["long_context_screen"] is None
        assert value["ordinary_regressions"] == []

        set_pair_speedup(json.loads(spec.read_text())["headline_pairs"][2], 1.005)
        result, unqualified = create(root, spec, "unqualified.json")
        assert result.returncode == 1
        assert "needs-two-more-pairs" in result.stderr
        unqualified_value = json.loads(unqualified.read_text())
        assert unqualified_value["performance"]["decision"] == "needs-two-more-pairs"
        assert unqualified_value["performance"]["passed"] is False
        assert unqualified_value["performance"]["metrics"]["prefill"][
            "qualification_consistent_direction"] is False

        control_log = Path(value["headline_pairs"][0]["control"]["artifacts"]
                           ["coordinator_log"]["path"])
        control_log.write_text(control_log.read_text().replace(
            "payload_fallback_calls=0", "payload_fallback_calls=1"))
        tampered = run(root, "verify", str(proof))
        assert tampered.returncode == 1
        assert "hash mismatch" in tampered.stderr

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec = build_spec(root)
        value = json.loads(spec.read_text())
        run_spec = value["headline_pairs"][0]["control"]
        coordinator_log = Path(run_spec["coordinator_log"])
        coordinator_log.write_text("\n".join(
            line for line in coordinator_log.read_text().splitlines()
            if not line.startswith("ds4-tp: benchmark run_id=")) + "\n")
        result, _ = create(root, spec)
        assert result.returncode == 1
        assert "does not bind benchmark run ID" in result.stderr

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec = build_spec(root)
        value = json.loads(spec.read_text())
        manifest_path = Path(value["headline_pairs"][0]["control"]["manifest"])
        fields = dict(line.split("=", 1) for line in
                      manifest_path.read_text().splitlines())
        fields["worker_env"] = fields["worker_env"].replace(
            fields["run_id"], "different-run-id")
        manifest(manifest_path, fields)
        result, _ = create(root, spec)
        assert result.returncode == 1
        assert "worker_env does not bind its run_id" in result.stderr

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec = build_spec(root)
        value = json.loads(spec.read_text())
        manifest_path = Path(value["headline_pairs"][0]["control"]["manifest"])
        fields = dict(line.split("=", 1) for line in
                      manifest_path.read_text().splitlines())
        fields["candidate_id"] = "post-hoc-candidate"
        manifest(manifest_path, fields)
        result, _ = create(root, spec)
        assert result.returncode == 1
        assert "does not match the proof candidate_id" in result.stderr

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec = build_spec(root)
        value = json.loads(spec.read_text())
        value["performance"]["public_claim"] = True
        spec.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        result = run(root, "create", "--spec", str(spec),
                     "--output", str(root / "proof.json"))
        assert result.returncode == 1
        assert "at least five matched pairs" in result.stderr

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec = build_spec(root, pair_count=5)
        value = json.loads(spec.read_text())
        for pair, speedup in zip(value["headline_pairs"],
                                 [1.2, 1.2, 1.005, 1.2, 1.2]):
            set_pair_speedup(pair, speedup)
        result, proof = create(root, spec)
        assert result.returncode == 0, result.stderr
        promoted = json.loads(proof.read_text())
        assert promoted["performance"]["decision"] == "pass"
        assert promoted["performance"]["merge_eligible"] is True

    for count in (5, 7, 9):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = build_spec(root, pair_count=count)
            result, proof = create(root, spec)
            assert result.returncode == 0, result.stderr
            performance = json.loads(proof.read_text())["performance"]
            assert performance["decision"] == "pass"
            assert performance["formal_test"]["boundary"] == 2.52
            assert all(metric["boundary"] == 2.52
                       for metric in performance["metrics"].values())

    for count, expected_decision in (
            (5, "needs-two-more-pairs"),
            (7, "needs-two-more-pairs"),
            (9, "not-demonstrated")):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = build_spec(root, pair_count=count)
            value = json.loads(spec.read_text())
            for pair in value["headline_pairs"]:
                set_pair_speedup(pair, 1.01)
            result, proof = create(root, spec)
            assert result.returncode == 1
            performance = json.loads(proof.read_text())["performance"]
            assert performance["decision"] == expected_decision
            assert performance["merge_eligible"] is False
            for metric in performance["metrics"].values():
                assert abs(metric["one_sided_lower"] - 0.01) < 1e-12
                assert metric["passed"] is False

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec = build_spec(root, pair_count=9)
        value = json.loads(spec.read_text())
        for pair in value["headline_pairs"]:
            set_pair_speedup(pair, 1.0)
        result, proof = create(root, spec)
        assert result.returncode == 1
        assert "not-demonstrated" in result.stderr
        performance = json.loads(proof.read_text())["performance"]
        assert performance["decision"] == "not-demonstrated"
        assert performance["merge_eligible"] is False

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec = build_spec(root, pair_count=9)
        value = json.loads(spec.read_text())
        value["performance"]["contract"]["formal_test"] = {
            "kind": "exact-sign-fixed-nine-v1",
            "pairs": 9,
            "minimum_positive": 8,
            "ties": "fail",
            "familywise_alpha": 0.05,
        }
        for pair, speedup in zip(value["headline_pairs"], [1.2] * 8 + [1.0]):
            set_pair_speedup(pair, speedup)
        spec.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        result, proof = create(root, spec)
        assert result.returncode == 0, result.stderr
        performance = json.loads(proof.read_text())["performance"]
        assert performance["decision"] == "pass"
        assert all(metric["sign_test"]["positive_adjusted_effects"] == 8
                   for metric in performance["metrics"].values())

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec = build_spec(root, pair_count=5)
        value = json.loads(spec.read_text())
        value["headline_pairs"][1] = make_pair(
            root, "nonalternating", "AB", frontier=2048)
        spec.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        result, _ = create(root, spec)
        assert result.returncode == 1
        assert "strictly alternate" in result.stderr

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec = build_spec(root)
        value = json.loads(spec.read_text())
        value["headline_pairs"][0]["allowed_fields"].append("prompt_sha256")
        spec.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        result, _ = create(root, spec)
        assert result.returncode == 1
        assert "not an approved build" in result.stderr

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec = build_spec(root)
        value = json.loads(spec.read_text())
        candidate_manifest = Path(value["headline_pairs"][0]["candidate"]["manifest"])
        fields = dict(line.split("=", 1) for line in
                      candidate_manifest.read_text().splitlines())
        for name in ("worker_env", "coordinator_env", "extra_env"):
            fields[name] += " DS4_UNDECLARED=1"
        manifest(candidate_manifest, fields)
        result, _ = create(root, spec)
        assert result.returncode == 1
        assert "outside registered switches" in result.stderr

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec = build_spec(root)
        value = json.loads(spec.read_text())
        candidate_manifest = Path(
            value["headline_pairs"][0]["candidate"]["manifest"])
        fields = dict(line.split("=", 1) for line in
                      candidate_manifest.read_text().splitlines())
        fields["ds4_bench_producer_source_sha256"] = "0" * 64
        manifest(candidate_manifest, fields)
        result, _ = create(root, spec)
        assert result.returncode == 1
        assert "producer binary does not match" in result.stderr

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec = build_spec(root, pair_count=5)
        value = json.loads(spec.read_text())
        value["long_context_screen"]["pair"] = make_pair(
            root, "short-long", "AB", frontier=4096)
        spec.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        result = run(root, "create", "--spec", str(spec),
                     "--output", str(root / "proof.json"))
        assert result.returncode == 1
        assert "at least 8192" in result.stderr

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec = build_spec(root, pair_count=5)
        value = json.loads(spec.read_text())
        value["source_commits"]["control"] = CONTROL_SOURCE
        for pair in [*value["headline_pairs"],
                     value["diverse_screen"]["pair"],
                     value["long_context_screen"]["pair"],
                     *(item["screen"]["pair"] for item in
                       value["ordinary_regressions"])]:
            pair["allowed_fields"].append("source_commit")
            pair["allowed_fields"].append(
                "ds4_bench_producer_source_sha256")
            control_manifest = Path(pair["control"]["manifest"])
            fields = dict(line.split("=", 1) for line in
                          control_manifest.read_text().splitlines())
            fields["source_commit"] = CONTROL_SOURCE
            fields["ds4_bench_producer_source_sha256"] = \
                CONTROL_PRODUCER_SHA256
            manifest(control_manifest, fields)
        spec.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        result = run(root, "create", "--spec", str(spec),
                     "--output", str(root / "proof.json"))
        assert result.returncode == 0, result.stderr

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec = build_spec(root)
        proof = root / "proof.json"
        assert run(root, "create", "--spec", str(spec),
                   "--output", str(proof)).returncode == 0
        value = json.loads(proof.read_text())
        value["spec_sha256"] = "0" * 64
        proof.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        result = run(root, "verify", str(proof))
        assert result.returncode == 1
        assert "spec digest" in result.stderr

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec = build_spec(root, pair_count=5)
        value = json.loads(spec.read_text())
        value["ordinary_regressions"][0]["model"]["sha256"] = "0" * 64
        spec.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        result = run(root, "create", "--spec", str(spec),
                     "--output", str(root / "proof.json"))
        assert result.returncode == 1
        assert "model bytes" in result.stderr

    for mutation, expected in (
            (lambda pair: pair.update({"prompt": "wrong"}), "wrong frozen prompt"),
            (lambda pair: None, "must use frontier 2048")):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = build_spec(root, pair_count=5)
            value = json.loads(spec.read_text())
            pair = value["ordinary_regressions"][0]["screen"]["pair"]
            if expected == "wrong frozen prompt":
                for arm in ("control", "candidate"):
                    path = Path(pair[arm]["manifest"])
                    fields = dict(line.split("=", 1) for line in
                                  path.read_text().splitlines())
                    fields["prompt_sha256"] = "0" * 64
                    manifest(path, fields)
            else:
                pair = make_pair(
                    root, "q4-wrong-frontier", "BA", frontier=1024,
                    prompt_sha=DEEPSEEK_PROMPT_SHA256,
                    fnv="4444444444444444",
                    model=Path(value["ordinary_regressions"][0]["model"]["path"]),
                    model_sample=value["ordinary_regressions"][0]["model"][
                        "sample_sha256"],
                    model_size=value["ordinary_regressions"][0]["model"]["size"],
                    speedup=1.0)
                value["ordinary_regressions"][0]["screen"]["pair"] = pair
            spec.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
            result, _ = create(root, spec)
            assert result.returncode == 1
            assert expected in result.stderr, result.stderr

    print("test_promotion_proof: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
