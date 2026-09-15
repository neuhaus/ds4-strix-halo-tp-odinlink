#!/usr/bin/env python3
"""Exercise append-only baseline genesis and its evidence bindings."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path


if os.environ.get("DS4_GATE_CLEAN_TEST") != "1":
    raise SystemExit("run via tests/test_baseline_genesis.sh")


REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "scripts" / "candidate-gate.py"
sys.path.insert(0, str(REPO / "scripts"))
from ds4_gate_controls import verifier_sha256  # noqa: E402
GLM_TP_LAYOUT = {
    "kind": "q4k-ffn-intermediate",
    "intermediate_size": 2048,
    "shards": [1024, 1024],
    "expert_count": 288,
    "experts_used": 8,
    "reduction": {
        "op": "sum", "scope": "all-ranks", "count": 42,
        "width": 4096, "dtype": "f32",
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def committed_digest(commit: str, relative: str) -> str:
    content = subprocess.check_output([
        "git", "-C", str(REPO), "cat-file", "blob", f"{commit}:{relative}",
    ])
    return hashlib.sha256(content).hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sampled(path: Path) -> tuple[int, str]:
    size = path.stat().st_size
    result = hashlib.sha256(f"{size}\n".encode())
    with path.open("rb") as stream:
        for offset in (0, max(0, size // 2 - 4 * 1024 * 1024),
                       max(0, size - 8 * 1024 * 1024)):
            stream.seek(offset)
            result.update(stream.read(8 * 1024 * 1024))
    return size, result.hexdigest()


def bind_reviews(root: Path, genesis: Path) -> None:
    value = json.loads(genesis.read_text())
    reviewable = deepcopy(value)
    reviewable["evidence"] = [
        item for item in reviewable.get("evidence", [])
        if item.get("kind") not in {"fable-review", "grok-review"}
    ]
    payload_sha = canonical(reviewable)
    reviews = []
    for reviewer in ("fable", "grok"):
        path = root / f"{reviewer}-review.md"
        path.write_text(
            "DS4-REVIEW-SCHEMA: 1\n"
            f"REVIEWER: {reviewer}\n"
            f"REVIEWED-PAYLOAD-SHA256: {payload_sha}\n"
            f"CALIBRATION-SHA256: {value['calibration_sha256']}\n"
            "VERDICT: GO\n"
            "Fixture review bound to the exact canonical payload.\n")
        reviews.append({
            "kind": f"{reviewer}-review",
            "path": str(path),
            "sha256": digest(path),
        })
    value["evidence"] = reviewable["evidence"] + reviews
    genesis.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_manifest(path: Path, values: dict[str, object]) -> None:
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()))


def v2_numerical_thresholds() -> dict:
    return {
        "schema_version": 2,
        "min_teacher_steps": 300,
        "allow_quality_difference": False,
        "decision": {
            "e_bound": 0.1,
            "confidence_level": 0.95,
            "max_near_tie_cluster_rate_upper": 0.25,
        },
        "distribution": {
            "bootstrap_method": "bca",
            "bootstrap_resamples": 1000,
            "bootstrap_seed": 11,
            "cluster_mode": "case-or-contiguous-block",
            "block_size": 30,
            "min_clusters": 5,
            "max_mean_kl_upper": 0.1,
            "max_mean_tvd_upper": 0.1,
            "max_mean_teacher_nll_delta_upper": 0.1,
            "min_same_top1_cluster_rate_lower": 0.7,
            "soft_limits": {
                "centered_p99_abs": 0.1,
                "centered_nrms": 0.1,
                "kl": 0.1,
                "tvd": 0.1,
            },
            "max_soft_exceedance_cluster_rate_upper": 0.25,
        },
        "safety": {
            "max_centered_abs": 2.0,
            "max_centered_nrms": 2.0,
            "max_kl": 2.0,
            "max_tvd": 1.0,
            "max_abs_teacher_nll_delta": 2.0,
        },
    }


def v2_quality_thresholds() -> dict:
    return {
        "schema_version": 2,
        "min_cases": 100,
        "min_target_tokens": 2289,
        "bootstrap": {
            "method": "bca", "resamples": 1000, "seed": 19,
            "nll_confidence_level": 0.95,
            "api_confidence_level": 0.95,
        },
        "nll": {"max_delta_upper": 0.02, "max_case_delta": 0.1},
        "api": {
            "required": True, "min_cases": 100,
            "min_top1_delta_lower": 0.0,
            "min_pair_delta_lower": 0.0,
        },
    }


def v2_performance_thresholds(noise_sha256: str = "e" * 64) -> dict:
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
            "timing_noise_qualification_sha256": noise_sha256,
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


def baseline_scope(record: dict) -> str:
    key = record["key"]
    return canonical({
        "model_sha256": key["model_sha256"],
        "model_size": key["model_size"],
        "quantization": key["quantization"],
        "architecture": key["architecture"],
        "tp_degree": key["tp_degree"],
        "tp_layout": {"tp_layout": key["tp_layout"]},
        "decode_mode": key["decode_mode"],
        "workload_id": key["workload_id"],
        "workload": key["workload"],
        "toolchain_family": key.get("toolchain_family", "unspecified"),
    })


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": digest(path)}


def make_timing_noise(root: Path, record: dict, model: Path) -> Path:
    key = record["key"]
    workload = key["workload"]
    provider = "roce-v2"
    binary = "b" * 64
    bench_binary = "d" * 64
    producer_source = committed_digest(key["source_commit"], "ds4_bench.c")
    effects = [0.0020, -0.0015, -0.0020, 0.0015, 0.0010,
               -0.0005, -0.0010, 0.0005, 0.0]
    decode_effects = effects[3:] + effects[:3]
    pairs = []
    common_log = (
        "ds4-tp: rdma GID index 3 (RoCE v2)\n"
        "ds4-tp: mlx5 queue pair uses RC\n"
        "ds4-tp: mlx5 registered host slab as 3 MRs\n"
        "ds4: memory promotion: expanded_weight_cache_bytes=0\n"
        "ds4-tp: transport proof requested=rdma active=rdma "
        "payload_fallback_calls=0 failed=0\n"
    )
    for pair_index, (prefill_effect, decode_effect) in enumerate(
            zip(effects, decode_effects), 1):
        pair_id = f"noise-{pair_index}"
        order = "AB" if pair_index % 2 else "BA"
        runs = {}
        for arm in ("control", "candidate"):
            suffix = 0 if arm == "control" else 1
            prefill = 200.0 if arm == "control" else 200.0 * math.exp(prefill_effect)
            decode = 19.0 if arm == "control" else 19.0 * math.exp(decode_effect)
            name = f"{pair_id}-{arm}"
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
                    "ctx_tokens": workload["frontier"],
                    "prefill_tokens": workload["frontier"],
                    "prefill_tps": prefill, "gen_tokens": workload["generated_tokens"],
                    "gen_tps": decode, "gen_first_ms": 60,
                    "gen_steady_tokens": 299, "gen_steady_tps": decode,
                    "kvcache_bytes": 0, "gen_cycles": 300,
                    "gen_token_fnv64": record["reference"]["fnv64"],
                })
            manifest_path = csv_path.with_suffix(".manifest")
            chronological = (2 * pair_index + suffix if order == "AB" else
                             2 * pair_index + 1 - suffix)
            run_id = f"20260914T000000.{chronological:09d}Z-{name}"
            manifest_values = {
                "tag": name,
                "run_id": run_id,
                "source_commit": key["source_commit"], "source_dirty": 0,
                "ds4_sha256": binary, "peer_ds4_sha256": binary,
                "ds4_bench_tp_sha256": bench_binary,
                "ds4_bench_producer_source_sha256": producer_source,
                "model": str(model), "model_arch": "glm5-next",
                "model_size": key["model_size"],
                "model_sample_sha256": key["model_sample_sha256"],
                "toolchain_id": key["toolchain_id"],
                "prompt_sha256": workload["prompt_sha256"],
                "frontier": workload["frontier"],
                "generated_tokens": workload["generated_tokens"],
                "context": workload["context"],
                "prefill_chunk": workload["prefill_chunk"],
                "rdma_profile": provider,
                "coordinator_rdma_device": "mlx5_0",
                "worker_rdma_device": "mlx5_1", "rdma_gid_index": 3,
                "candidate": 1, "candidate_lane": "A",
                "baseline_id": "GENESIS",
                "expected_fnv64": record["reference"]["fnv64"],
                "pair_id": pair_id, "pair_order": order, "pair_arm": arm,
                "dspark": 0,
                "common_env": f"DS4_BENCH_RUN_ID={run_id}",
                "worker_env": f"DS4_BENCH_RUN_ID={run_id}",
                "coordinator_env": f"DS4_BENCH_RUN_ID={run_id}",
                "extra_env": "",
                "tp_weight_layout": "q4k-ffn-intermediate",
                "tp_intermediate_size": "2048",
                "tp_intermediate_shards": "1024/1024",
                "tp_expert_count": "288", "tp_experts_used": "8",
                "tp_reduce_op": "sum", "tp_reduce_scope": "all-ranks",
                "tp_reduce_count": "42", "tp_reduce_width": "4096",
                "tp_reduce_dtype": "f32",
            }
            write_manifest(manifest_path, manifest_values)
            coordinator = root / f"coordinator-{name}.log"
            worker = root / f"worker-{name}.log"
            coordinator.write_text(
                "ds4: GLM5 compact Q4_K K-shard active: rank=0 layers=42 rows=0:1024 down-bytes=0:576\n"
                "ds4-tp: worker connected, transport=rdma\n"
                "ds4-tp: rdma device mlx5_0 (port state 4)\n"
                f"ds4-tp: benchmark run_id={run_id}\n" + common_log)
            worker.write_text(
                "ds4: GLM5 compact Q4_K K-shard active: rank=1 layers=42 rows=1024:2048 down-bytes=576:1152\n"
                "ds4-tp: leader connected, transport=rdma\n"
                "ds4-tp: rdma device mlx5_1 (port state 4)\n"
                f"ds4-tp: benchmark run_id={run_id}\n" + common_log)
            status = root / f"worker-{name}.status"
            write_manifest(status, {"exit_code": 0, "signal": 0})
            runs[arm] = {
                "csv": artifact(csv_path), "manifest": artifact(manifest_path),
                "coordinator_log": artifact(coordinator),
                "worker_log": artifact(worker), "worker_status": artifact(status),
            }
        pairs.append({"pair_id": pair_id, "order": order, **runs})
    path = root / "timing-noise-qualification.json"
    path.write_text(json.dumps({
        "schema_version": 2, "kind": "ds4-timing-noise-qualification",
        "scope_sha256": baseline_scope(record),
        "formal_test": "repeated-student-v1", "provider": provider,
        "source_commit": key["source_commit"], "ds4_sha256": binary,
        "model": artifact(model), "pairs": pairs,
    }, indent=2, sort_keys=True) + "\n")
    return path


def run_command(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["DS4_RESEARCH_ROOT"] = str(root)
    return subprocess.run(
        [sys.executable, str(GATE), *arguments], cwd=REPO, env=environment,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def register_calibration(root: Path, record: dict, numerical_dir: Path,
                         quality_path: Path) -> tuple[dict, str]:
    commits = [subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", revision], text=True).strip()
               for revision in ("HEAD", "HEAD^")]
    key = record["key"]
    scope = canonical({
        "model_sha256": key["model_sha256"],
        "model_size": key["model_size"],
        "quantization": key["quantization"],
        "architecture": key["architecture"],
        "tp_degree": key["tp_degree"],
        "tp_layout": {"tp_layout": key["tp_layout"]},
        "decode_mode": key["decode_mode"],
        "workload_id": key["workload_id"],
        "workload": key["workload"],
        "toolchain_family": "unspecified",
    })

    legal_artifacts = []
    for index, commit in enumerate(commits):
        logits = root / f"control-legal-{index}-logits"
        shutil.copytree(numerical_dir, logits)
        manifest = logits / "manifest"
        values = {
            line.partition("=")[0]: line.partition("=")[2]
            for line in manifest.read_text().splitlines() if "=" in line
        }
        values["source_commit"] = commit
        write_manifest(manifest, values)
        quality = root / f"control-legal-{index}.tsv"
        shutil.copyfile(quality_path, quality)
        quality_manifest = quality.with_suffix(".manifest")
        values = {
            line.partition("=")[0]: line.partition("=")[2]
            for line in quality_path.with_suffix(".manifest").read_text().splitlines()
            if "=" in line
        }
        values["source_commit"] = commit
        write_manifest(quality_manifest, values)
        legal_artifacts.append((commit, logits, quality))

    negative_logits = root / "control-negative-logits"
    shutil.copytree(legal_artifacts[0][1], negative_logits)
    first = negative_logits / "decode_000000.logits.json"
    value = json.loads(first.read_text())
    value["logits"] = [0.0, 1.0, 4.0, 3.0]
    first.write_text(json.dumps(value))
    negative_quality = root / "control-negative-quality.tsv"
    shutil.copyfile(legal_artifacts[0][2], negative_quality)
    rows = list(csv.DictReader(negative_quality.read_text().splitlines(), delimiter="\t"))
    with negative_quality.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        for row in rows:
            row["nll"] = str(float(row["nll"]) + int(row["target_tokens"]))
            row["avg_nll"] = str(float(row["avg_nll"]) + 1.0)
            writer.writerow(row)
    shutil.copyfile(legal_artifacts[0][2].with_suffix(".manifest"),
                    negative_quality.with_suffix(".manifest"))

    def register(name: str, role: str, commit: str, comparisons: dict,
                 expected: dict) -> str:
        descriptor = root / f"{name}-descriptor.json"
        descriptor.write_text(json.dumps({
            "schema_version": 2, "kind": "ds4-gate-control",
            "control_id": name, "scope_sha256": scope, "role": role,
            "source_commit": commit, "comparisons": comparisons,
            "expected_failures": expected, "notes": "genesis integration fixture",
        }, indent=2, sort_keys=True) + "\n")
        result = run_command(root, "register-control", str(descriptor))
        if result.returncode != 0:
            raise AssertionError(result.stderr)
        return result.stdout.strip()

    def legal_comparisons(index: int) -> dict:
        return {
            "numerical": {
                "reference_dir": str(numerical_dir),
                "candidate_dir": str(legal_artifacts[index][1]),
                "allow_quality_difference": False,
            },
            "quality": {
                "reference": str(quality_path),
                "candidate": str(legal_artifacts[index][2]),
            },
        }

    calibration = {
        "self_repeat": [register(
            "self-repeat", "self-repeat", commits[0], legal_comparisons(0), {})],
        "positive": [
            register("positive-a", "positive", commits[0], legal_comparisons(0), {}),
            register("positive-b", "positive", commits[1], legal_comparisons(1), {}),
        ],
        "holdout": [register(
            "holdout", "holdout", commits[1], legal_comparisons(1), {})],
        "negative": [
            register("negative-far-margin", "negative", commits[0], {
                "numerical": {
                    "reference_dir": str(numerical_dir),
                    "candidate_dir": str(negative_logits),
                    "allow_quality_difference": False,
                }}, {"numerical": "far-margin"}),
            register("negative-quality", "negative", commits[0], {
                "quality": {
                    "reference": str(quality_path),
                    "candidate": str(negative_quality),
                }}, {"quality": "quality-nll"}),
        ],
    }
    request = root / "calibration-request.json"
    request.write_text(json.dumps({
        "schema_version": 1, "kind": "ds4-calibration-request",
        "scope_sha256": scope, "previous_thresholds": None,
        "proposed_thresholds": {
            "numerical": record["thresholds"]["numerical"],
            "quality": record["thresholds"]["quality"],
        },
        "calibration": calibration,
    }, indent=2, sort_keys=True) + "\n")
    result = run_command(root, "calibrate", str(request))
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return calibration, json.loads(result.stdout)["calibration_sha256"]


def build_fixture(root: Path, structured: bool = True,
                  providers: tuple[str, ...] = ("roce-v2", "odinlink")) -> Path:
    model = root / "model.gguf"
    model.write_bytes(b"synthetic model")
    model_size, model_sample = sampled(model)
    model_sha = digest(model)
    source_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    producer_source = committed_digest(source_commit, "ds4_bench.c")
    token_file = root / "frozen.tokens"
    token_file.write_text("1\n" * 300)
    token_sha = digest(token_file)
    prompt_sha = "a" * 64
    toolchain = "fixture-rocm-7.14"
    fnv = "1234567890abcdef"

    benchmarks: list[dict[str, str]] = []
    for provider in providers:
        for index in range(3):
            name = f"{provider}-{index + 1}"
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
                    "ctx_tokens": 2048, "prefill_tokens": 2048,
                    "prefill_tps": 200 + index, "gen_tokens": 300,
                    "gen_tps": 19 + index / 100, "gen_first_ms": 60,
                    "gen_steady_tokens": 299, "gen_steady_tps": 19,
                    "kvcache_bytes": 0, "gen_cycles": 300,
                    "gen_token_fnv64": fnv,
                })
            manifest = csv_path.with_suffix(".manifest")
            run_id = f"{name}-run-id"
            manifest_values = {
                "tag": name, "run_id": run_id,
                "source_commit": source_commit, "source_dirty": 0,
                "model_size": model_size, "model_sample_sha256": model_sample,
                "toolchain_id": toolchain, "prompt_sha256": prompt_sha,
                "frontier": 2048, "generated_tokens": 300, "context": 4096,
                "prefill_chunk": 2048, "dspark": 0, "rdma_profile": provider,
                "coordinator_rdma_device": "mlx5_0",
                "worker_rdma_device": "mlx5_1", "rdma_gid_index": 3,
                "ds4_sha256": "b" * 64, "peer_ds4_sha256": "b" * 64,
                "ds4_bench_tp_sha256": "d" * 64,
                "ds4_bench_producer_source_sha256": producer_source,
                "common_env": f"DS4_BENCH_RUN_ID={run_id}",
                "worker_env": f"DS4_BENCH_RUN_ID={run_id}",
                "coordinator_env": f"DS4_BENCH_RUN_ID={run_id}",
                "extra_env": "",
            }
            if structured:
                manifest_values.update({
                    "tp_weight_layout": "q4k-ffn-intermediate",
                    "tp_intermediate_size": "2048",
                    "tp_intermediate_shards": "1024/1024",
                    "tp_expert_count": "288", "tp_experts_used": "8",
                    "tp_reduce_op": "sum", "tp_reduce_scope": "all-ranks",
                    "tp_reduce_count": "42", "tp_reduce_width": "4096",
                    "tp_reduce_dtype": "f32",
                })
            write_manifest(manifest, manifest_values)
            benchmark = {
                "path": str(csv_path), "sha256": digest(csv_path),
                "manifest_sha256": digest(manifest),
            }
            for rank, role, device in ((0, "coordinator", "mlx5_0"),
                                       (1, "worker", "mlx5_1")):
                log = root / f"{role}-{name}.log"
                status = log.with_suffix(".status")
                connected = "worker" if rank == 0 else "leader"
                log.write_text(
                    f"ds4-tp: {connected} connected, transport=rdma\n"
                    f"ds4-tp: benchmark run_id={run_id}\n"
                    f"ds4-tp: rdma device {device} (port state 4)\n"
                    "ds4-tp: rdma GID index 3 (RoCE v2)\n"
                    "ds4-tp: mlx5 queue pair uses RC\n"
                    "ds4-tp: mlx5 registered host slab as 3 MRs\n"
                    '{"fallback_calls":0}\n'
                    "ds4: memory promotion: expanded_weight_cache_bytes=0\n"
                    "ds4-tp: transport proof requested=rdma active=rdma "
                    "payload_fallback_calls=0 failed=0\n"
                    f"ds4: GLM5 compact Q4_K K-shard active: rank={rank} layers=42 "
                    f"rows={rank * 1024}:{(rank + 1) * 1024} "
                    f"down-bytes={rank * 576}:{(rank + 1) * 576}\n")
                write_manifest(status, {"exit_code": 0, "signal": 0})
                benchmark[f"{role}_log"] = artifact(log)
                benchmark[f"{role}_status"] = artifact(status)
            benchmarks.append(benchmark)

    numerical_dir = root / "numerical"
    numerical_dir.mkdir()
    numerical_files = []
    for index in range(300):
        path = numerical_dir / f"decode_{index:06d}.logits.json"
        value = {
            "source": "ds4-bench-frozen-teacher", "model": str(model),
            "backend": "rocm", "quality": False, "dspark": False,
            "dspark_strict": False, "quant_bits": 4,
            "prefix_tokens": 2048, "decode_step": index,
            "position": 2048 + index, "vocab": 4, "teacher_token": 3,
            "teacher_logit": 3.0, "argmax_id": 3, "argmax_logit": 3.0,
            "runner_up_id": 2, "runner_up_logit": 2.0,
            "top1_margin": 1.0, "teacher_gap": 0.0,
            "logits": [0.0, 1.0, 2.0, 3.0],
        }
        path.write_text(json.dumps(value))
        numerical_files.append({"name": path.name, "sha256": digest(path)})
    numerical_manifest = numerical_dir / "manifest"
    write_manifest(numerical_manifest, {
        "model_size": model_size, "model_sample_sha256": model_sample,
        "source_commit": source_commit, "source_dirty": 0,
        "toolchain_id": toolchain, "prefix_tokens": 2048,
        "file_count": 300, "frozen_token_sha256": token_sha, "dspark": 0,
    })

    quality_path = root / "quality.tsv"
    fields = ["id", "target_tokens", "nll", "avg_nll", "api_top1_count",
              "api_top1_match", "api_pair_total", "api_pair_agree"]
    with quality_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index in range(100):
            writer.writerow({
                "id": f"case-{index}", "target_tokens": 24, "nll": 12,
                "avg_nll": 0.5, "api_top1_count": 24,
                "api_top1_match": 22, "api_pair_total": 24,
                "api_pair_agree": 22,
            })
    quality_manifest = root / "quality.manifest"
    write_manifest(quality_manifest, {
        "model_size": model_size, "model_sample_sha256": model_sample,
        "quality_input_sha256": "b" * 64,
        "source_commit": source_commit, "source_dirty": 0, "dspark": 0,
    })
    numerical_thresholds = v2_numerical_thresholds()
    quality_thresholds = v2_quality_thresholds()
    quality_ref = {"sha256": digest(quality_path),
                   "manifest_sha256": digest(quality_manifest)}
    record = {
        "schema_version": 2 if structured else 1,
        "kind": "ds4-numerical-baseline",
        "key": {
            "model_sample_sha256": model_sample, "model_sha256": model_sha,
            "model_size": model_size, "quantization": "Q4_K",
            "source_commit": source_commit, "toolchain_id": toolchain,
            "architecture": "gfx1151", "tp_degree": 2,
            "decode_mode": "ordinary-greedy",
            "workload_id": "ds4-bench-tp-2048x300",
            "workload": {
                "prompt_sha256": prompt_sha, "frontier": "2048",
                "generated_tokens": "300", "context": "4096",
                "prefill_chunk": "2048", "dspark": "0",
                "frozen_token_sha256": token_sha,
            },
            "rdma_providers": list(providers),
            "tp_layout": GLM_TP_LAYOUT,
        },
        "reference": {
            "fnv64": fnv,
            "numerical": {"files": numerical_files,
                          "manifest_sha256": digest(numerical_manifest)},
            "quality": quality_ref, "quality_anchor": quality_ref,
        },
        "thresholds": {"numerical": numerical_thresholds,
                       "quality": quality_thresholds},
    }
    timing_noise = make_timing_noise(root, record, model)
    record["thresholds"]["performance"] = v2_performance_thresholds(
        digest(timing_noise))
    if not structured:
        record["key"].pop("tp_layout")
        record["key"]["expert_split"] = "128/128"
    calibration, calibration_sha256 = register_calibration(
        root, record, numerical_dir, quality_path)
    genesis = {
        "schema_version": 2, "kind": "ds4-baseline-genesis",
        "genesis_id": "fixture-rocm714", "rationale": "test fixture",
        "record": record,
        "artifacts": {
            "model_path": str(model), "benchmarks": benchmarks,
            "frozen_token_file": str(token_file),
            "numerical_dir": str(numerical_dir),
            "quality_tsv": str(quality_path),
            "quality_manifest": str(quality_manifest),
            "timing_noise_qualification": str(timing_noise),
        },
        "calibration": calibration,
        "calibration_sha256": calibration_sha256,
        "verifier": {
            "source_commit": source_commit,
            "sha256": verifier_sha256(REPO),
        },
        "evidence": [],
    }
    genesis_path = root / "genesis.json"
    genesis_path.write_text(json.dumps(genesis, indent=2, sort_keys=True) + "\n")
    bind_reviews(root, genesis_path)
    return genesis_path


def run_gate(root: Path, genesis: Path) -> subprocess.CompletedProcess[str]:
    return run_command(root, "bootstrap-baseline", str(genesis))


def expect_failure(mutator, expected: str, structured: bool = True,
                   rebind_reviews: bool = True) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        genesis = build_fixture(root, structured=structured)
        mutator(root, genesis)
        if rebind_reviews:
            bind_reviews(root, genesis)
        result = run_gate(root, genesis)
        assert result.returncode != 0
        assert expected in result.stderr, result.stderr


def mutate_genesis(genesis: Path, callback) -> None:
    value = json.loads(genesis.read_text())
    callback(value)
    genesis.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def mutate_benchmark_manifest(genesis: Path, index: int, callback) -> None:
    value = json.loads(genesis.read_text())
    item = value["artifacts"]["benchmarks"][index]
    manifest = Path(item["path"]).with_suffix(".manifest")
    values = {}
    for line in manifest.read_text().splitlines():
        key, separator, content = line.partition("=")
        if separator:
            values[key] = content
    callback(values)
    write_manifest(manifest, values)
    item["manifest_sha256"] = digest(manifest)
    genesis.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def mutate_timing_noise(genesis: Path, callback) -> None:
    value = json.loads(genesis.read_text())
    path = Path(value["artifacts"]["timing_noise_qualification"])
    noise = json.loads(path.read_text())
    callback(noise)
    path.write_text(json.dumps(noise, indent=2, sort_keys=True) + "\n")
    value["record"]["thresholds"]["performance"]["formal_test"][
        "timing_noise_qualification_sha256"] = digest(path)
    genesis.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        genesis = build_fixture(root)
        result = run_gate(root, genesis)
        if result.returncode != 0:
            raise AssertionError(result.stderr)
        baseline_id = result.stdout.strip()
        assert baseline_id.startswith("sha256:")
        baseline = root / "baselines" / "sha256" / f"{baseline_id[7:]}.json"
        assert baseline.is_file()
        value = json.loads(baseline.read_text())
        assert canonical(value) == baseline_id[7:]
        assert value["provenance"]["lane_origin"] == "bootstrap"
        assert value["provenance"]["performance_method"]["pairs"] == 9
        assert value["provenance"]["performance_method"]["provider"] == "roce-v2"
        assert all(item["sigma_log"] > 0 for item in
                   value["provenance"]["performance_method"]["metrics"].values())
        duplicate = run_gate(root, genesis)
        assert duplicate.returncode != 0
        assert "already consumed" in duplicate.stderr

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        genesis = build_fixture(root, structured=True)
        result = run_gate(root, genesis)
        if result.returncode != 0:
            raise AssertionError(result.stderr)
        baseline_id = result.stdout.strip()
        baseline = root / "baselines" / "sha256" / f"{baseline_id[7:]}.json"
        value = json.loads(baseline.read_text())
        assert value["schema_version"] == 2
        assert value["key"]["tp_layout"] == GLM_TP_LAYOUT

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        genesis = build_fixture(root, providers=("roce-v2",))
        result = run_gate(root, genesis)
        if result.returncode != 0:
            raise AssertionError(result.stderr)
        baseline_id = result.stdout.strip()
        baseline = root / "baselines" / "sha256" / f"{baseline_id[7:]}.json"
        value = json.loads(baseline.read_text())
        assert value["key"]["rdma_providers"] == ["roce-v2"]

    expect_failure(
        lambda _root, genesis: mutate_genesis(
            genesis, lambda value: value["record"]["reference"].__setitem__(
                "oracle_numerical", {
                    "definition_id": "planted", "files": [
                        {"name": "fake", "sha256": "f" * 64}
                    ], "manifest_sha256": "e" * 64,
                })),
        "cannot preapprove canonical oracles")
    expect_failure(
        lambda _root, genesis: mutate_genesis(
            genesis, lambda value: value["record"].__setitem__(
                "oracle_generators", [{"id": "planted"}])),
        "cannot preapprove canonical oracles")
    expect_failure(
        lambda _root, genesis: mutate_genesis(
            genesis, lambda value: value["record"]["thresholds"].__setitem__(
                "oracle_numerical", value["record"]["thresholds"]["numerical"])),
        "cannot preapprove canonical oracles")
    expect_failure(
        lambda _root, genesis: mutate_genesis(
            genesis, lambda value: value["record"]["key"].__setitem__(
                "rdma_providers", [])),
        "one or more distinct validated RDMA providers")
    expect_failure(
        lambda _root, genesis: mutate_genesis(
            genesis, lambda value: value["record"]["key"].__setitem__(
                "rdma_providers", ["roce-v2", "roce-v2"])),
        "one or more distinct validated RDMA providers")
    expect_failure(
        lambda _root, genesis: mutate_genesis(
            genesis, lambda value: value["record"]["key"].__setitem__(
                "rdma_providers", ["tcp"])),
        "one or more distinct validated RDMA providers")
    expect_failure(
        lambda _root, genesis: mutate_benchmark_manifest(
            genesis, 0,
            lambda values: (values.pop("ds4_sha256"),
                            values.pop("peer_ds4_sha256"))),
        "binaries differed across ranks")
    expect_failure(
        lambda _root, genesis: mutate_benchmark_manifest(
            genesis, 0,
            lambda values: values.__setitem__("peer_ds4_sha256", "c" * 64)),
        "binaries differed across ranks")
    expect_failure(
        lambda _root, genesis: mutate_benchmark_manifest(
            genesis, 0,
            lambda values: (values.__setitem__("ds4_sha256", "c" * 64),
                            values.__setitem__("peer_ds4_sha256", "c" * 64))),
        "provider runs used different binaries")
    expect_failure(
        lambda _root, genesis: mutate_benchmark_manifest(
            genesis, 0,
            lambda values: values.pop("ds4_bench_producer_source_sha256")),
        "producer binary does not match")
    expect_failure(
        lambda _root, genesis: mutate_benchmark_manifest(
            genesis, 0,
            lambda values: values.__setitem__(
                "ds4_bench_producer_source_sha256", "0" * 64)),
        "producer binary does not match")
    expect_failure(
        lambda _root, genesis: mutate_benchmark_manifest(
            genesis, 0,
            lambda values: values.pop("ds4_bench_tp_sha256")),
        "no valid benchmark binary identity")
    expect_failure(
        lambda _root, genesis: mutate_benchmark_manifest(
            genesis, 0,
            lambda values: values.__setitem__(
                "ds4_bench_tp_sha256", "unverified")),
        "no valid benchmark binary identity")
    expect_failure(
        lambda _root, genesis: mutate_benchmark_manifest(
            genesis, 0,
            lambda values: values.__setitem__("ds4_bench_tp_sha256", "e" * 64)),
        "provider runs used different benchmark binaries")
    expect_failure(
        lambda _root, genesis: mutate_genesis(
            genesis, lambda value: (
                value["record"]["key"].pop("source_commit"),
                value["record"]["key"].pop("toolchain_id"))),
        "requires source and toolchain identity")
    expect_failure(
        lambda _root, genesis: mutate_timing_noise(
            genesis, lambda value: value.__setitem__("scope_sha256", "0" * 64)),
        "timing-noise qualification is invalid")
    expect_failure(
        lambda _root, genesis: mutate_timing_noise(
            genesis, lambda value: value.__setitem__("pairs", [])),
        "timing-noise qualification is invalid")
    expect_failure(
        lambda _root, genesis: mutate_genesis(
            genesis, lambda value: value["verifier"]["sha256"].__setitem__(
                "statistics", "0" * 64)),
        "reviewed verifier identity differs")
    expect_failure(
        lambda _root, genesis: mutate_genesis(
            genesis, lambda value: value["record"]["key"]["tp_layout"].__setitem__(
                "kind", "q4k-ffn-intermedate")),
        "unsupported kind", structured=True)
    expect_failure(
        lambda _root, genesis: mutate_genesis(
            genesis, lambda value: value["record"]["key"].__setitem__(
                "expert_split", "128/128")),
        "exactly one", structured=True)
    expect_failure(
        lambda _root, genesis: mutate_genesis(
            genesis, lambda value: value["record"]["key"]["tp_layout"].__setitem__(
                "shards", [1024, 1000])),
        "shards must be positive", structured=True)
    expect_failure(
        lambda _root, genesis: mutate_genesis(
            genesis, lambda value: value["record"]["key"]["tp_layout"].__setitem__(
                "experts_used", 289)),
        "invalid expert topology", structured=True)
    expect_failure(
        lambda _root, genesis: mutate_benchmark_manifest(
            genesis, 0, lambda values: values.pop("tp_reduce_width")),
        "differs in tp_reduce_width", structured=True)

    def mutate_rank_artifact(genesis, name, transform):
        value = json.loads(genesis.read_text())
        bound = value["artifacts"]["benchmarks"][0][name]
        path = Path(bound["path"])
        path.write_text(transform(path.read_text()))
        bound["sha256"] = digest(path)
        genesis.write_text(json.dumps(value))

    expect_failure(
        lambda _root, genesis: mutate_rank_artifact(
            genesis, "worker_log", lambda text: "\n".join(
                line for line in text.splitlines() if "K-shard active" not in line)),
        "lacks a unique matching TP allocation", structured=True)
    expect_failure(
        lambda _root, genesis: mutate_rank_artifact(
            genesis, "worker_status", lambda text: text.replace("exit_code=0", "exit_code=1")),
        "worker did not exit cleanly", structured=True)
    expect_failure(
        lambda _root, genesis: mutate_rank_artifact(
            genesis, "coordinator_log", lambda text: text.replace(
                "payload_fallback_calls=0", "payload_fallback_calls=1")),
        "zero-payload-fallback proof", structured=True)
    expect_failure(
        lambda _root, genesis: mutate_genesis(
            genesis, lambda value: value["record"].__setitem__(
                "schema_version", 1)),
        "record must use production schema v2", structured=True)

    def symlink_escape(root: Path, genesis: Path) -> None:
        value = json.loads(genesis.read_text())
        item = value["record"]["reference"]["numerical"]["files"][0]
        path = Path(value["artifacts"]["numerical_dir"]) / item["name"]
        outside = Path("/etc/hosts")
        path.unlink()
        path.symlink_to(outside)
        item["sha256"] = digest(outside)
        genesis.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    expect_failure(symlink_escape, "numerical file escapes canonical root")

    def tamper_quality(_root: Path, genesis: Path) -> None:
        value = json.loads(genesis.read_text())
        quality = Path(value["artifacts"]["quality_tsv"])
        quality.write_text(quality.read_text() + "tampered\n")
    expect_failure(tamper_quality, "quality binding mismatch")

    print("PASS baseline-genesis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
