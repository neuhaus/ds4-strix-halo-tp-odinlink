#!/usr/bin/env python3
"""Exercise schema-v2 canonical-oracle amendments and their safety gates."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path


if os.environ.get("DS4_GATE_CLEAN_TEST") != "1":
    raise SystemExit("run via tests/test_lane_c_oracle_gate.sh")


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "candidate_gate", ROOT / "scripts" / "candidate-gate.py")
assert SPEC and SPEC.loader
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)
sys.path.insert(0, str(ROOT / "scripts"))
import ds4_gate_controls as CONTROLS  # noqa: E402
import test_baseline_genesis as GENESIS  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_reviews(root: Path, amendment: Path) -> None:
    value = json.loads(amendment.read_text())
    reviewable = deepcopy(value)
    reviewable["evidence"] = [
        item for item in reviewable.get("evidence", [])
        if not isinstance(item, dict) or
        item.get("kind") not in {"fable-review", "grok-review"}
    ]
    payload_sha = GATE.canonical_sha256(reviewable)
    calibration_sha = value["calibration_sha256"]
    reviews = []
    for reviewer in ("fable", "grok"):
        path = root / f"{reviewer}-{value['amendment_id']}.txt"
        path.write_text(
            "DS4-REVIEW-SCHEMA: 1\n"
            f"REVIEWER: {reviewer}\n"
            f"REVIEWED-PAYLOAD-SHA256: {payload_sha}\n"
            f"CALIBRATION-SHA256: {calibration_sha}\n"
            "VERDICT: GO\n"
            "Schema-v2 fixture review bound to the exact amendment payload.\n")
        reviews.append({"kind": f"{reviewer}-review", "path": str(path),
                        "sha256": digest(path)})
    value["evidence"] = reviews
    amendment.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_gate(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["DS4_RESEARCH_ROOT"] = str(root)
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "candidate-gate.py"), *arguments],
        cwd=ROOT, env=environment, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def generator_bundle(root: Path) -> dict:
    generator = root / "oracle-generator.py"
    generator.write_text(
        "#!/usr/bin/env python3\n"
        "import argparse\n"
        "import json\n"
        "from pathlib import Path\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--model', required=True)\n"
        "p.add_argument('--definition', required=True)\n"
        "p.add_argument('--prefix', required=True)\n"
        "p.add_argument('--token-file', required=True)\n"
        "p.add_argument('--output-dir', required=True)\n"
        "a = p.parse_args()\n"
        "out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)\n"
        "for i in range(300):\n"
        "  (out / f'decode_{i:06d}.logits.json').write_text(json.dumps({'i': i}))\n")
    generator.chmod(0o755)
    generator_item = {"scope": "research", "path": generator.name,
                      "sha256": digest(generator)}
    runner = Path(sys.executable).resolve()
    runner_item = {"scope": "system", "path": str(runner),
                   "sha256": digest(runner)}
    closure = [generator_item, runner_item]
    environment = {
        "kind": "python", "implementation": sys.implementation.name,
        "version": sys.version, "runner": runner_item, "packages": [],
    }
    return {
        "id": "fixture-oracle-v2", "entrypoint": generator_item,
        "runner": runner_item, "environment": environment,
        "environment_id": GATE.canonical_sha256(environment),
        "closure": closure, "closure_sha256": GATE.canonical_sha256(closure),
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        genesis = GENESIS.build_fixture(root, structured=True,
                                        providers=("roce-v2",))
        created = GENESIS.run_gate(root, genesis)
        assert created.returncode == 0, created.stderr
        predecessor_id = created.stdout.strip()
        predecessor_path = root / "baselines" / "sha256" / \
            f"{predecessor_id[7:]}.json"
        predecessor = json.loads(predecessor_path.read_text())
        numerical = deepcopy(predecessor["thresholds"]["numerical"])

        calibration = json.loads(genesis.read_text())["calibration"]
        calibration_result = CONTROLS.evaluate_calibration(
            ROOT, root, calibration, predecessor["scope_sha256"],
            {"oracle_numerical": numerical},
            {"oracle_numerical": numerical},
        )
        generator = generator_bundle(root)
        amendment = {
            "schema_version": 2,
            "kind": "ds4-baseline-amendment",
            "amendment_id": "fixture-oracle-adoption-v2",
            "baseline_id": predecessor_id,
            "rationale": "Approve a canonical producer without changing the incumbent trajectory.",
            "add_oracle_generator": generator,
            "threshold_updates": {"oracle_numerical": numerical},
            "calibration": calibration,
            "calibration_sha256": calibration_result["calibration_sha256"],
            "timing_noise_qualification": None,
            "verifier": {
                "source_commit": subprocess.check_output(
                    ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                    text=True).strip(),
                "sha256": GATE.verifier_sha256(ROOT),
            },
            "evidence": [],
        }
        amendment_path = root / "amendment.json"
        amendment_path.write_text(json.dumps(amendment, indent=2, sort_keys=True) + "\n")
        bind_reviews(root, amendment_path)
        adopted = run_gate(root, "amend-baseline", str(amendment_path))
        assert adopted.returncode == 0, adopted.stderr
        successor_id = adopted.stdout.strip()
        _, successor = GATE.load_baseline(root, successor_id)
        assert successor["reference"]["fnv64"] == predecessor["reference"]["fnv64"]
        assert successor["oracle_generators"][0]["id"] == generator["id"]
        assert successor["provenance"]["replaces"] == predecessor_id
        assert successor["provenance"]["calibration"]["calibration_sha256"] == \
            amendment["calibration_sha256"]

        stale = deepcopy(amendment)
        stale["amendment_id"] = "stale-amendment"
        stale_path = root / "stale-amendment.json"
        stale_path.write_text(json.dumps(stale, indent=2, sort_keys=True) + "\n")
        bind_reviews(root, stale_path)
        stale_result = run_gate(root, "amend-baseline", str(stale_path))
        assert stale_result.returncode != 0
        assert "superseded" in stale_result.stderr or "active head" in stale_result.stderr

        unsafe = deepcopy(amendment)
        unsafe["amendment_id"] = "unsafe-oracle-amendment"
        unsafe["baseline_id"] = successor_id
        unsafe["add_oracle_generator"] = None
        unsafe["threshold_updates"]["oracle_numerical"] = deepcopy(numerical)
        unsafe["threshold_updates"]["oracle_numerical"]["safety"]["max_kl"] = \
            numerical["safety"]["max_kl"] + 1.0
        unsafe_path = root / "unsafe-amendment.json"
        unsafe_path.write_text(json.dumps(unsafe, indent=2, sort_keys=True) + "\n")
        bind_reviews(root, unsafe_path)
        unsafe_result = run_gate(root, "amend-baseline", str(unsafe_path))
        assert unsafe_result.returncode != 0
        assert "widens hard safety" in unsafe_result.stderr, unsafe_result.stderr

    print("test_lane_c_oracle_gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
