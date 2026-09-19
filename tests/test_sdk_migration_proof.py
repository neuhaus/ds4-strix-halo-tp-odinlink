#!/usr/bin/env python3
"""SDK contrasts cannot weaken ordinary optimization or authorize a migration."""

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import test_promotion_proof as fixture

sys.path.insert(0, str(fixture.REPO / "scripts"))
spec = importlib.util.spec_from_file_location("proof", fixture.TOOL)
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)


def arm_identity(arm):
    compiler = ("7" if arm == "control" else "a") * 64
    binary = ("b" if arm == "control" else "c") * 64
    return {
        "source_commit": fixture.SOURCE,
        "ds4_sha256": binary, "peer_ds4_sha256": binary,
        "ds4_bench_tp_sha256": ("d" if arm == "control" else "e") * 64,
        "toolchain_id": "elf-comment-sha256:" + compiler,
        "binary_toolchain_sha256": compiler,
        "expected_binary_toolchain_sha256": compiler,
        "binary_toolchain_comment": "fixture compiler " + arm,
        "binary_runpath": "/sdk/" + arm + "/lib",
    }


def migration_policy():
    contract = fixture.performance_contract()
    contract["maximum_untargeted_regression"] = 0.03
    contract["formal_test"] = {
        "kind": "exact-sign-fixed-nine-v1", "pairs": 9,
        "minimum_positive": 8, "ties": "fail", "familywise_alpha": 0.05,
    }
    return {
        "contract": contract, "target_metrics": [], "candidate_switches": {},
        "public_claim": False,
        "sdk_contrast": {"kind": "sdk-contrast-v1",
                         "cell_switches": {cell: {} for cell in proof.SDK_CELLS},
                         "runtimes": {
                             arm: {"sdk_root": "/sdk/" + arm, "kernel": "fixture-kernel",
                                   "libraries": {"/sdk/" + arm + "/lib/" + prefix + "1":
                                                 ("7" if arm == "control" else "a") * 64
                                                 for prefix in proof.SDK_LIBRARY_PREFIXES}}
                             for arm in ("control", "candidate")},
                         **{arm: arm_identity(arm)
                            for arm in ("control", "candidate")}},
    }


def timing_pairs(ratios):
    return [{"order": "AB" if index % 2 == 0 else "BA",
             "control": {"result": {"prefill_tps": 100, "decode_tps": 10}},
             "candidate": {"result": {"prefill_tps": 100 * ratio,
                                       "decode_tps": 10 * ratio}}}
            for index, ratio in enumerate(ratios)]


def migration_spec(root):
    path = fixture.build_spec(root, pair_count=9)
    value = json.loads(path.read_text())
    value["stage"] = "sdk-diagnostic"
    value["performance"] = migration_policy()
    for cell in proof.SDK_CELLS:
        value["performance"]["sdk_contrast"]["cell_switches"][cell] = {
            "DS4_FEATURE": {"control": "0", "candidate": "1"}}
    glm_q2 = root / "glm-q2.gguf"
    glm_q2.write_bytes(b"glm-q2")
    glm_q2_pair = fixture.make_pair(
        root, "glm-q2", "AB", frontier=2048,
        prompt_sha=fixture.DEEPSEEK_PROMPT_SHA256, model=glm_q2,
        model_sample=fixture.sample_digest(glm_q2), model_size=glm_q2.stat().st_size)
    value["ordinary_regressions"].append({
        "name": "glm-53-q2", "baseline_fnv64": fixture.FNV,
        "model": {"path": str(glm_q2), "size": glm_q2.stat().st_size,
                  "sample_sha256": fixture.sample_digest(glm_q2),
                  "sha256": fixture.digest_bytes(glm_q2.read_bytes()), "quantization": "Q2_K"},
        "screen": fixture.screen(glm_q2_pair),
    })
    pairs = [*value["headline_pairs"], value["diverse_screen"]["pair"],
             value["long_context_screen"]["pair"],
             *(item["screen"]["pair"] for item in value["ordinary_regressions"])]
    for pair in pairs:
        pair["allowed_fields"] = sorted(proof.SDK_ARM_FIELDS)
        for arm in ("control", "candidate"):
            manifest_path = Path(pair[arm]["manifest"])
            fields = dict(line.split("=", 1) for line in
                          manifest_path.read_text().splitlines())
            fields.update(arm_identity(arm))
            if pair["pair_id"] in ("q4", "q2"):
                fields["model_arch"] = "deepseek4"
            for name in proof.RANK_ENV_FIELDS:
                fields[name] += " DS4_GLM5_NATIVE_DRAFT=0"
            fixture.manifest(manifest_path, fields)
            runtime = value["performance"]["sdk_contrast"]["runtimes"][arm]
            capture = {"tag": fields["tag"], "source": fixture.SOURCE, "ranks": {},
                       "manifest_sha256": fixture.digest_bytes(manifest_path.read_bytes())}
            for rank in ("coordinator", "worker"):
                binary = "ds4_bench_tp_sha256" if rank == "coordinator" else "peer_ds4_sha256"
                capture["ranks"][rank] = {
                    "pid": 42, "start_ticks": "1", "observed_unix_ns": 123,
                    "executable_sha256": fields[binary], "kernel": runtime["kernel"],
                    "effective_environment": proof.parse_env(fields[rank + "_env"], rank),
                    "runtime_environment": proof.parse_env(fields[rank + "_env"], rank),
                    "libraries": [{"path": path, "resolved": path, "sha256": sha}
                                  for path, sha in runtime["libraries"].items()],
                    "mappings": ["0000-1000 r--p 00000000 01:01 1 " + path
                                 for path in runtime["libraries"]],
                }
            runtime_path = manifest_path.with_suffix(".runtime.json")
            runtime_path.write_text(json.dumps(capture))
            pair[arm]["runtime"] = str(runtime_path)
    path.write_text(json.dumps(value))
    return path, value


class SDKMigrationProofTests(unittest.TestCase):
    def test_guard_only_pass_has_no_merge_authority(self):
        result = proof.performance_decision(timing_pairs([0.98] * 9), migration_policy())
        self.assertTrue(result["passed"])
        self.assertFalse(result["merge_eligible"])
        self.assertEqual(result["metrics"]["decode"]["required_lower_bound"], -0.03)

    def test_original_margin_ties_and_nine_pair_rule_remain(self):
        for ratios in ([0.96] * 9, [0.97] * 9, [0.98] * 7,
                       [0.96, 0.96] + [1.0] * 7):
            with self.subTest(ratios=ratios):
                self.assertFalse(proof.performance_decision(
                    timing_pairs(ratios), migration_policy())["passed"])
        self.assertTrue(proof.performance_decision(
            timing_pairs([0.96] + [1.0] * 8), migration_policy())["passed"])

    def test_cannot_mislabel_optimization_or_claim_speedup(self):
        for update in ({"public_claim": True}, {"target_metrics": ["decode"]},
                       {"sdk_contrast": None}):
            policy = migration_policy()
            policy.update(update)
            with self.subTest(update=update), self.assertRaises(proof.ProofError):
                proof.validate_performance_policy(policy)
        policy = migration_policy()
        del policy["sdk_contrast"]
        with self.assertRaisesRegex(proof.ProofError, "target declaration"):
            proof.validate_performance_policy(policy)
        policy = migration_policy()
        policy["contract"]["formal_test"] = fixture.performance_contract()["formal_test"]
        with self.assertRaisesRegex(proof.ProofError, "guard-only exact-sign"):
            proof.validate_performance_policy(policy)

    def test_ordinary_create_rejects_empty_targets_and_sdk_allowance(self):
        for mutation, expected in (("targets", "target declaration"),
                                   ("sdk", "not an approved build")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = fixture.build_spec(root)
                value = json.loads(path.read_text())
                if mutation == "targets":
                    value["performance"]["target_metrics"] = []
                else:
                    value["headline_pairs"][0]["allowed_fields"] = ["toolchain_id"]
                path.write_text(json.dumps(value))
                result, _ = fixture.create(root, path)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)

    def test_sdk_schema_refuses_missing_or_unbounded_identity(self):
        original = migration_policy()["sdk_contrast"]
        for arm in ("control", "candidate"):
            for field in proof.SDK_ARM_FIELDS:
                value = copy.deepcopy(original)
                del value[arm][field]
                with self.subTest(arm=arm, field=field), self.assertRaises(proof.ProofError):
                    proof.validate_sdk_contrast(value)
        for mutate in (
                lambda v: v.update(candidate=v["control"]),
                lambda v: v["candidate"].update(peer_ds4_sha256="0" * 64),
                lambda v: v["candidate"].update(expected_binary_toolchain_sha256="0" * 64),
                lambda v: v["candidate"].update(unreviewed_allowance="anything"),
                lambda v: v["candidate"].update(ds4_bench_tp_sha256=v["control"]["ds4_bench_tp_sha256"]),
                lambda v: v["runtimes"].pop("candidate"),
                lambda v: v["runtimes"]["candidate"]["libraries"].popitem()):
            value = copy.deepcopy(original)
            mutate(value)
            with self.assertRaises(proof.ProofError):
                proof.validate_sdk_contrast(value)

    def test_each_arm_binds_identity_and_disables_native(self):
        contrast = migration_policy()["sdk_contrast"]
        manifest = {**contrast["candidate"], "dspark": "0",
                    "worker_env": "DS4_GLM5_NATIVE_DRAFT=0",
                    "coordinator_env": "DS4_GLM5_NATIVE_DRAFT=0"}
        proof.verify_sdk_arm(manifest, "candidate", contrast)
        for field in proof.SDK_ARM_FIELDS:
            value = {**manifest, field: "changed"}
            with self.subTest(field=field), self.assertRaises(proof.ProofError):
                proof.verify_sdk_arm(value, "candidate", contrast)
        for field, value in (("dspark", "1"), ("worker_env", ""),
                             ("coordinator_env", "DS4_GLM5_NATIVE_DRAFT=6"),
                             ("common_env", "DS4_GLM5_NATIVE_DRAFT=6"),
                             ("extra_env", "DS4_GLM5_NATIVE_PHASE_PROFILE=0"),
                             ("rocprof_binary", "/sdk/rocprof")):
            with self.subTest(field=field), self.assertRaises(proof.ProofError):
                proof.verify_sdk_arm({**manifest, field: value}, "candidate", contrast)

    def test_full_proof_roundtrip_is_diagnostic_and_cannot_promote(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, value = migration_spec(root)
            result, output = fixture.create(root, path)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["stage"], "sdk-diagnostic")
            self.assertEqual(report["admission"], "none-diagnostic-only")
            self.assertFalse(report["performance"]["merge_eligible"])
            self.assertEqual(fixture.run(root, "verify", str(output)).returncode, 0)
            for stage in ("qualification", "promotion"):
                value["stage"] = stage
                path.write_text(json.dumps(value))
                result, _ = fixture.create(root, path)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("cannot qualify or promote", result.stderr)

    def test_swapped_screen_sdk_fails_even_with_allowed_toolchain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, value = migration_spec(root)
            screen = value["long_context_screen"]["pair"]
            manifest_path = Path(screen["candidate"]["manifest"])
            fields = dict(line.split("=", 1) for line in manifest_path.read_text().splitlines())
            fields.update(arm_identity("control"))
            fixture.manifest(manifest_path, fields)
            result, _ = fixture.create(root, path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("frozen SDK identity", result.stderr)

    def test_missing_q2_or_foreign_teacher_thresholds_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, value = migration_spec(root)
            original = copy.deepcopy(value)
            value["ordinary_regressions"].pop()
            path.write_text(json.dumps(value))
            result, _ = fixture.create(root, path)
            self.assertIn("each required model", result.stderr)
            value = original
            value["ordinary_regressions"][0]["screen"]["trajectory"] = {"mode": "teacher"}
            path.write_text(json.dumps(value))
            result, _ = fixture.create(root, path)
            self.assertIn("model-specific numerical/quality admission", result.stderr)

    def test_model_switches_do_not_leak_into_glm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, value = migration_spec(root)
            cell = value["ordinary_regressions"][0]
            switches = value["performance"]["sdk_contrast"]["cell_switches"][cell["name"]]
            switches["DS4_TP_HOST_CALLBACK"] = {"control": "1", "candidate": "0"}
            cell["screen"]["pair"]["allowed_env"].append("DS4_TP_HOST_CALLBACK")
            for arm in ("control", "candidate"):
                manifest_path = Path(cell["screen"]["pair"][arm]["manifest"])
                fields = dict(line.split("=", 1) for line in manifest_path.read_text().splitlines())
                for field in (*proof.RANK_ENV_FIELDS, "extra_env"):
                    fields[field] += " DS4_TP_HOST_CALLBACK=" + switches["DS4_TP_HOST_CALLBACK"][arm]
                fixture.manifest(manifest_path, fields)
                runtime_path = Path(cell["screen"]["pair"][arm]["runtime"])
                capture = json.loads(runtime_path.read_text())
                for rank in ("coordinator", "worker"):
                    capture["ranks"][rank]["effective_environment"] = proof.parse_env(
                        fields[rank + "_env"], rank)
                    capture["ranks"][rank]["runtime_environment"] = proof.parse_env(
                        fields[rank + "_env"], rank)
                capture["manifest_sha256"] = fixture.digest_bytes(manifest_path.read_bytes())
                runtime_path.write_text(json.dumps(capture))
            path.write_text(json.dumps(value))
            result, _ = fixture.create(root, path)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_mixed_or_wrong_run_runtime_fails(self):
        for mutation in ("missing", "library", "binary", "run_id", "kernel",
                         "coordinator", "extra_env", "mappings", "symlink", "sixth_library", "manifest"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path, value = migration_spec(root)
                arm = value["headline_pairs"][0]["candidate"]
                if mutation == "missing":
                    del arm["runtime"]
                    path.write_text(json.dumps(value))
                else:
                    runtime_path = Path(arm["runtime"])
                    capture = json.loads(runtime_path.read_text())
                    record = capture["ranks"]["worker"]
                    if mutation == "library":
                        record["libraries"][0]["sha256"] = "0" * 64
                    elif mutation == "binary":
                        record["executable_sha256"] = "0" * 64
                    elif mutation == "run_id":
                        record["effective_environment"]["DS4_BENCH_RUN_ID"] = "another-process"
                    elif mutation == "coordinator":
                        capture["ranks"]["coordinator"]["executable_sha256"] = "0" * 64
                    elif mutation == "extra_env":
                        record["runtime_environment"]["DS4_GLM5_NATIVE_PHASE_PROFILE"] = "0"
                    elif mutation == "mappings":
                        record["mappings"].pop()
                    elif mutation == "symlink":
                        record["libraries"][0]["path"] = "/some-other-library"
                    elif mutation == "sixth_library":
                        record["libraries"].append(record["libraries"][0])
                    elif mutation == "manifest":
                        capture["manifest_sha256"] = "0" * 64
                    else:
                        record["kernel"] = "another-kernel"
                    runtime_path.write_text(json.dumps(capture))
                result, _ = fixture.create(root, path)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("runtime", result.stderr)

    def test_all_sdk_teacher_modes_are_unadmitted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, value = migration_spec(root)
            for name in ("diverse_screen", "long_context_screen"):
                for trajectory in ({"mode": "teacher"}, []):
                    changed = copy.deepcopy(value)
                    changed[name]["trajectory"] = trajectory
                    path.write_text(json.dumps(changed))
                    result, _ = fixture.create(root, path)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("missing model-specific", result.stderr)

    def test_model_labels_bind_quantization_and_architecture(self):
        for mutation in ("quantization", "architecture", "duplicate"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path, value = migration_spec(root)
                cell = value["ordinary_regressions"][-1]
                if mutation == "quantization":
                    cell["model"]["quantization"] = "Q4_K"
                elif mutation == "duplicate":
                    cell["model"] = value["ordinary_regressions"][1]["model"]
                else:
                    for arm in ("control", "candidate"):
                        manifest_path = Path(cell["screen"]["pair"][arm]["manifest"])
                        fields = dict(line.split("=", 1) for line in manifest_path.read_text().splitlines())
                        fields["model_arch"] = "deepseek4"
                        fixture.manifest(manifest_path, fields)
                        runtime_path = Path(cell["screen"]["pair"][arm]["runtime"])
                        capture = json.loads(runtime_path.read_text())
                        capture["manifest_sha256"] = fixture.digest_bytes(manifest_path.read_bytes())
                        runtime_path.write_text(json.dumps(capture))
                path.write_text(json.dumps(value))
                result, _ = fixture.create(root, path)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("SDK regression cell", result.stderr)

    def test_capture_binds_only_the_documented_manifest_append(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, value = migration_spec(root)
            manifest_path = Path(value["headline_pairs"][0]["candidate"]["manifest"])
            with manifest_path.open('a') as stream:
                stream.write("dump_generated_token_sha256=" + "f" * 64 + "\n")
            # The other arm needs the same non-SDK field for pair matching.
            control_path = Path(value["headline_pairs"][0]["control"]["manifest"])
            with control_path.open('a') as stream:
                stream.write("dump_generated_token_sha256=" + "f" * 64 + "\n")
            result, _ = fixture.create(root, path)
            self.assertEqual(result.returncode, 0, result.stderr)
            with manifest_path.open('a') as stream:
                stream.write("another_field=unbound\n")
            result, _ = fixture.create(root, path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("capture manifest hash", result.stderr)


if __name__ == "__main__":
    unittest.main()
