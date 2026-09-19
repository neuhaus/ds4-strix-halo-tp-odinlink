#!/usr/bin/env python3
"""Adversarial checks of recorded quality process-completion evidence."""
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
TOOL = REPO / "scripts/compare-quality-scores.py"
terminal_proof = runpy.run_path(str(TOOL))["terminal_proof"]


class TerminalProof(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.scores = self.root / "quality.tsv"
        self.run_id = "12345678-1234-1234-1234-123456789abc"
        self.metadata = dict(completion_schema="quality-terminal-v1", tag="quality",
                             run_id=self.run_id, worker_supervisor_sha256="a" * 64,
                             quality_launcher_sha256="b" * 64)
        self.bind("scores", self.scores, b"scores fixture\n")
        for rank in ("coordinator", "worker"):
            self.metadata[rank + "_env"] = "DS4_BENCH_RUN_ID=" + self.run_id
            self.bind(rank + "_status", self.root / f"{rank}-quality.status",
                      b"exit_code=0\nsignal=0\n")
            self.bind(rank + "_log", self.root / f"{rank}-quality.log",
                      f"ds4-tp: benchmark run_id={self.run_id}\n".encode())

    def bind(self, name, path, data):
        path.write_bytes(data)
        self.metadata[name + "_path"] = str(path)
        self.metadata[name + "_sha256"] = hashlib.sha256(data).hexdigest()

    def check(self):
        terminal_proof(self.scores, self.metadata, required=True)

    def test_valid(self):
        self.check()

    def test_old_reference_allowed_but_new_candidate_required(self):
        terminal_proof(self.scores, {}, required=False)
        with self.assertRaisesRegex(ValueError, "missing or unsupported"):
            terminal_proof(self.scores, {}, required=True)

    def test_failed_status_even_with_rebound_hash(self):
        for status in (b"exit_code=7\nsignal=0\n", b"exit_code=143\nsignal=15\n",
                       b"exit_code=0\nsignal=0\nexit_code=7\n", b""):
            with self.subTest(status=status):
                self.bind("worker_status", self.root / "worker-quality.status", status)
                with self.assertRaisesRegex(ValueError, "unsuccessful"):
                    self.check()

    def test_mutated_score_log_or_status(self):
        for name in ("scores", "coordinator_log", "worker_status"):
            path = Path(self.metadata[name + "_path"])
            before = path.read_bytes()
            path.write_bytes(before + b"changed\n")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "mismatch"):
                self.check()
            path.write_bytes(before)

    def test_absent_or_wrong_status_path(self):
        path = Path(self.metadata["worker_status_path"])
        path.unlink()
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.check()
        self.bind("worker_status", self.root / "other-run.status", b"exit_code=0\nsignal=0\n")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.check()

    def test_rebound_log_from_other_run(self):
        self.bind("worker_log", self.root / "worker-quality.log", b"ds4-tp: benchmark run_id=other\n")
        with self.assertRaisesRegex(ValueError, "run identity"):
            self.check()

    def test_mixed_ids_and_non_utf8_logging(self):
        correct = f"ds4-tp: benchmark run_id={self.run_id}\n".encode()
        self.bind("worker_log", self.root / "worker-quality.log", b"diagnostic: \xff\n" + correct)
        self.check()
        self.bind("worker_log", self.root / "worker-quality.log", correct + b"ds4-tp: benchmark run_id=other\n")
        with self.assertRaisesRegex(ValueError, "run identity"):
            self.check()

    def test_mismatched_rank_environment(self):
        self.metadata["worker_env"] = "DS4_BENCH_RUN_ID=other"
        with self.assertRaisesRegex(ValueError, "environment"):
            self.check()

    def test_cli_enforces_candidate_completion(self):
        fixture = runpy.run_path(str(REPO / "tests/test_compare_quality_scores.py"))
        reference, candidate = self.root / "reference.tsv", self.root / "candidate.tsv"
        fixture["write_scores"](reference, [0.5, 0.6, 0.7])
        fixture["write_scores"](candidate, [0.5, 0.6, 0.7])
        thresholds = self.root / "thresholds.json"
        thresholds.write_text(json.dumps(dict(
            baseline_id="fixture", min_cases=3, min_target_tokens=30,
            max_mean_nll_delta=0, max_ci95_high_nll_delta=0,
            min_api_top1_rate_delta=0, min_api_pair_rate_delta=0)))
        result = subprocess.run([sys.executable, str(TOOL), str(reference), str(candidate),
                                 "--thresholds", str(thresholds), "--require-candidate-status"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("terminal proof is missing", result.stderr)

        # Exercise actual promotion orchestration: dropping its required-status
        # argument must make this test fail, even though the TSV comparison passes.
        diagnostic = subprocess.run([sys.executable, str(TOOL), str(reference), str(candidate),
                                     "--thresholds", str(thresholds)],
                                    capture_output=True, text=True, check=True)
        summary = self.root / "comparison.json"
        summary.write_text(diagnostic.stdout)
        gate = runpy.run_path(str(REPO / "scripts/candidate-gate.py"))
        baseline = {"thresholds": {"quality": json.loads(thresholds.read_text())}}
        with self.assertRaisesRegex(gate["GateError"], "terminal proof is missing"):
            gate["verify_quality_evidence"](
                REPO, self.root, summary, "fixture", baseline, {}, {}, {}, "")


if __name__ == "__main__":
    unittest.main()
