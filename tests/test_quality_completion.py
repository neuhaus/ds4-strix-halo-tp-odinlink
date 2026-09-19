#!/usr/bin/env python3
"""Run the production quality launcher with local fake ranks (no GPU/network)."""
import os
from pathlib import Path
import shutil
import runpy
import subprocess
import sys
import tempfile
import time
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
terminal_proof = runpy.run_path(str(REPO / "scripts/compare-quality-scores.py"))["terminal_proof"]


def executable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


class QualityCompletion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "source"
        self.out = self.root / "evidence"
        self.peer = self.root / "peer-evidence"
        for folder in (self.repo / "scripts", self.out, self.peer):
            folder.mkdir(parents=True)
        for name in ("run-tp-quality-score.sh", "scripts/ds4-research-root.sh",
                     "scripts/tp-worker-supervisor.sh"):
            shutil.copy2(REPO / name, self.repo / name)
        executable(self.repo / "scripts/gguf_tensor_types.py",
                   "#!/usr/bin/python3\nprint('deepseek4')\n")
        self.model = self.root / "unchanged.gguf"
        self.model.write_bytes(b"fixture model")
        self.inputs = self.root / "manifest.tsv"
        self.inputs.write_text("fixture\n")
        executable(self.repo / "ds4", '''#!/usr/bin/python3
import os, time
print('transport=rdma', flush=True)
print('ds4-tp: benchmark run_id=' + os.environ.get('DS4_BENCH_RUN_ID', ''), flush=True)
time.sleep(float(os.environ.get('DS4_TEST_WORKER_SLEEP', '0')))
raise SystemExit(int(os.environ.get('DS4_TEST_WORKER_RC', '0')))
''')
        executable(self.repo / "gguf-tools/quality-testing/score_official", '''#!/usr/bin/python3
import os, sys, time
from pathlib import Path
print('transport=rdma', flush=True)
print('ds4-tp: benchmark run_id=' + os.environ.get('DS4_BENCH_RUN_ID', ''), flush=True)
print('cases=1 api_ref_tokens=10', flush=True)
Path(sys.argv[3]).write_text('id\\tnll\\ttarget_tokens\\tavg_nll\\tapi_top1_count\\tapi_top1_match\\tapi_pair_total\\tapi_pair_agree\\nfixture\\t5\\t10\\t0.5\\t10\\t9\\t10\\t9\\n')
time.sleep(float(os.environ.get('DS4_TEST_COORD_SLEEP', '0')))
raise SystemExit(int(os.environ.get('DS4_TEST_COORD_RC', '0')))
''')
        fake = self.root / "fake-tools"
        executable(fake / "ssh", '''#!/usr/bin/python3
import os, sys
os.execv('/bin/bash', ['bash', '-c', sys.argv[-1]])
''')
        executable(fake / "scp", '''#!/usr/bin/python3
import os, shutil, sys
source, dest = sys.argv[-2:]
source = source.removeprefix('fixture:')
dest = dest.removeprefix('fixture:')
if os.environ.get('TEST_DROP_STATUS') == '1' and source.endswith('.status'):
    raise SystemExit(1)
shutil.copyfile(source, dest)
''')
        executable(fake / "cat", '''#!/usr/bin/python3
import os, sys
if len(sys.argv) == 2 and sys.argv[1].startswith('/sys/class/infiniband/'):
    print('RoCE v2')
else:
    os.execv('/usr/bin/cat', ['cat', *sys.argv[1:]])
''')
        executable(fake / "grep", '''#!/usr/bin/python3
import os, sys
if sys.argv[-1].startswith('/sys/class/infiniband/'):
    raise SystemExit(0)
os.execv('/usr/bin/grep', ['grep', *sys.argv[1:]])
''')
        executable(fake / "pgrep", "#!/bin/bash\nexit 1\n")
        self.env = dict(os.environ, PATH=str(fake) + os.pathsep + os.environ['PATH'],
                        DS4_RESEARCH_ROOT=str(self.root / "archive"),
                        DS4_PEER_MGMT="fixture", DS4_BENCH_CONFIG="/dev/null",
                        DS4_COORDINATOR_ADDR="127.0.0.1", DS4_QUALITY_RDMA_PROFILE="roce-v2",
                        DS4_QUALITY_OUT=str(self.out), DS4_PEER_QUALITY_OUT=str(self.peer),
                        DS4_QUALITY_MANIFEST=str(self.inputs), DS4_QUALITY_MAX_CASES="1")

    def run_launcher(self, *settings):
        return subprocess.run(["bash", str(self.repo / "run-tp-quality-score.sh"),
                               "fixture", str(self.model), *settings],
                              env=self.env, capture_output=True, text=True, timeout=30)

    def status(self, rank):
        return dict(line.split("=", 1) for line in
                    (self.out / f"{rank}-fixture.status").read_text().splitlines())

    def test_success_binds_statuses_and_refuses_reuse(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        for rank in ("coordinator", "worker"):
            self.assertEqual(self.status(rank), {"exit_code": "0", "signal": "0"})
        manifest = (self.out / "fixture.manifest").read_text()
        self.assertIn("completion_schema=quality-terminal-v1\n", manifest)
        self.assertIn("worker_status_sha256=", manifest)
        terminal_proof(self.out / "fixture.tsv",
                       dict(line.split("=", 1) for line in manifest.splitlines()), required=True)
        before = {p.name: p.read_bytes() for p in self.out.iterdir() if p.is_file()}
        self.assertNotEqual(self.run_launcher().returncode, 0)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.out.iterdir() if p.is_file()})

    def test_worker_failure_cannot_publish_quality_manifest(self):
        result = self.run_launcher("DS4_TEST_WORKER_RC=7")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.status("worker"), {"exit_code": "7", "signal": "0"})
        self.assertEqual(self.status("coordinator")["exit_code"], "0")
        self.assertFalse((self.out / "fixture.manifest").exists())

    def test_coordinator_failure_retains_both_statuses(self):
        result = self.run_launcher("DS4_TEST_COORD_RC=9")
        self.assertEqual(result.returncode, 9, result.stderr)
        self.assertEqual(self.status("coordinator"), {"exit_code": "9", "signal": "0"})
        self.assertIn(self.status("worker")["exit_code"], ("0", "143"))
        self.assertFalse((self.out / "fixture.manifest").exists())

    def test_missing_worker_status_refuses_success(self):
        # Simulate a supervisor that returns normally but fails to write status.
        supervisor = self.repo / "scripts/tp-worker-supervisor.sh"
        supervisor.write_text(supervisor.read_text().replace(
            'write_status "$rc"', '[[ $status_file == *coordinator-* ]] && write_status "$rc"'))
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse((self.out / "fixture.manifest").exists())

    def test_launcher_termination_reaps_both_owned_ranks(self):
        process = subprocess.Popen([
            "bash", str(self.repo / "run-tp-quality-score.sh"), "fixture", str(self.model),
            "DS4_TEST_COORD_SLEEP=20", "DS4_TEST_WORKER_SLEEP=20"],
            env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(100):
                logs = [self.out / "coordinator-fixture.log", self.peer / "worker-fixture.log"]
                if all(p.exists() and 'benchmark run_id=' in p.read_text() for p in logs):
                    break
                time.sleep(0.02)
            else:
                self.fail("fake ranks did not start")
            process.terminate()
            self.assertEqual(process.wait(timeout=10), 143)
            for rank in ("coordinator", "worker"):
                self.assertEqual(self.status(rank), {"exit_code": "143", "signal": "15"})
            self.assertFalse((self.out / "fixture.manifest").exists())
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)

    def test_worker_cleanup_refuses_another_runs_supervisor(self):
        supervisor = self.repo / "scripts/tp-worker-supervisor.sh"
        foreign_status = self.peer / "foreign.status"
        process = subprocess.Popen([str(supervisor), str(foreign_status),
                                    "/usr/bin/sleep", "20"])
        pidfile = self.peer / "worker-fixture.pid"
        pidfile.write_text(str(process.pid) + "\n")
        source = (self.repo / "run-tp-quality-score.sh").read_text()
        function = 'worker_action() {' + source.split('worker_action() {', 1)[1].split('\n}\n', 1)[0] + '\n}\n'
        try:
            result = subprocess.run([
                "bash", "-c", function + '\nPEER_SSH=(bash -c)\nworker_action term'],
                env=dict(self.env, WORKER_PIDFILE=str(pidfile), PEER_SUPERVISOR=str(supervisor),
                         REMOTE_WORKER_STATUS=str(self.peer / "worker-fixture.status")),
                capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 3, result.stderr)
            self.assertIsNone(process.poll())
        finally:
            process.terminate()
            process.wait(timeout=5)

    def test_relative_output_is_rejected_before_launch(self):
        self.env["DS4_QUALITY_OUT"] = "relative-evidence"
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be absolute", result.stderr)
        self.assertFalse((self.peer / "worker-fixture.pid").exists())


if __name__ == "__main__":
    unittest.main()
