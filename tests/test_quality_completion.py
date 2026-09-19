#!/usr/bin/env python3
"""Run the production quality launcher with local fake ranks (no GPU/network)."""
import os
import hashlib
import json
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

    def run_launcher(self, *settings, tag="fixture", cwd=None):
        return subprocess.run(["bash", str(self.repo / "run-tp-quality-score.sh"),
                               tag, str(self.model), *settings],
                              env=self.env, cwd=cwd, capture_output=True, text=True, timeout=30)

    def status(self, rank):
        return dict(line.split("=", 1) for line in
                    (self.out / f"{rank}-fixture.status").read_text().splitlines())

    def teacher_fixture(self, architecture="deepseek4", arm="deepseek-ordinary"):
        directory = self.root / "archive" / "teacher"
        self.env.update(DS4_QUALITY_TEACHER_LOGITS_DIR=str(directory),
                        DS4_QUALITY_TEACHER_ARM=arm)
        prompt, continuation = self.root / 'prompt.txt', self.root / 'continuation.txt'
        prompt.write_text('fixture prompt')
        continuation.write_text('fixture continuation')
        self.inputs.write_text(f'fixture\t{prompt}\t{continuation}\n')
        executable(self.repo / "scripts/gguf_tensor_types.py",
                   "#!/usr/bin/python3\nprint(" + repr(architecture) + ")\n")
        for name in ('compare-teacher-logits.py', 'ds4_gate_stats.py'):
            shutil.copy2(REPO / 'scripts' / name, self.repo / 'scripts' / name)
        startup = '''
rank = 0 if 'score_official' in sys.argv[0] else 1
if os.environ.get('DS4_TEST_NEGOTIATION') != 'missing':
    mask = '0x000780c9'
    if rank == 1 and os.environ.get('DS4_TEST_NEGOTIATION') == 'mismatch':
        mask = '0x000780c8'
    print(f'ds4: ROCm Q4_K WMMA startup rank={rank} negotiated={mask} gate=1 up=1 down=1 kshard=0 kda_tp=0 kda_output_kslice=0 quality=0 kill_switch=0', flush=True)
if os.environ.get('DS4_TEST_TRANSPORT') != 'no-gid':
    print('rdma GID index 3 (RoCE v2)', flush=True)
cache = 1 if os.environ.get('DS4_TEST_TRANSPORT') == 'cache' else 0
print(f'ds4: ROCm memory before engine cleanup: expanded_weight_cache_bytes={cache}', flush=True)
if os.environ.get('DS4_TEST_TRANSPORT') != 'missing':
    active = 'tcp' if os.environ.get('DS4_TEST_TRANSPORT') == 'tcp' else 'rdma'
    fallback = 1 if os.environ.get('DS4_TEST_TRANSPORT') == 'fallback' else 0
print(f'ds4-tp: transport proof requested=rdma active={active} payload_fallback_calls={fallback} failed=0', flush=True)
'''
        if architecture == 'glm5-next':
            startup += "print('GLM5 TP features: kda_tp=1 kda_output_kslice=0', flush=True)\n"
        worker = self.repo / "ds4"
        worker.write_text(worker.read_text().replace('import os, time', 'import os, sys, time').replace(
            "time.sleep(float", startup + "\ntime.sleep(float", 1))
        executable(self.repo / "gguf-tools/quality-testing/score_official", '''#!/usr/bin/python3
import json, math, os, sys
from pathlib import Path
print('transport=rdma', flush=True)
print('ds4-tp: benchmark run_id=' + os.environ.get('DS4_BENCH_RUN_ID', ''), flush=True)
print('cases=1 api_ref_tokens=2', flush=True)
''' + startup + '''
out = Path(sys.argv[sys.argv.index('--teacher-logits-dir') + 1])
values = [0.0, 1.0, 0.0, -1.0]
nll = math.log(sum(math.exp(v) for v in values)) - values[1]
Path(sys.argv[3]).write_text('id\\tprompt_tokens\\ttarget_tokens\\tnll\\tavg_nll\\nfixture\\t16\\t2\\t%.9f\\t%.9f\\n' % (2*nll, nll))
mode = os.environ.get('DS4_TEST_DUMP_MODE', '')
for step in range(1 if mode == 'missing' else 3 if mode == 'extra' else 2):
    record = dict(source='ds4-score-official-frozen-teacher', model=sys.argv[1],
        case_id='fixture', backend='rocm', quality=False, dspark=False, dspark_strict=False,
        quant_bits=2, prefix_tokens=16, decode_step=step, case_step=step,
        position=16+step, vocab=4, teacher_token=1, teacher_logit=1.0,
        argmax_id=1, argmax_logit=1.0, runner_up_id=0, runner_up_logit=0.0,
        top1_margin=1.0, teacher_gap=0.0, logits=list(values))
    if mode == 'nonfinite': record['logits'][0] = float('nan')
    if mode == 'wrong-case': record['case_id'] = 'another-case'
    if mode == 'wrong-step': record['case_step'] = 0
    if mode == 'truncated': record['logits'].pop()
    (out / ('decode_%06d.logits.json' % step)).write_text(json.dumps(record))
raise SystemExit(int(os.environ.get('DS4_TEST_COORD_RC', '0')))
''')
        (self.repo / '.gitignore').write_text('__pycache__/\n')
        subprocess.run(['git', 'init', '-q', str(self.repo)], check=True)
        subprocess.run(['git', '-C', str(self.repo), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(self.repo), '-c', 'user.name=Fixture',
                        '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture'], check=True)
        return directory

    def test_deepseek_teacher_capture_binds_clean_ranks_and_dumps(self):
        directory = self.teacher_fixture()
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        meta = dict(line.split("=", 1) for line in (directory / "manifest").read_text().splitlines())
        self.assertEqual(meta["teacher_arm"], "deepseek-ordinary")
        self.assertEqual(meta["model_arch"], "deepseek4")
        self.assertEqual(meta["teacher_positions"], "2")
        self.assertIn("negotiated=0x000780c9", meta["coordinator_features"])
        terminal_proof(self.out / "fixture.tsv", meta, required=True)
        self.assertEqual(len((directory / "files.sha256").read_text().splitlines()), 2)

    def test_teacher_scores_the_hashed_fixture_from_foreign_cwd(self):
        directory = self.teacher_fixture()
        foreign = self.root / 'foreign'
        foreign.mkdir()
        for parent, contents in ((self.repo, 'trusted repository prompt'),
                                 (foreign, 'different caller prompt')):
            (parent / 'prompt.txt').write_text(contents)
            (parent / 'continuation.txt').write_text('fixture continuation')
        self.inputs.write_text('fixture\tprompt.txt\tcontinuation.txt\n')
        self.env['DS4_QUALITY_MANIFEST'] = '../manifest.tsv'
        self.model = Path('../unchanged.gguf')
        scorer = self.repo / 'gguf-tools/quality-testing/score_official'
        program = scorer.read_text().replace(
            'from pathlib import Path\n',
            'from pathlib import Path\n'
            "prompt_path = Path(Path(sys.argv[2]).read_text().split('\\t')[1])\n"
            "if prompt_path.read_text() != 'trusted repository prompt':\n"
            '    raise SystemExit(23)\n', 1)
        scorer.write_text(program)
        result = self.run_launcher(cwd=foreign)
        self.assertEqual(result.returncode, 0, result.stderr)
        meta = dict(line.split('=', 1) for line in (directory / 'manifest').read_text().splitlines())
        expected = subprocess.check_output([
            sys.executable, str(self.repo / 'scripts/compare-teacher-logits.py'),
            '--validate-capture', '--root', str(self.repo), '--fixture', str(self.inputs),
            '--start-case', '0', '--cases', '1'], text=True).strip()
        self.assertEqual(meta['fixture_content_sha256'], expected)
        terminal_proof(self.out / 'fixture.tsv', meta, required=True)

    def test_deepseek_teacher_rejects_wrong_architecture(self):
        self.teacher_fixture(architecture="glm5-next")
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.peer / "worker-fixture.pid").exists())

    def test_deepseek_teacher_rejects_glm_label(self):
        self.teacher_fixture(arm="kda-tp")
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.peer / "worker-fixture.pid").exists())

    def test_glm_teacher_arm_still_completes(self):
        directory = self.teacher_fixture(architecture='glm5-next', arm='kda-tp')
        result = self.run_launcher('DS4_GLM5_KDA_TP=1', 'DS4_GLM5_KDA_OUTPUT_KSLICE=0',
                                   'DS4_GLM5_NEXT_PREFILL_BATCH=256')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('teacher_arm=kda-tp\n', (directory / 'manifest').read_text())

    def test_deepseek_sdk_comparison_is_attested_and_diagnostic_only(self):
        first = self.teacher_fixture()
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        second = first.with_name('teacher-second')
        self.env['DS4_QUALITY_TEACHER_LOGITS_DIR'] = str(second)
        result = self.run_launcher(tag='fixture-second')
        self.assertEqual(result.returncode, 0, result.stderr)
        command = [sys.executable, str(REPO / 'scripts/compare-teacher-logits.py'),
                   str(first), str(second), '--score-arm-mode', 'deepseek-sdk']
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNone(json.loads(result.stdout)['passed'])
        self.assertEqual(json.loads(result.stdout)['argmax_mismatches'], 0)
        # Independent SDK builds may have different executable identities.
        manifest = second / 'manifest'
        original = manifest.read_text()
        metadata = dict(line.split('=', 1) for line in original.splitlines())
        changed = original.replace('ds4_sha256=' + metadata['ds4_sha256'], 'ds4_sha256=' + 'a' * 64)
        changed = changed.replace('scorer_sha256=' + metadata['scorer_sha256'], 'scorer_sha256=' + 'b' * 64)
        manifest.write_text(changed)
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for key, value in (('fixture_content_sha256', '0' * 64), ('inference_source_commit', 'a' * 40),
                           ('context', '8192'), ('teacher_positions', '3')):
            with self.subTest(field=key):
                manifest.write_text(changed.replace(key + '=' + metadata[key], key + '=' + value))
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(key, result.stderr)
        manifest.write_text(original)
        result = subprocess.run(command + ['--thresholds', '/not/read.json'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('diagnostic-only', result.stderr)
        (second / 'decode_000001.logits.json').write_text('{}')
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('changed teacher dump', result.stderr)

    def test_deepseek_teacher_rejects_native_mtp(self):
        self.teacher_fixture()
        result = self.run_launcher("DS4_GLM5_NATIVE_DRAFT=6")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.peer / "worker-fixture.pid").exists())

    def test_deepseek_teacher_rejects_disabled_rdma_logits(self):
        self.teacher_fixture()
        result = self.run_launcher("DS4_TP_RDMA_LOGITS=0")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.peer / "worker-fixture.pid").exists())

    def test_deepseek_teacher_refuses_odinlink(self):
        self.teacher_fixture()
        self.env.update(DS4_QUALITY_RDMA_PROFILE='odinlink', DS4_ODINLINK_ROOT=str(self.root))
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires deepseek4 over RoCE v2', result.stderr)
        self.assertFalse((self.peer / "worker-fixture.pid").exists())

    def test_teacher_explicit_inference_identity(self):
        directory = self.teacher_fixture()
        self.env.update(DS4_QUALITY_INFERENCE_SOURCE_COMMIT=subprocess.check_output(
            ['git', '-C', str(self.repo), 'rev-parse', 'HEAD'], text=True).strip(),
            DS4_QUALITY_EXPECT_DS4_SHA256=hashlib.sha256((self.repo / 'ds4').read_bytes()).hexdigest(),
            DS4_QUALITY_EXPECT_SCORER_SHA256=hashlib.sha256(
                (self.repo / 'gguf-tools/quality-testing/score_official').read_bytes()).hexdigest())
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('inference_source_commit=' + self.env['DS4_QUALITY_INFERENCE_SOURCE_COMMIT'],
                      (directory / 'manifest').read_text())

    def test_teacher_refuses_wrong_inference_binary_pin(self):
        self.teacher_fixture()
        self.env.update(DS4_QUALITY_INFERENCE_SOURCE_COMMIT='a' * 40,
                        DS4_QUALITY_EXPECT_DS4_SHA256='0' * 64,
                        DS4_QUALITY_EXPECT_SCORER_SHA256='0' * 64)
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires matching ds4/scorer hashes', result.stderr)
        self.assertFalse((self.peer / 'worker-fixture.pid').exists())

    def test_deepseek_teacher_rejects_negotiation_or_capture_failures(self):
        cases = (("DS4_TEST_NEGOTIATION=missing",),
                 ("DS4_TEST_NEGOTIATION=mismatch",),
                 ("DS4_TEST_DUMP_MODE=missing",),
                 ("DS4_TEST_DUMP_MODE=extra",),
                 ("DS4_TEST_DUMP_MODE=nonfinite",),
                 ("DS4_TEST_DUMP_MODE=wrong-case",),
                 ("DS4_TEST_DUMP_MODE=wrong-step",),
                 ("DS4_TEST_DUMP_MODE=truncated",),
                 ("DS4_TEST_TRANSPORT=missing",),
                 ("DS4_TEST_TRANSPORT=fallback",),
                 ("DS4_TEST_TRANSPORT=tcp",),
                 ("DS4_TEST_TRANSPORT=no-gid",),
                 ("DS4_TEST_TRANSPORT=cache",),
                 ("DS4_TEST_WORKER_RC=7",))
        for settings in cases:
            with self.subTest(settings=settings):
                # Each negative case needs fresh launcher paths and rank identities.
                fixture = QualityCompletion(methodName="runTest")
                fixture.setUp()
                try:
                    directory = fixture.teacher_fixture()
                    result = fixture.run_launcher(*settings)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse((directory / "manifest").exists())
                    self.assertFalse((fixture.out / "fixture.manifest").exists())
                finally:
                    fixture.doCleanups()

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
