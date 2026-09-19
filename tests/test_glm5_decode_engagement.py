#!/usr/bin/env python3
"""Exercise the launcher's actual engagement check with scalar/native logs."""
from pathlib import Path
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = (REPO / 'run-tp-ds4-bench.sh').read_text()
START = 'if [[ $GLM5_BF16_WMMA_HILO == 1 ||\n'
END = 'if [[ $DECODE_SELF_CHECK == 1 ]]; then\n'
CHECK = START + LAUNCHER.split(START, 1)[1].split(END, 1)[0]
SUMMARY = ('ds4: GLM5 BF16 WMMA hi/lo summary q=0 k=0 v=0 qkv_fused=34 '
           'kda_six_fused=0 output=34 other=0 not_applicable=10 hard_failure=0\n')
CONFIG = ('ds4: native GLM5 config proposals=5 hello=0x0000178800000000 '
          'verifier_workspaces=2/4/6\n')
SCALAR = 'ds4: GLM5 BF16 decode QKV multiptr engaged out_dim=4096\n'


class DecodeEngagement(unittest.TestCase):
    def run_check(self, native=False, scalar=False, missing_rank=None,
                  wrong_rows=False, small_m=True, bad_summary=False):
        with tempfile.TemporaryDirectory() as directory:
            logs = [Path(directory) / name for name in ('coordinator', 'worker')]
            for rank, path in enumerate(logs):
                batch = (f'ds4: GLM5 native KDA verifier batch engaged rank={rank} '
                         f'rows={8 if wrong_rows else 6} qkv=bf16-small-m-exact\n')
                path.write_text(SUMMARY.replace('hard_failure=0', 'hard_failure=1')
                                if bad_summary else SUMMARY)
                with path.open('a') as stream:
                    stream.write(SCALAR if scalar else '')
                    stream.write(CONFIG + (batch if missing_rank != rank else ''))
            script = '''set -euo pipefail
MODEL_ARCH=glm5-next
GLM5_BF16_WMMA_HILO=1
GLM5_BF16_WMMA_QKV_FUSED=1
GLM5_BF16_QKV_DECODE_MULTIPTR=1
GLM5_BF16_QKV_SHARED_A_PREFILL=0
GLM5_BF16_KDA_SIX_DECODE_MULTIPTR=0
GLM5_BF16_KDA_SIX_MULTIPTR=0
GLM5_BF16_KDA_SIX_PREFILL=0
COORD_LOG=$1
WORKER_LOG=$2
COORD_ENV=("DS4_GLM5_NATIVE_DRAFT=$3" "DS4_ROCM_GLM5_BF16_SMALL_M_EXACT=$4")
''' + CHECK
            return subprocess.run(['bash', '-c', script, 'engagement', *map(str, logs),
                                   '6' if native else '0', '1' if small_m else '0'],
                                  capture_output=True, text=True)

    def test_ordinary_requires_scalar_engagement(self):
        self.assertEqual(self.run_check(scalar=True).returncode, 0)
        self.assertNotEqual(self.run_check().returncode, 0)

    def test_native_without_scalar_tail_has_its_own_proof(self):
        result = self.run_check(native=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_still_requires_both_ranks(self):
        for rank in (0, 1):
            self.assertNotEqual(self.run_check(native=True, missing_rank=rank).returncode, 0)

    def test_native_rejects_wrong_width_or_disabled_exact_path(self):
        self.assertNotEqual(self.run_check(native=True, wrong_rows=True).returncode, 0)
        self.assertNotEqual(self.run_check(native=True, small_m=False).returncode, 0)

    def test_native_scalar_tail_and_prefill_proof(self):
        self.assertEqual(self.run_check(native=True, scalar=True).returncode, 0)
        self.assertNotEqual(self.run_check(native=True, bad_summary=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
