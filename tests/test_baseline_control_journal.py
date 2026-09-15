#!/usr/bin/env python3
"""Baseline-source controls must survive sealing and invalidation replay."""
import json
import os
from pathlib import Path
import sys
import tempfile

if os.environ.get('DS4_GATE_CLEAN_TEST') != '1':
    raise SystemExit('run via tests/run-clean-gate-python.sh')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_candidate_gate as fixture


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        genesis = fixture.genesis_fixture.build_fixture(root, providers=('roce-v2',))
        created = fixture.genesis_fixture.run_gate(root, genesis)
        assert created.returncode == 0, created.stderr
        baseline_id = created.stdout.strip()
        baseline = json.loads((root / 'baselines/sha256' / (baseline_id[7:] + '.json')).read_text())
        baseline_source = baseline['key']['source_commit']
        fixture.advance_source_commit('test: candidate differs from its baseline source')
        assert fixture.current_head() != baseline_source
        candidate_id = 'baseline-source-journal'
        initialized = fixture.run(root, 'init', candidate_id, 'A', '--switch', 'DS4_FEATURE=0,1')
        assert initialized.returncode == 0, initialized.stderr
        fixture.configure_candidate(root, candidate_id, baseline_id, baseline, None)
        begun = fixture.run(root, 'begin-pair', candidate_id)
        assert begun.returncode == 0, begun.stderr
        pair = json.loads(begun.stdout)
        control, run_id = fixture.make_headline_run(
            root, candidate_id, baseline_id, baseline, pair, 'control', 1, 'baseline-source-control')
        manifest_path = Path(control['manifest'])
        manifest = fixture.read_manifest(manifest_path)
        manifest['source_commit'] = baseline_source
        fixture.write_manifest(manifest_path, manifest)
        # A genuine failed first control arm also must be invalidatable/replayable.
        csv = Path(control['csv'])
        csv.write_text(csv.read_text().splitlines()[0] + '\n')
        log = Path(control['coordinator_log'])
        log.write_text(log.read_text().replace(
            'ds4-bench-launcher: headline_csv_complete=1',
            'ds4-bench-launcher: headline_csv_complete=0').replace(
            'ds4-bench: headline_csv_complete=1\n', ''))
        for rank in ('coordinator', 'worker'):
            fixture.genesis_fixture.write_manifest(Path(control[rank + '_status']),
                                                   {'exit_code': 1, 'signal': 0})
        recorded = fixture.run(root, 'record-run', candidate_id, pair['pair_id'], 'AB', 'control', run_id)
        assert recorded.returncode == 0, recorded.stderr
        fixture.record_result(root, candidate_id, pair['pair_id'], run_id, control)
        invalidated = fixture.run(root, 'invalidate-pair', candidate_id, pair['pair_id'],
            '--run-id', run_id, '--manifest', control['manifest'],
            '--coordinator-log', control['coordinator_log'], '--coordinator-status', control['coordinator_status'],
            '--worker-log', control['worker_log'], '--worker-status', control['worker_status'],
            '--reason', 'injected pre-result baseline-source control failure')
        assert invalidated.returncode == 0, invalidated.stderr
        # Beginning replacement reopens and validates the invalidated control event.
        replacement = fixture.run(root, 'begin-pair', candidate_id)
        assert replacement.returncode == 0, replacement.stderr
        pair = json.loads(replacement.stdout)
        value = json.loads((root / 'candidates' / candidate_id / 'candidate.json').read_text())
        matches = fixture.GATE_MODULE.headline_source_matches
        assert matches(root, value, 'control', baseline_source)
        assert matches(root, value, 'control', fixture.current_head())
        assert not matches(root, value, 'candidate', baseline_source)
        assert not matches(root, value, 'control', 'f' * 40)
        assert not matches(root, value, 'control', None)
        for offset, arm in enumerate(('control', 'candidate'), start=1):
            artifact, run_id = fixture.make_headline_run(
                root, candidate_id, baseline_id, baseline, pair, arm, offset,
                f'replacement-{arm}')
            path = Path(artifact['manifest'])
            fields = fixture.read_manifest(path)
            fields['source_commit'] = baseline_source if arm == 'control' else fixture.current_head()
            fixture.write_manifest(path, fields)
            recorded = fixture.run(root, 'record-run', candidate_id, pair['pair_id'],
                                   'AB', arm, run_id)
            assert recorded.returncode == 0, recorded.stderr
            fixture.record_result(root, candidate_id, pair['pair_id'], run_id, artifact)
        assert fixture.run(root, 'begin-pair', candidate_id).returncode == 0
        print('PASS baseline-source control sealing, invalidation and journal replay')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
