#!/usr/bin/env python3
"""A redundant launcher assignment must not change an effective control anchor."""
import copy
from pathlib import Path
import runpy
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
gate = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/candidate-gate.py'))


class ControlAnchorEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.environment = {
            'common_env': 'DS4_BENCH_RUN_ID=baseline',
            'coordinator_env': 'DS4_BENCH_RUN_ID=baseline DS4_METAL_MEMORY_REPORT=1',
            'worker_env': 'DS4_BENCH_RUN_ID=baseline DS4_METAL_MEMORY_REPORT=1',
            'extra_env': 'DS4_METAL_MEMORY_REPORT=1',
        }
        self.control = dict(self.environment, ds4_sha256='a' * 64)
        self.control['common_env'] += ' DS4_METAL_MEMORY_REPORT=1'
        self.baseline = {'key': {'source_commit': 'b' * 40}, 'reference': {
            'performance': {'roce-v2': {'environment': self.environment,
                                       'ds4_sha256': 'a' * 64}}}}
        self.proof = {'required_provider': 'roce-v2',
                      'source_commits': {'control': 'b' * 40},
                      'headline_pairs': [{'control': {'manifest': self.control}}]}

    def verify(self):
        gate['verify_control_anchor'](self.proof, self.baseline, {})

    def test_common_assignment_redundant_in_both_ranks_and_extra(self):
        self.verify()

    def test_reverse_representation_is_equivalent(self):
        self.environment['common_env'], self.control['common_env'] = (
            self.control['common_env'], self.environment['common_env'])
        self.verify()

    def test_actual_rank_or_extra_change_is_rejected(self):
        for field in ('coordinator_env', 'worker_env', 'extra_env'):
            with self.subTest(field=field):
                original = self.control[field]
                self.control[field] = original.replace('MEMORY_REPORT=1', 'MEMORY_REPORT=0')
                with self.assertRaises(gate['GateError']):
                    self.verify()
                self.control[field] = original

    def test_inconsistent_common_value_is_rejected(self):
        self.control['common_env'] = self.control['common_env'].replace('REPORT=1', 'REPORT=0')
        with self.assertRaises(gate['GateError']):
            self.verify()

    def test_missing_effective_assignment_is_not_a_default(self):
        for field in ('coordinator_env', 'worker_env', 'extra_env'):
            with self.subTest(field=field):
                original = self.environment[field]
                self.environment[field] = ''
                self.control[field] = ''
                with self.assertRaises(gate['GateError']):
                    self.verify()
                self.environment[field] = self.control[field] = original

    def test_other_common_difference_is_rejected(self):
        self.control['common_env'] += ' DS4_UNKNOWN_KERNEL=1'
        with self.assertRaises(gate['GateError']):
            self.verify()

    def test_every_control_is_checked_without_mutation(self):
        before = copy.deepcopy((self.proof, self.baseline))
        self.verify()
        self.assertEqual((self.proof, self.baseline), before)
        later = copy.deepcopy(self.control)
        later['worker_env'] += ' DS4_UNKNOWN_KERNEL=1'
        self.proof['headline_pairs'].append({'control': {'manifest': later}})
        with self.assertRaises(gate['GateError']):
            self.verify()


if __name__ == '__main__':
    unittest.main()
