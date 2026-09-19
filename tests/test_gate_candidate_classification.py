#!/usr/bin/env python3
"""Archival research notes must not masquerade as open formal candidates."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
spec = importlib.util.spec_from_file_location('gate_classification', REPO / 'scripts/candidate-gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class CandidateClassification(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # Scope authentication has its own end-to-end fixtures. Isolate this
        # guard's treatment of journal identity, lifecycle and legacy notes.
        self.scope = patch.object(gate, 'candidate_scope', side_effect=lambda root, value: value.get('baseline_id'))
        self.scope.start()
        self.addCleanup(self.scope.stop)

    def note(self, name='old-note', **fields):
        path = self.root / 'candidates' / name / 'candidate.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(candidate_id=name, baseline_id='', **fields)))
        return path

    def event(self, kind='candidate-init', name='old-note'):
        gate.append_event(self.root, dict(type=kind, candidate_id=name))

    def check(self):
        gate.reject_open_scope_candidates(self.root, 'scope-a')

    def blocked(self):
        with self.assertRaises(gate.GateError):
            self.check()

    def test_archival_schemas_do_not_require_closing(self):
        for i, fields in enumerate(({}, {'schema_version': 1}, {'schema_version': 2})):
            path = self.note('archive-' + str(i), **fields)
            before = path.read_bytes()
            self.check()
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse((path.parent / 'CLOSED.json').exists())
        self.assertFalse((self.root / 'gate-governance').exists())

    def test_any_intent_field_blocks(self):
        for intent in (None, {}, {'target_metrics': ['decode']}):
            self.note(promotion_intent=intent)
            self.blocked()

    def test_initialized_note_without_intent_blocks(self):
        self.note()
        self.event()
        self.blocked()

    def test_registered_unresolvable_scope_blocks(self):
        path = self.note()
        path.write_text(json.dumps(dict(candidate_id='old-note', baseline_id='missing-baseline')))
        self.event()
        with patch.object(gate, 'candidate_scope', return_value=None):
            self.blocked()

    def test_directory_identity_survives_removed_or_changed_id(self):
        path = self.note()
        self.event()
        for fields in ({'baseline_id': ''}, {'baseline_id': '', 'candidate_id': 'renamed'}):
            path.write_text(json.dumps(fields))
            self.blocked()

    def test_declared_identity_cannot_be_relocated_to_hide_initialization(self):
        path = self.note()
        self.event()
        moved = self.root / 'candidates' / 'relocated'
        path.parent.rename(moved)
        self.blocked()

    def test_missing_open_dossier_blocks(self):
        self.event()
        self.blocked()

    def test_unjournaled_closed_marker_cannot_hide_registered_candidate(self):
        path = self.note()
        self.event()
        (path.parent / 'CLOSED.json').write_text('{}')
        self.blocked()

    def test_journal_finished_candidates_do_not_block(self):
        for kind in ('candidate-close', 'candidate-promote'):
            name = kind
            self.note(name, promotion_intent=None)
            self.event(name=name)
            self.event(kind, name=name)
        self.check()

    def test_assigned_scope_keeps_existing_behavior(self):
        path = self.note()
        path.write_text(json.dumps(dict(candidate_id='old-note', baseline_id='scope-a')))
        self.blocked()
        path.write_text(json.dumps(dict(candidate_id='old-note', baseline_id='scope-b')))
        self.check()

    def test_corrupt_journal_is_not_ignored_for_legacy_notes(self):
        self.note()
        self.event()
        path = self.root / 'gate-governance/events.jsonl'
        value = json.loads(path.read_text())
        value['event']['candidate_id'] = 'tampered'
        path.write_text(json.dumps(value) + '\n')
        with self.assertRaises(gate.ControlError):
            self.check()

    def test_duplicate_initialization_is_rejected(self):
        self.note()
        self.event()
        self.event()
        with self.assertRaises(gate.ControlError):
            self.check()


if __name__ == '__main__':
    unittest.main()
