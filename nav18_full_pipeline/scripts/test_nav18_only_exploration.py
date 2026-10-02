"""Contract tests for isolated Nav1.8 exploration; no docking executable is run."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nav18_only_exploration as runner


class ExplorationContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = {
            'mode': runner.MODE,
            'files': {'receptor.pdbqt': 'receptor-digest', 'lipid.sdf': 'lipid-digest'},
            'center': list(runner.CENTER), 'size': runner.SIZE,
            'exhaustiveness': 8, 'num_modes': 9, 'cpu': 2, 'num_confs': 20,
            'versions': {'rdkit': 'test'}, 'vina_sha256': 'vina-digest',
            'production_code_hashes': {'score.py': 'original-score-digest'},
            'exploration_code_hashes': {'nav18_only_exploration.py': 'runner-digest'},
        }

    def reference_fixture(self, passed=15):
        folder = self.root / 'reference'
        folder.mkdir(exist_ok=True)
        rows = []
        for repeat in range(runner.REPEATS):
            for seed in runner.SEEDS:
                ok = len(rows) < passed
                row = {'repeat': repeat, 'vina_seed': seed, 'status': 'ok',
                       'reference_passed': ok, 'top1_passed': ok,
                       'top1_rmsd_A': 0.8 if ok else 4.0,
                       'quality': {'quality_status': 'pass'}}
                rows.append(row)
                runner.write_json(folder / 'jobs' / f'c{repeat}_s{seed}' / 'result.json', row)
        runner.write_json(folder / 'results.json', rows)
        evidence = {str(p.relative_to(folder)): runner.sha256(p)
                    for p in folder.rglob('*.json') if p.name != 'report.json'}
        report = runner.gate_summary(rows, self.protocol, evidence)
        path = folder / 'report.json'
        runner.write_json(path, report)
        return path, report, rows

    def check(self, path):
        return runner.check_reference_gate(path, self.protocol, runner.SEEDS[0])

    def test_twelve_of_fifteen_reference_passes_are_sufficient(self):
        path, _, _ = self.reference_fixture(passed=12)
        self.assertTrue(self.check(path)['nav18_exploration_eligible'])
        self.assertFalse(self.check(path)['production_gate_released'])

    def test_eleven_reference_passes_do_not_release_exploration(self):
        path, _, _ = self.reference_fixture(passed=11)
        with self.assertRaises(ValueError):
            self.check(path)

    def test_protocol_changes_are_not_covered_by_reference(self):
        path, _, _ = self.reference_fixture()
        changes = {'center': [0, 0, 0], 'size': 26, 'exhaustiveness': 16,
                   'num_modes': 12, 'cpu': 4, 'num_confs': 40,
                   'files': {'receptor.pdbqt': 'different'},
                   'versions': {'rdkit': 'different'}, 'vina_sha256': 'different',
                   'production_code_hashes': {'score.py': 'different'},
                   'exploration_code_hashes': {'nav18_only_exploration.py': 'different'}}
        for key, value in changes.items():
            with self.subTest(parameter=key):
                changed = {**self.protocol, key: value}
                with self.assertRaises(ValueError):
                    runner.check_reference_gate(path, changed, runner.SEEDS[0])

    def test_unvalidated_seed_is_rejected(self):
        path, _, _ = self.reference_fixture()
        with self.assertRaises(ValueError):
            runner.check_reference_gate(path, self.protocol, 99999)

    def test_pass_flag_cannot_override_incomplete_or_wrong_mode_report(self):
        path, report, _ = self.reference_fixture()
        mutations = [dict(successful=14), dict(top1_pass_count=11), dict(evidence={}),
                     dict(stage='reference'), dict(production_gate_released=True),
                     dict(nav18_exploration_eligible=False)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                runner.write_json(path, {**report, **mutation})
                with self.assertRaises(ValueError):
                    self.check(path)

    def test_changed_reference_evidence_is_rejected(self):
        path, _, _ = self.reference_fixture()
        with (path.parent / 'results.json').open('a', encoding='utf-8') as handle:
            handle.write(' ')
        with self.assertRaises(ValueError):
            self.check(path)

    def test_arbitrary_hashed_file_does_not_prove_reference_passed(self):
        path, report, _ = self.reference_fixture()
        unrelated = path.parent / 'unrelated.txt'
        unrelated.write_text('not a reference experiment', encoding='utf-8')
        report['evidence'] = {unrelated.name: runner.sha256(unrelated)}
        runner.write_json(path, report)
        with self.assertRaises(ValueError):
            self.check(path)

    def test_forged_pass_counts_conflicting_with_hashed_results_are_rejected(self):
        path, report, _ = self.reference_fixture(passed=11)
        report.update(nav18_exploration_eligible=True, top1_pass_count=15)
        runner.write_json(path, report)
        with self.assertRaises(ValueError):
            self.check(path)

    def test_duplicate_reference_jobs_cannot_count_as_fifteen_independent_jobs(self):
        path, report, rows = self.reference_fixture()
        rows[-1] = copy.deepcopy(rows[0])
        runner.write_json(path.parent / 'results.json', rows)
        report['evidence']['results.json'] = runner.sha256(path.parent / 'results.json')
        runner.write_json(path, report)
        with self.assertRaises(ValueError):
            self.check(path)

    def test_batch_refuses_to_schedule_candidates_before_gate_passes(self):
        path, _, _ = self.reference_fixture(passed=11)
        with patch.object(runner, 'run_one') as run_one:
            with self.assertRaises(ValueError):
                runner.run_batch([{'candidate_id': 'a', 'smiles': 'CCO'}], self.root,
                                 self.root, self.protocol,
                                 SimpleNamespace(workers=1, resume=False), 'library')
        run_one.assert_not_called()

    def test_checkpoint_preserves_all_ids_order_metadata_and_unknown_values(self):
        entries = [{'candidate_id': 'z', 'smiles': 'CCO', 'methods': '["mol2mol"]'},
                   {'candidate_id': 'a', 'smiles': 'CCN', 'methods': '["site_enumeration"]'},
                   {'candidate_id': 'm', 'smiles': 'CC', 'methods': '["amide_recombination"]'}]
        finished = {**entries[1], **runner.unknown_fields(), 'nav18_score': -8.2,
                    'docking_status': 'ok', 'quality_status': 'pass'}
        summary = runner.checkpoint(self.root, entries, {'a': finished}, 'library')
        rows = runner.read_json(self.root / 'library_results.json')
        self.assertEqual([r['candidate_id'] for r in rows], ['z', 'a', 'm'])
        self.assertEqual(summary['expected'], 3)
        self.assertEqual(summary['completed'], 1)
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['status_counts'], {'queued': 2, 'ok': 1})
        for source, row in zip(entries, rows):
            self.assertEqual(row['methods'], source['methods'])
            self.assertEqual(row['mode'], runner.MODE)
            self.assertEqual(row['nav15_risk'], 'unknown')
            self.assertFalse(row['production_gate_released'])
            for key in ('nav15_score', 'delta', 'combined_score'):
                self.assertIsNone(row[key])
        with (self.root / 'library_status.csv').open(encoding='utf-8-sig', newline='') as handle:
            table = list(csv.DictReader(handle))
        self.assertEqual(len(table), 3)
        self.assertTrue(all(row[k] == '' for row in table
                            for k in ('nav15_score', 'delta', 'combined_score')))
        self.assertFalse((self.root / 'library_raw_score_order.csv').exists())

    def test_completed_ranking_only_includes_quality_passing_docked_rows(self):
        entries = [{'candidate_id': key, 'smiles': 'CC'} for key in ('good', 'flagged', 'deferred')]
        results = {r['candidate_id']: {**r, **runner.unknown_fields(),
                   'docking_status': 'deferred' if r['candidate_id'] == 'deferred' else 'ok',
                   'quality_status': 'pass' if r['candidate_id'] == 'good' else 'review',
                   'nav18_score': -5 if r['candidate_id'] == 'good' else -20} for r in entries}
        runner.checkpoint(self.root, entries, results, 'library')
        with (self.root / 'library_raw_score_order.csv').open(encoding='utf-8-sig', newline='') as handle:
            self.assertEqual([r['candidate_id'] for r in csv.DictReader(handle)], ['good'])

    def test_unspecified_stereochemistry_defers_without_preparation_or_docking(self):
        entry = {'candidate_id': 'stereo_pending', 'smiles': 'CC(O)F', 'scope': 'outside_series'}
        with patch.object(runner, 'prep_ligand') as prep, patch.object(runner, 'recorded_dock') as dock:
            row = runner.run_one(entry, self.root, self.root, resume=False)
        self.assertEqual(row['docking_status'], 'deferred')
        self.assertEqual(row['defer_reason'], 'requires_stereochemistry_review')
        self.assertTrue(row['unspecified_stereo'])
        self.assertIsNone(row['nav18_score'])
        prep.assert_not_called()
        dock.assert_not_called()

    def completed_candidate(self):
        entry = {'candidate_id': 'complete_candidate', 'smiles': 'CCO'}
        folder = runner.candidate_folder(self.root, entry['candidate_id'])
        folder.mkdir(parents=True)
        pose = folder / 'poses.pdbqt'
        pose.write_text('mock completed pose', encoding='utf-8')
        row = {**entry, **runner.unknown_fields(), 'docking_status': 'ok',
               'quality_status': 'pass', 'nav18_score': -7.1,
               'artifacts': {pose.name: runner.sha256(pose)}}
        runner.write_json(folder / 'result.json', row)
        return entry, folder, pose, row

    def test_intact_successful_candidate_resume_does_not_repeat_docking(self):
        entry, _, _, expected = self.completed_candidate()
        with patch.object(runner, 'prep_ligand') as prep, patch.object(runner, 'recorded_dock') as dock:
            row = runner.run_one(entry, self.root, self.root, resume=True)
        self.assertEqual(row, expected)
        prep.assert_not_called()
        dock.assert_not_called()

    def test_changed_successful_pose_is_rejected_on_resume(self):
        entry, _, pose, _ = self.completed_candidate()
        pose.write_text('changed pose', encoding='utf-8')
        with patch.object(runner, 'recorded_dock') as dock:
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                runner.run_one(entry, self.root, self.root, resume=True)
        dock.assert_not_called()

    def test_changed_identity_is_rejected_on_resume(self):
        entry, _, _, _ = self.completed_candidate()
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            runner.run_one({**entry, 'smiles': 'CCN'}, self.root, self.root, resume=True)

    def test_changed_input_hash_is_rejected_by_run_manifest(self):
        directory = self.root / 'run'
        signature = {'protocol': self.protocol, 'input_sha256': 'original-input'}
        runner.work_manifest(directory, signature)
        runner.work_manifest(directory, signature, resume=True)
        with self.assertRaisesRegex(ValueError, 'Resume inputs'):
            runner.work_manifest(directory, {**signature, 'input_sha256': 'changed-input'}, resume=True)

    def test_changed_frozen_receptor_is_rejected(self):
        source = self.root / 'source.pdbqt'
        source.write_text('original receptor', encoding='utf-8')
        signature = {'protocol': {'files': {'receptor.pdbqt': runner.sha256(source)}}}
        with patch.object(runner, 'SOURCES', {'receptor.pdbqt': source}):
            folder = runner.snapshot(self.root, signature)
            (folder / 'receptor.pdbqt').write_text('changed receptor', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Frozen artifact changed'):
                runner.snapshot(self.root, signature)


if __name__ == '__main__':
    unittest.main()
