import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import score
from scgd import runtime, workflow
from scgd.geometry import fixed_frame_rmsd
from scgd.ligand_quality import prepare_candidate
from scgd.prep import prep_ligand, PreparationReviewRequired
from scgd.validation import reference_molecule, summarize


def test_original_scoring_functions_and_weights_are_unchanged():
    tree = ast.parse(Path(score.__file__).read_text(encoding='utf-8-sig'))
    frozen = runtime.read_json(runtime.BASE / 'data/reference_validation/scoring_contract.json')
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in frozen:
            assert hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() == frozen.pop(node.name)
    assert not frozen


def test_bom_csv_preserves_ids_multiline_metadata_and_bad_structures(tmp_path):
    path = tmp_path / 'library.csv'
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['candidate_id', 'smiles', 'scope', 'note'])
        writer.writeheader()
        writer.writerows([{'candidate_id': 'original_1', 'smiles': 'CCO', 'scope': 'strict', 'note': 'line1\nline2'},
                         {'candidate_id': 'empty_structure', 'smiles': ''}])
    entries = workflow.read_entries(path)
    assert entries[0]['candidate_id'] == 'original_1' and entries[0]['note'] == 'line1\nline2'
    assert score.read_molecules(str(path)) == [('original_1', 'CCO'), ('empty_structure', '')]
    assert workflow.input_audit(entries, tmp_path)['state_status']['invalid_smiles'] == 1
    with pytest.raises(ValueError, match='absent'):
        workflow.read_entries(path, label_column='missing')
    path.write_text('candidate_id,smiles\nsame,CCO\nsame,CCC\n', encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate'):
        workflow.read_entries(path)


def test_checked_ligand_preserves_charge_stereo_and_can_resume(tmp_path):
    path = tmp_path / 'ligand.pdbqt'
    prep_ligand('C[C@H](O)F', path, candidate_id='chiral', num_confs=3)
    record = runtime.read_json(path.with_suffix('.preparation.json'))
    assert record['status'] == 'ok'
    assert record['state']['state_smiles'] == 'C[C@H](O)F'
    assert all(r['minimization_return_codes'][-1] == 0 for r in record['conformers'] if r['status'] == 'retained')
    original = path.read_bytes()
    prep_ligand('C[C@H](O)F', path, candidate_id='chiral', num_confs=3, resume=True)
    assert path.read_bytes() == original
    with pytest.raises(PreparationReviewRequired, match='stereochemistry'):
        prep_ligand('CC(O)F', tmp_path / 'ambiguous.pdbqt', num_confs=2)
    state, _, _, kept = prepare_candidate({'candidate_id': 'ion', 'smiles': 'C[NH2+]C'}, num_confs=2)
    assert state['formal_charge'] == 1 and kept


def test_nonconverged_conformers_never_reach_docking(monkeypatch):
    class BadForceField:
        def Minimize(self, **kwargs):
            return 1
        def CalcEnergy(self):
            return -10.0
    monkeypatch.setattr('scgd.ligand_quality.AllChem.MMFFGetMoleculeForceField', lambda *a, **kw: BadForceField())
    state, records, _, kept = prepare_candidate({'candidate_id': 'bad', 'smiles': 'CCO'}, num_confs=2)
    assert not kept and state['status'] == 'no_usable_conformer'
    assert records and all(r['status'] == 'optimization_not_converged' for r in records)


def test_docking_resume_is_per_target_and_checks_hashes(tmp_path, monkeypatch):
    vina, ligand, receptor = [tmp_path / s for s in ('vina.exe', 'lig.pdbqt', 'rec.pdbqt')]
    for path in (vina, ligand, receptor):
        path.write_text(path.name)
    calls = []
    def run(command, log, timeout):
        calls.append(command)
        Path(command[command.index('--out') + 1]).write_text('REMARK VINA RESULT: -7.0 0 0\n')
    monkeypatch.setattr(runtime, 'logged_run', run)
    pose = tmp_path / 'pose.pdbqt'
    opts = {'vina': vina, 'seed': 42}
    runtime.dock_job(ligand, receptor, pose, (0, 0, 0), 24, **opts)
    runtime.dock_job(ligand, receptor, pose, (0, 0, 0), 24, resume=True, **opts)
    assert len(calls) == 1
    with pytest.raises(ValueError, match='signature changed'):
        runtime.dock_job(ligand, receptor, pose, (0, 0, 0), 24, resume=True, vina=vina, seed=43)
    pose.write_text('tampered')
    with pytest.raises(ValueError, match='changed'):
        runtime.dock_job(ligand, receptor, pose, (0, 0, 0), 24, resume=True, **opts)


def test_failed_job_retries_and_timeout_is_logged(tmp_path, monkeypatch):
    for name in ('vina.exe', 'lig', 'rec'):
        (tmp_path / name).write_text(name)
    def fail(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 1, output='partial stdout')
    monkeypatch.setattr(runtime.subprocess, 'run', fail)
    pose = tmp_path / 'pose.pdbqt'
    with pytest.raises(subprocess.TimeoutExpired):
        runtime.dock_job(tmp_path / 'lig', tmp_path / 'rec', pose, (0, 0, 0), 24, vina=tmp_path / 'vina.exe')
    assert runtime.read_json(pose.with_suffix('.job.json'))['status'] == 'failed'
    assert runtime.read_json(pose.with_suffix('.log.json'))['status'] == 'timeout'
    def recover(command, log, timeout):
        Path(command[command.index('--out') + 1]).write_text('REMARK VINA RESULT: -6.0 0 0\n')
    monkeypatch.setattr(runtime, 'logged_run', recover)
    runtime.dock_job(tmp_path / 'lig', tmp_path / 'rec', pose, (0, 0, 0), 24, vina=tmp_path / 'vina.exe', resume=True)
    assert runtime.read_json(pose.with_suffix('.job.json'))['status'] == 'ok'


def test_atomic_checkpoint_retries_transient_windows_reader_lock(tmp_path, monkeypatch):
    original = Path.replace
    attempts = []
    def busy_once(path, target):
        attempts.append(str(target))
        if len(attempts) == 1:
            raise PermissionError('Windows reader holds destination')
        return original(path, target)
    monkeypatch.setattr(Path, 'replace', busy_once)
    monkeypatch.setattr(runtime.time, 'sleep', lambda seconds: None)
    target = tmp_path / 'checkpoint.json'
    runtime.write_json(target, {'complete': True})
    assert runtime.read_json(target) == {'complete': True} and len(attempts) == 2


def test_rmsd_does_not_align_away_wrong_binding_location():
    mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    moved = Chem.Mol(mol)
    for i, xyz in enumerate(moved.GetConformer().GetPositions()):
        moved.GetConformer().SetAtomPosition(i, xyz + np.array([5., 0., 0.]))
    assert fixed_frame_rmsd(moved, mol) == pytest.approx(5.)


def test_reference_fixtures_match_current_experimental_coordinates():
    for name in ('nav18', 'nav15', 'nav15_qdn_protonated'):
        assert reference_molecule(name, score) is not None


def test_validation_counts_failures_and_rejects_too_few_repeats(tmp_path):
    row = {'reference': 'nav18', 'status': 'ok', 'top1_passed': True,
           'top1_rmsd_A': 1., 'best_of_modes_rmsd_A': 1., 'affinity_kcal_mol': -7.}
    result = summarize([row] * 11 + [{'reference': 'nav18', 'status': 'failed'}] * 4, 15, True)[0]
    assert result['top1_pass_fraction'] == 11 / 15 and not result['gate_passed']
    assert not summarize([row] * 3, 3, False)[0]['gate_passed']
    report = tmp_path / 'report.json'
    runtime.write_json(report, {'stage': 'reference', 'gate_passed': False})
    with pytest.raises(ValueError, match='not passed'):
        workflow.check_gate(report, {}, 42)


def test_batch_preserves_metadata_and_failure_records(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'protocol', lambda *a: {})
    monkeypatch.setattr(workflow, 'check_gate', lambda *a: {})
    def mock_score(label, smiles, *a, **kw):
        if label == 'bad':
            raise ValueError('explicit failure')
        return score.Scored(label, smiles, -8, -7, 1, .5, 2, 1, .6)
    monkeypatch.setattr(score, 'score_one', mock_score)
    args = SimpleNamespace(validation_report=None, seed=42, preparation_seed=42, num_confs=2,
                           resume=False, vina=None, exhaustiveness=8, num_modes=9, cpu=1, timeout=10, workers=2, out=None)
    entries = [{'candidate_id': 'good', 'smiles': 'CCO', 'scope': 'strict'},
               {'candidate_id': 'bad', 'smiles': 'bad', 'scope': 'expanded'}]
    assert workflow.run_candidates(score, entries, {}, tmp_path, args) == 2
    results = runtime.read_json(tmp_path / 'results.json')
    assert len(results) == 2 and results[0]['scope'] == 'strict'
    assert results[1]['status'] == 'failed' and 'explicit failure' in results[1]['error']
    assert 'good' in (tmp_path / 'subtype_scores.tsv').read_text()
