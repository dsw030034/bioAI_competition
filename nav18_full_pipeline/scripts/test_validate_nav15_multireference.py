"""Protect exploratory reference tests from pose leakage and gate promotion."""
from argparse import Namespace
from pathlib import Path
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_nav15_multireference as diagnostic


def row(conformer, seed, score=-7., passed=False, any_pass=False, rmsd=9.):
    return {'conformer_id': conformer, 'vina_seed': seed, 'status': 'ok',
            'affinity_kcal_mol': score, 'top1_passed': passed,
            'any_mode_passed': any_pass, 'top1_rmsd_A': rmsd,
            'best_of_modes_rmsd_A': 1. if any_pass else rmsd}


def test_any_mode_hits_do_not_release_top1_diagnostic():
    rows = [row(f'c{i}', seed, any_pass=True) for i in range(5) for seed in diagnostic.VINA_SEEDS]
    result = diagnostic.summarize(rows, 5)
    assert result['top1_pass_count'] == 0
    assert result['any_mode_pass_count'] == 15
    assert result['reference_diagnostic_passed'] is False
    assert result['production_gate_released'] is False


def test_ensemble_winners_follow_original_score_not_native_rmsd():
    rows = []
    for seed in diagnostic.VINA_SEEDS:
        rows.extend([row('wrong_but_lower_score', seed, -9., rmsd=10.),
                     row('near_native', seed, -6., passed=True, any_pass=True, rmsd=1.)])
    result = diagnostic.summarize(rows, 2)
    assert len(result['original_score_ensemble_winners']) == 3
    assert all(x['conformer_id'] == 'wrong_but_lower_score'
               for x in result['original_score_ensemble_winners'])


def test_even_successful_exploratory_reference_cannot_release_production_gate():
    rows = [row(f'c{i}', seed, passed=True, any_pass=True, rmsd=1.)
            for i in range(5) for seed in diagnostic.VINA_SEEDS]
    result = diagnostic.summarize(rows, 5)
    assert result['reference_diagnostic_passed'] is True
    assert result['production_gate_released'] is False


def test_failed_jobs_count_against_denominator_and_disable_pass():
    rows = [row(f'c{i}', seed, passed=True, any_pass=True, rmsd=1.)
            for i in range(5) for seed in diagnostic.VINA_SEEDS]
    rows[-1] = {'conformer_id': 'c4', 'vina_seed': diagnostic.VINA_SEEDS[-1], 'status': 'failed'}
    result = diagnostic.summarize(rows, 5)
    assert result['top1_pass_fraction'] == pytest.approx(14 / 15)
    assert result['reference_diagnostic_passed'] is False
    assert len(result['original_score_ensemble_winners']) == 2


@pytest.mark.parametrize('change', ['modified', 'missing'])
def test_frozen_sampling_rejects_tampered_or_missing_files(tmp_path, change):
    path = tmp_path / 'selected.pdbqt'
    path.write_text('original geometry')
    hashes = diagnostic.freeze_artifacts(tmp_path)
    diagnostic.verify_artifacts(tmp_path, hashes)
    if change == 'modified':
        path.write_text('different geometry')
    else:
        path.unlink()
    with pytest.raises(ValueError, match='Frozen input/output changed'):
        diagnostic.verify_artifacts(tmp_path, hashes)


@pytest.mark.parametrize('approvals', [
    {'eligible_for_reference_redocking': False, 'chemistry_usable': False},
    {'eligible_for_reference_redocking': False, 'chemistry_usable': True},
    {'eligible_for_reference_redocking': True, 'chemistry_usable': False},
])
def test_chemical_conflict_is_rejected_before_sampling_or_docking(tmp_path, monkeypatch, approvals):
    diagnostic.write_json(tmp_path / 'prepare_manifest.json', {
        **approvals})
    def forbidden(*args, **kwargs):
        raise AssertionError('unapproved chemistry reached calculations')
    monkeypatch.setattr(diagnostic, 'sample', forbidden)
    monkeypatch.setattr(diagnostic, 'dock_job', forbidden)
    with pytest.raises(ValueError, match='explicitly approve chemical identity'):
        diagnostic.run(Namespace(reference_dir=tmp_path, output=tmp_path / 'run',
                                 size=24, workers=1, resume=False))


def test_incomplete_sampling_is_preserved_with_actionable_error(tmp_path):
    partial = tmp_path / 'sampling'
    partial.mkdir()
    witness = partial / 'progress.json'
    witness.write_text('incomplete pool evidence')
    with pytest.raises(ValueError, match='use a new output directory'):
        diagnostic.sample(tmp_path / 'irrelevant_reference.sdf', partial)
    assert witness.read_text() == 'incomplete pool evidence'


def test_sampling_ignores_experimental_coordinate_frame(tmp_path, monkeypatch):
    """Changing only deposited xyz must not change independently sampled inputs."""
    mol = Chem.AddHs(Chem.MolFromSmiles('CC[C@H](O)CO'))
    assert AllChem.EmbedMolecule(mol, randomSeed=923) == 0
    references = []
    for name, shift in [('first', np.array([0., 0., 0.])), ('translated', np.array([100., -50., 200.]))]:
        moved = Chem.Mol(mol)
        conformer = moved.GetConformer()
        for i, xyz in enumerate(conformer.GetPositions()):
            conformer.SetAtomPosition(i, xyz + shift)
        path = tmp_path / f'{name}.sdf'
        with Chem.SDWriter(str(path)) as writer:
            writer.write(moved)
        references.append(path)
    actual_embed = diagnostic.embed_seed_pool
    def small_embedding(graph, seed, count):
        assert graph.GetNumConformers() == 0
        return actual_embed(graph, seed, 4)
    monkeypatch.setattr(diagnostic, 'EMBED_SEEDS', (4567,))
    monkeypatch.setattr(diagnostic, 'embed_seed_pool', small_embedding)
    reports = [diagnostic.sample(path, tmp_path / f'sample{i}') for i, path in enumerate(references)]
    assert reports[0]['smiles'] == reports[1]['smiles']
    assert reports[0]['reference_coordinates_used'] is False
    assert reports[0]['selection_uses_reference_rmsd'] is False
    entries = [report['selected_entries'] for report in reports]
    assert len(entries[0]) == len(entries[1]) > 0
    for first, translated in zip(*entries):
        # Byte equality includes Meeko typing, all coordinates, and torsions.
        assert Path(first['ligand']).read_bytes() == Path(translated['ligand']).read_bytes()
