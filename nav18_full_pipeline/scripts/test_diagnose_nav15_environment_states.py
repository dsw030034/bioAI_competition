"""Protect matched diagnostics, interpretation, and immutable inputs."""
from pathlib import Path
import sys
import numpy as np
import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_nav15_environment_states as diag


def mol_at(x):
    mol = Chem.MolFromSmiles('C')
    conf = Chem.Conformer(1)
    conf.SetAtomPosition(0, (x, 0., 0.))
    mol.AddConformer(conf)
    return mol


def rows(protein=True, environment=False, any_hit=True):
    return [{'status': 'ok', 'conformer_id': f'c{i}', 'vina_seed': seed,
             'top1_passed': protein, 'top1_environment_passed': environment,
             'any_mode_passed': any_hit, 'top1_rmsd_A': 1., 'best_of_modes_rmsd_A': 1.,
             'affinity_kcal_mol': -8. if i == 0 else -7.,
             'modes': [{'actual_grid_face_margin_A': 2.}]}
            for i in range(5) for seed in diag.SEEDS]


def test_lipid_clash_is_measured_separately_from_protein_pass():
    d = diag.lipid_distances(mol_at(0), mol_at(1.9))
    assert d['lipid_heavy_pairs_below_2A'] == 1
    assert d['min_lipid_heavy_distance_A'] == pytest.approx(1.9)
    result = diag.summarize(rows(), 5, True)
    assert result['top1_protein_pass_count'] == 15
    assert result['top1_pass_count'] == 0
    assert result['reference_diagnostic_passed'] is False


def test_no_lipid_not_measured_is_null_not_zero_distance():
    result = diag.lipid_distances(mol_at(0), None)
    assert result['min_lipid_heavy_distance_A'] is None
    assert result['lipid_heavy_pairs_below_2A'] is None


def test_success_never_releases_production_or_validates_selectivity():
    result = diag.summarize(rows(environment=True), 5, True)
    assert result['reference_diagnostic_passed'] is True
    assert result['production_gate_released'] is False
    assert result['selectivity_validated'] is False


def test_failed_job_counts_in_denominator_and_cannot_pass():
    data = rows(environment=True)
    data[-1] = {'status': 'failed', 'conformer_id': 'c4', 'vina_seed': diag.SEEDS[-1]}
    result = diag.summarize(data, 5, False)
    assert result['top1_pass_fraction'] == pytest.approx(14/15)
    assert result['reference_diagnostic_passed'] is False


def test_ensemble_selection_uses_original_score_not_pose_pass():
    data = rows(environment=True)
    for row in data:
        if row['conformer_id'] == 'c0':
            row['top1_environment_passed'] = False
            row['top1_rmsd_A'] = 9.
    result = diag.summarize(data, 5, False)
    assert all(x['conformer_id'] == 'c0' and not x['top1_environment_passed']
               for x in result['original_score_ensemble_winners'])


@pytest.mark.parametrize('missing', [True, False])
def test_frozen_input_missing_or_changed_refuses_reuse(tmp_path, missing):
    path = tmp_path/'input.sdf'
    path.write_text('initial')
    hashes = diag.frozen_paths([path])
    diag.verify_frozen(hashes)
    if missing:
        path.unlink()
    else:
        path.write_text('changed')
    with pytest.raises(ValueError, match='Frozen input changed'):
        diag.verify_frozen(hashes)


def test_unchanged_old_neutral_inputs_are_accepted_without_resampling():
    prep = diag.BASE/'outputs/nav15_multireference_20261002/rat_8f6p'
    config = {'condition': 'test', 'preparation_manifest': str(prep/'prepare_manifest.json'),
              'sampling_report': str(prep/'redocking/sampling/report.json')}
    if not Path(config['sampling_report']).exists():
        pytest.skip('Historical fixture unavailable')
    before = diag.sha256(config['sampling_report'])
    ref, protein, receptor, entries, lipid, files = diag.read_inputs(config)
    assert len(entries) == 5 and lipid is None
    assert ref.GetNumHeavyAtoms() == 31
    assert diag.sha256(config['sampling_report']) == before
