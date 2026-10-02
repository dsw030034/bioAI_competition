"""Protect candidate identity, independent geometry QA, and cached contacts."""
import io
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nav18_pose_quality as qa
from scgd.geometry import pose_contacts


def molecule(smiles='C', coordinates=((0., 0., 0.),)):
    mol = Chem.MolFromSmiles(smiles)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for index, xyz in enumerate(coordinates):
        conf.SetAtomPosition(index, tuple(xyz))
    mol.AddConformer(conf)
    return mol


def pdb(path, positions):
    path.write_text(''.join(
        f'ATOM  {i:5d}  CA  ALA A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 10.00           C\n'
        for i, (x, y, z) in enumerate(positions, 1)) + 'END\n')
    return path


def sdf_string(molecules):
    stream = io.StringIO()
    writer = Chem.SDWriter(stream)
    for mol in molecules:
        writer.write(mol)
    writer.flush()
    return stream.getvalue()


def setup_audit(tmp_path, monkeypatch, molecules=None, scores=None):
    molecules = [molecule()] if molecules is None else molecules
    scores = [-7. + i for i in range(len(molecules))] if scores is None else scores
    pose = tmp_path/'poses.pdbqt'
    pose.write_text(''.join(f'REMARK VINA RESULT: {score} 0 0\n' for score in scores))
    monkeypatch.setattr(qa, 'PDBQTMolecule', lambda *a, **k: object())
    monkeypatch.setattr(qa.RDKitMolCreate, 'write_sd_string', lambda obj: (sdf_string(molecules), []))
    protein = pdb(tmp_path/'protein.pdb', [(5., 0., 0.), (50., 0., 0.)])
    lipid = tmp_path/'lipid.sdf'
    lipid.write_text(sdf_string([molecule(coordinates=((8., 0., 0.),))]))
    return pose, protein, lipid


def assess(files, smiles='C', **kwargs):
    pose, protein, lipid = files
    return qa.assess_candidate_pose(pose, smiles, protein, lipid,
                                    center=(0., 0., 0.), size=20., **kwargs)


def test_candidate_needs_no_reference_pose_and_rmsd_is_fixed_frame(tmp_path, monkeypatch):
    files = setup_audit(tmp_path, monkeypatch,
                        [molecule(), molecule(coordinates=((1., 0., 0.),))])
    result = assess(files)
    assert result['quality_status'] == 'pass'
    assert result['all_pose_rmsd_to_top1_A'] == pytest.approx([0., 1.])
    assert result['all_pose_affinity_kcal_mol'] == [-7., -6.]
    assert result['chemical_identity_verified']
    assert result['stereochemistry_3d_verified']
    assert len(result['modes']) == 2
    assert Path(result['pose_sdf']).exists()
    json.dumps(result, allow_nan=False)


def test_top1_is_never_replaced_with_better_looking_alternative(tmp_path, monkeypatch):
    files = setup_audit(tmp_path, monkeypatch,
                        [molecule(coordinates=((5., 0., 0.),)), molecule()])
    result = assess(files)
    assert result['quality_status'] == 'review'
    assert result['affinity_kcal_mol'] == -7.
    assert result['heavy_pairs_below_2A'] == 1
    assert result['modes'][1]['quality_status'] == 'pass'


def test_lipid_clash_is_measured_separately(tmp_path, monkeypatch):
    files = setup_audit(tmp_path, monkeypatch, [molecule(coordinates=((8., 0., 0.),))])
    result = assess(files)
    assert result['heavy_pairs_below_2A'] == 0
    assert result['lipid_heavy_pairs_below_2A'] == 1
    assert result['quality_flags'] == ['lipid_heavy_clash_below_2A']


@pytest.mark.parametrize('x,flag,severe', [(9.5, 'box_boundary_within_1A', False),
                                         (10.5, 'outside_requested_box', True)])
def test_requested_box_boundary_is_explicit(tmp_path, monkeypatch, x, flag, severe):
    files = setup_audit(tmp_path, monkeypatch, [molecule(coordinates=((0., x, 0.),))])
    result = assess(files)
    assert result['min_requested_box_edge_margin_A'] == pytest.approx(10. - x)
    assert flag in result['quality_flags']
    assert (flag in result['severe_quality_flags']) is severe


def test_graph_and_charge_mismatch_fail_instead_of_scoring_wrong_molecule(tmp_path, monkeypatch):
    files = setup_audit(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='graph mismatch'):
        assess(files, smiles='N')
    with pytest.raises(ValueError, match='graph mismatch'):
        assess(files, smiles='[CH3-]')


def test_unspecified_stereo_is_never_chosen_by_docking(tmp_path, monkeypatch):
    files = setup_audit(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='requires stereochemistry review'):
        assess(files, smiles='CC(O)F')


def test_mirrored_stereoisomer_is_rejected(tmp_path, monkeypatch):
    mol = Chem.AddHs(Chem.MolFromSmiles('C[C@H](O)F'))
    assert AllChem.EmbedMolecule(mol, randomSeed=81) == 0
    mirrored = Chem.Mol(mol)
    conf = mirrored.GetConformer()
    for i, xyz in enumerate(conf.GetPositions()):
        conf.SetAtomPosition(i, (-xyz[0], xyz[1], xyz[2]))
    files = setup_audit(tmp_path, monkeypatch, [mirrored])
    with pytest.raises(ValueError, match='stereochemistry mismatch'):
        assess(files, smiles='C[C@H](O)F')


def test_implausible_bond_length_is_reviewed(tmp_path, monkeypatch):
    files = setup_audit(tmp_path, monkeypatch,
                        [molecule('CC', ((0., 0., 0.), (0.1, 0., 0.)))])
    result = assess(files, smiles='CC')
    assert result['quality_status'] == 'review'
    assert 'intramolecular_geometry_issue' in result['quality_flags']
    assert any(issue.startswith('bond_length:') for issue in result['geometry_issues'])


@pytest.mark.parametrize('score', ['nan', 'inf', '-inf', 'invalid'])
def test_nonfinite_or_malformed_scores_cannot_escape_to_json(tmp_path, monkeypatch, score):
    files = setup_audit(tmp_path, monkeypatch, scores=[score])
    with pytest.raises(ValueError, match='affinity'):
        assess(files)


def test_pose_count_must_match_scores(tmp_path, monkeypatch):
    files = setup_audit(tmp_path, monkeypatch, scores=[-8., -7.])
    with pytest.raises(ValueError, match='Pose/score count mismatch'):
        assess(files)


def test_cached_contacts_match_original_and_refresh_when_source_changes(tmp_path, monkeypatch):
    files = setup_audit(tmp_path, monkeypatch)
    qa._protein_data.cache_clear()
    original = qa.pdb_atoms
    calls = []
    monkeypatch.setattr(qa, 'pdb_atoms', lambda path: (calls.append(path), original(path))[1])
    first = assess(files)
    second = assess(files)
    direct = pose_contacts(molecule(), files[1])
    assert all(first[key] == value for key, value in direct.items())
    assert first['min_protein_heavy_distance_A'] == second['min_protein_heavy_distance_A']
    assert len(calls) == 1
    before = files[1].stat().st_mtime_ns
    pdb(files[1], [(1., 0., 0.), (50., 0., 0.)])
    os.utime(files[1], ns=(before + 10_000_000, before + 10_000_000))
    changed = assess(files)
    assert len(calls) == 2
    assert changed['heavy_pairs_below_2A'] == 1


def test_bounding_filter_preserves_exact_distant_minimum(tmp_path):
    protein_path = pdb(tmp_path/'distant.pdb', [(3.9, 3.9, 3.9), (4.1, 0., 0.)])
    result = qa._protein_contacts(np.zeros((1, 3)), *qa._protein_data(*qa._cache_key(protein_path)))
    assert result == pose_contacts(molecule(), protein_path)
    assert result['min_protein_heavy_distance_A'] == pytest.approx(4.1)


def test_real_historical_nav18_pose_exports_and_matches_existing_contacts(tmp_path):
    base = Path(__file__).resolve().parents[1]
    folder = base/'outputs/nav18_lipid_control_20261002/zwitterion'
    source = folder/'jobs/nav18_c4_s20261004/poses.pdbqt'
    protein = base/'outputs/receptors_20261002/nav18/prepared.pdb'
    reference_path = base/'data/reference_validation/nav18.sdf'
    if not all(p.exists() for p in (source, protein, reference_path)):
        pytest.skip('Historical reference fixtures unavailable')
    pose = tmp_path/'historical_copy.pdbqt'
    shutil.copyfile(source, pose)
    expected = Chem.SDMolSupplier(str(reference_path), removeHs=True)[0]
    result = qa.assess_candidate_pose(pose, Chem.MolToSmiles(expected), protein, folder/'truncated_lpe.sdf')
    exported = Chem.SDMolSupplier(result['pose_sdf'], removeHs=True)[0]
    assert result['num_modes'] == 9
    assert result['quality_status'] == 'pass'
    assert all(result[key] == value for key, value in pose_contacts(exported, protein).items())
    assert result['lipid_heavy_pairs_below_2A'] == 0
    json.dumps(result, allow_nan=False)
