"""Guard independence, chemical identity and geometry diversity of QDN sampling."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from nav15_conformer_sampling import (
    BASE, deduplicate, embed_seed_pool, export_typed, graph_only,
)


@pytest.fixture(params=['nav15', 'nav15_qdn_protonated'])
def native_graph(request):
    return Chem.SDMolSupplier(str(BASE / 'data/reference_validation' / f'{request.param}.sdf'))[0]


def test_embedding_discards_input_coordinates_and_keeps_ordered_identity(native_graph):
    changed = Chem.Mol(native_graph)
    conf = changed.GetConformer()
    for index in range(changed.GetNumAtoms()):
        conf.SetAtomPosition(index, (99999 + index, -99999, 98765))
    graph = graph_only(changed)
    assert graph.GetNumConformers() == 0
    assert Chem.MolToSmiles(graph) == Chem.MolToSmiles(native_graph)
    assert [(a.GetAtomicNum(), a.GetFormalCharge(), a.GetChiralTag()) for a in graph.GetAtoms()] == [
        (a.GetAtomicNum(), a.GetFormalCharge(), a.GetChiralTag()) for a in native_graph.GetAtoms()]
    first, ids1 = embed_seed_pool(native_graph, 345123, 3)
    second, ids2 = embed_seed_pool(changed, 345123, 3)
    assert len(ids1) == len(ids2) == 3
    for a, b in zip(ids1, ids2):
        assert np.array_equal(first.GetConformer(a).GetPositions(), second.GetConformer(b).GetPositions())


def test_deduplication_rejects_rigid_copy_and_higher_energy_duplicate():
    mol = Chem.AddHs(Chem.MolFromSmiles('CCCO'))
    assert AllChem.EmbedMolecule(mol, randomSeed=43123) == 0
    copy = Chem.Conformer(mol.GetConformer())
    for index in range(mol.GetNumAtoms()):
        p = copy.GetAtomPosition(index)
        copy.SetAtomPosition(index, (-p.y + 7, p.x + 9, p.z - 3))
    duplicate_id = mol.AddConformer(copy, assignId=True)
    records = [{'conformer_id': 'first', 'rdkit_conf_id': 0, 'energy_kcal_mol': 0., 'status': 'geometry_passed'},
               {'conformer_id': 'rigid_copy', 'rdkit_conf_id': duplicate_id,
                'energy_kcal_mol': 1., 'status': 'geometry_passed'}]
    before = mol.GetConformer(duplicate_id).GetPositions().copy()
    unique, selected = deduplicate(mol, records)
    assert len(unique) == len(selected) == 1
    assert selected[0]['conformer_id'] == 'first'
    assert records[1]['status'] == 'duplicate_geometry'
    assert records[1]['duplicate_of'] == 'first'
    assert records[1]['nearest_prior_representative_rmsd_A'] < 1e-6
    assert np.array_equal(before, mol.GetConformer(duplicate_id).GetPositions())


def test_sampled_qdn_identity_and_typing_roundtrip(native_graph, tmp_path):
    mol, ids = embed_seed_pool(native_graph, 456788, 1)
    assert ids
    expected = Chem.MolToSmiles(native_graph)
    assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == expected
    audit = export_typed(mol, tmp_path / 'sample.pdbqt')
    restored = Chem.SDMolSupplier(str(tmp_path / 'sample.roundtrip.sdf'))[0]
    assert Chem.MolToSmiles(restored) == expected
    assert audit['formal_charge'] == Chem.GetFormalCharge(native_graph)
    assert audit['pdbqt_roundtrip_rmsd_A'] < .002
    assert audit['torsdof'] == 'TORSDOF 5'


def test_unique_count_includes_window_exclusions_and_output_cap():
    mol = Chem.AddHs(Chem.MolFromSmiles('CCCCCC'))
    assert AllChem.EmbedMolecule(mol, randomSeed=75147) == 0
    rdMolTransforms.SetDihedralDeg(mol.GetConformer(0), 0, 1, 2, 3, 0.)
    for angle in (60., 180.):
        idx = mol.AddConformer(Chem.Conformer(mol.GetConformer(0)), assignId=True)
        rdMolTransforms.SetDihedralDeg(mol.GetConformer(idx), 0, 1, 2, 3, angle)
    records = [{'conformer_id': f'c{i}', 'rdkit_conf_id': i, 'energy_kcal_mol': energy,
                'status': 'geometry_passed'} for i, energy in enumerate((0., 1., 15.))]
    unique, selected = deduplicate(mol, records, threshold=.01, energy_window=10., max_selected=1)
    assert len(unique) == 3
    assert len(selected) == 1
    assert [r['status'] for r in records] == ['selected', 'unique_over_selection_cap', 'unique_outside_energy_window']
