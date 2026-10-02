"""Chemical identity and coordinate invariants for +1 sensitivity controls."""
from pathlib import Path
import sys

import numpy as np
import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_nav15_reference_states import BASE, protonate_one_nitrogen, is_amide_nitrogen
from scgd.runtime import read_json


@pytest.mark.parametrize('parent,selected,expected_h', [
    ('rat_6uz0', 'N19', 2), ('rat_8f6p', 'N13', 1), ('rat_8f6p', 'N16', 1)])
def test_single_proton_changes_charge_without_moving_or_inverting_heavy_atoms(parent, selected, expected_h):
    folder = BASE / 'outputs/nav15_multireference_20261002' / parent
    original = Chem.SDMolSupplier(str(folder / 'reference.sdf'), removeHs=True)[0]
    names = read_json(folder / 'ligand_audit.json')['atom_names']
    index = names.index(selected)
    state, audit = protonate_one_nitrogen(original, index)
    assert Chem.GetFormalCharge(state) == 1
    assert state.GetAtomWithIdx(index).GetTotalNumHs() == expected_h
    assert np.array_equal(state.GetConformer().GetPositions(), original.GetConformer().GetPositions())
    assert state.GetNumAtoms() == original.GetNumAtoms()
    assert state.GetNumBonds() == original.GetNumBonds()
    assert all(x['source_CIP'] == x['state_CIP'] == 'R' for x in audit['carbon_stereo'])
    assert all(x['CIP'] is None and x['chiral_tag'] == 'CHI_UNSPECIFIED' for x in audit['nitrogen_stereo'])
    expected_changes = [(a.GetIdx(), a.GetFormalCharge() - b.GetFormalCharge())
                        for a, b in zip(state.GetAtoms(), original.GetAtoms())
                        if a.GetFormalCharge() != b.GetFormalCharge()]
    assert expected_changes == [(index, 1)]


@pytest.mark.parametrize('parent,amide_name', [('rat_6uz0', 'N12'), ('rat_8f6p', 'N09')])
def test_amide_nitrogens_are_rejected(parent, amide_name):
    folder = BASE / 'outputs/nav15_multireference_20261002' / parent
    original = Chem.SDMolSupplier(str(folder / 'reference.sdf'), removeHs=True)[0]
    names = read_json(folder / 'ligand_audit.json')['atom_names']
    index = names.index(amide_name)
    assert is_amide_nitrogen(original.GetAtomWithIdx(index))
    with pytest.raises(ValueError, match='non-amide amine'):
        protonate_one_nitrogen(original, index)


def test_ranolazine_microstates_are_distinct_graphs_not_new_enantiomers():
    folder = BASE / 'outputs/nav15_multireference_20261002/rat_8f6p'
    original = Chem.SDMolSupplier(str(folder / 'reference.sdf'), removeHs=True)[0]
    names = read_json(folder / 'ligand_audit.json')['atom_names']
    states = [protonate_one_nitrogen(original, names.index(name))[0] for name in ('N13', 'N16')]
    assert Chem.MolToSmiles(states[0]) != Chem.MolToSmiles(states[1])
    for state in states:
        assert [a.GetProp('_CIPCode') for a in state.GetAtoms() if a.HasProp('_CIPCode')] == ['R']
        assert Chem.GetFormalCharge(state) == 1
