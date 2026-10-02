"""Tests for fixed experimental coordinates and diagnostic output interpretation."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pytest
from rdkit import Chem

from diagnose_nav15_native import BASE, export_fixed_pose, parse_affinity


def test_native_pose_typing_preserves_coordinates_graph_and_charge(tmp_path):
    for name, charge in [('nav15', 0), ('nav15_qdn_protonated', 1)]:
        mol = Chem.SDMolSupplier(str(BASE / 'data/reference_validation' / (name + '.sdf')))[0]
        coords = mol.GetConformer().GetPositions().copy()
        audit = export_fixed_pose(mol, tmp_path / (name + '.pdbqt'))
        assert audit['formal_charge'] == charge
        assert audit['heavy_atoms'] == 24
        assert audit['hydrogen_addition_max_heavy_displacement_A'] == 0
        assert audit['pdbqt_roundtrip_fixed_frame_rmsd_A'] < 0.002
        assert (mol.GetConformer().GetPositions() == coords).all()
        assert audit['smiles'] == Chem.MolToSmiles(mol)


def test_parse_actual_vina_diagnostic_output_without_confusing_components():
    text = ('Estimated Free Energy of Binding   : -6.679 (kcal/mol) [=(1)+(2)+(3)-(4)]\n'
            '(1) Final Intermolecular Energy    : -8.436 (kcal/mol)\n')
    assert parse_affinity(text) == pytest.approx(-6.679)
    assert parse_affinity('Affinity: -8.490 (kcal/mol)\n') == pytest.approx(-8.49)
    with pytest.raises(ValueError):
        parse_affinity('No computed affinity in a failed process')
