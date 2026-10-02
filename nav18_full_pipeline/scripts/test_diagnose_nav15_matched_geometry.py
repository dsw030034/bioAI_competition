"""Guard the chemical identity and fixed-geometry controls used in this diagnostic."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pytest
from rdkit import Chem

from diagnose_nav15_matched_geometry import (
    BASE, BASELINE, add_hydrogens, export_typed, place_rigid,
    reorder_to_reference,
)


@pytest.fixture(params=['nav15', 'nav15_qdn_protonated'])
def ligand_pair(request):
    name = request.param
    native = Chem.SDMolSupplier(str(BASE / 'data/reference_validation' / f'{name}.sdf'))[0]
    other = Chem.SDMolSupplier(str(BASELINE / 'jobs' / f'{name}_c0_s20261002' / 'poses.sdf'))[0]
    return native, reorder_to_reference(other, native)


def test_atom_correspondence_survives_renumbering_and_rejects_wrong_stereo(ligand_pair):
    native, _ = ligand_pair
    shuffled = Chem.RenumberAtoms(native, list(reversed(range(native.GetNumAtoms()))))
    restored = reorder_to_reference(shuffled, native)
    assert np.allclose(restored.GetConformer().GetPositions(), native.GetConformer().GetPositions())
    epimer = Chem.Mol(native)
    atom = next(a for a in epimer.GetAtoms()
                if a.GetAtomicNum() == 6 and a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)
    atom.InvertChirality()
    with pytest.raises(ValueError, match='identity or stereo'):
        reorder_to_reference(epimer, native)


def test_hydrogen_optimization_preserves_experimental_heavy_atoms(ligand_pair):
    native, _ = ligand_pair
    before = native.GetConformer().GetPositions().copy()
    optimized, audit = add_hydrogens(native, 'fixed_heavy_mmff')
    assert audit['minimization_codes'][-1] == 0
    assert optimized.GetNumAtoms() > native.GetNumAtoms()
    assert np.array_equal(before, optimized.GetConformer().GetPositions()[:len(before)])
    assert Chem.MolToSmiles(Chem.RemoveHs(optimized)) == Chem.MolToSmiles(native)


def test_site_transfer_preserves_internal_geometry_and_vina_types(ligand_pair, tmp_path):
    native, other = ligand_pair
    hydrogenated, _ = add_hydrogens(native, 'fixed_heavy_mmff')
    placed, _ = place_rigid(hydrogenated, other)
    a = hydrogenated.GetConformer().GetPositions()
    b = placed.GetConformer().GetPositions()
    assert np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)) > 5
    assert np.allclose(np.linalg.norm(a[:, None] - a[None, :], axis=2),
                       np.linalg.norm(b[:, None] - b[None, :], axis=2), atol=1e-8)
    before = export_typed(hydrogenated, tmp_path / 'native.pdbqt')
    after = export_typed(placed, tmp_path / 'wrong.pdbqt')
    assert before['atom_types'] == after['atom_types']
    assert before['torsdof'] == after['torsdof'] == 'TORSDOF 5'
    assert before['formal_charge'] == after['formal_charge']
    assert max(before['pdbqt_roundtrip_rmsd_A'], after['pdbqt_roundtrip_rmsd_A']) < .002
