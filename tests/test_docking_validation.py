import importlib.util
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from navgen.docking_validation import fixed_frame_rmsd, select_chain, pdb_atoms


def molecule():
    m = Chem.AddHs(Chem.MolFromSmiles("CCOc1ccccc1"))
    AllChem.EmbedMolecule(m, randomSeed=42)
    return Chem.RemoveHs(m)


def test_rmsd_does_not_align_away_translation():
    ref = molecule()
    pose = Chem.Mol(ref)
    xyz = pose.GetConformer().GetPositions()
    for i, p in enumerate(xyz):
        pose.GetConformer().SetAtomPosition(i, p + np.array([3., 0., 0.]))
    assert fixed_frame_rmsd(pose, ref) == pytest.approx(3.0)


def test_rmsd_handles_atom_reordering_and_rejects_wrong_graph():
    ref = molecule()
    reordered = Chem.RenumberAtoms(ref, list(reversed(range(ref.GetNumAtoms()))))
    assert fixed_frame_rmsd(reordered, ref) == pytest.approx(0., abs=1e-6)
    with pytest.raises(ValueError):
        fixed_frame_rmsd(ref, Chem.MolFromSmiles("CC"))


def test_chain_selection_preserves_incomplete_residue_and_ids(tmp_path):
    source = tmp_path / "in.pdb"
    source.write_text(
        "ATOM      1  N   SER B  42A     10.000  10.000  10.000  1.00 20.00           N  \n"
        "ATOM      2  CA  SER B  42A     11.000  10.000  10.000  1.00 20.00           C  \n"
        "ATOM      3  N   GLY B  50      20.000  10.000  10.000  1.00 20.00           N  \nEND\n")
    out = tmp_path / "out.pdb"
    assert select_chain(source, out, "B") == (2, 3)
    assert ("B", "42A", "SER", "CA") in pdb_atoms(out)
    assert out.read_text().count("TER") == 2


def test_validation_gate_counts_failures_and_top1_only():
    pytest.importorskip("meeko")
    path = Path(__file__).resolve().parents[1] / "scripts/run_validated_docking.py"
    spec = importlib.util.spec_from_file_location("validated_runner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    row = dict(receptor="nav18", candidate_id="reference", is_reference=True,
               status="ok", affinity_kcal_mol=-8., heavy_pairs_below_2A=0,
               top1_rmsd_A=4., best_of_modes_rmsd_A=0.5, contact_recall=0.8)
    assert not mod.summarize([row])[0]["gate_passed"]
    row["top1_rmsd_A"] = 1.
    assert mod.summarize([row])[0]["gate_passed"]
    bad = dict(row, status="preparation_failed")
    assert not mod.summarize([row, bad])[0]["gate_passed"]
