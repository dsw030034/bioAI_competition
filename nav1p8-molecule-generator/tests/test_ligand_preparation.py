import json

import numpy as np
import pytest
from rdkit import Chem

from navgen.ligand_preparation import state_record, prepare_candidate, geometry_issues, run


def candidate(smiles, name="test"):
    return {"candidate_id": name, "smiles": smiles, "scope": "test", "methods": [], "event_ids": []}


def test_charge_and_specified_stereo_preserved_and_undefined_stereo_deferred():
    _, record = state_record(candidate("C[NH2+]C"))
    assert record["formal_charge"] == 1 and record["state_smiles"] == "C[NH2+]C"
    state, records, _, kept = prepare_candidate(candidate("CC(O)F"), num_confs=2)
    assert state["status"] == "requires_stereochemistry_review"
    assert not records and not kept
    _, specified = state_record(candidate("C[C@H](O)F"))
    assert specified["status"] == "pending" and not specified["unspecified_stereo"]
    _, salt = state_record(candidate("C[NH3+].[Cl-]"))
    assert salt["status"] == "requires_component_review"


def test_real_3d_geometry_export_lineage_and_failure_isolation(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"candidates": [candidate("CCCCCC", "hexane"), candidate("invalid", "bad"),
                                             candidate("C[C@H](O)F", "chiral")]}))
    result = run(path, tmp_path / "out", num_confs=12, max_keep=3)
    assert result["states"]["prepared_input_state"] == 2
    assert result["states"]["invalid_smiles"] == 1
    mols = list(Chem.SDMolSupplier(str(tmp_path / "out/prepared_input_states_3d.sdf"), removeHs=False))
    records = json.loads((tmp_path / "out/conformers.json").read_text())
    retained = {r["conformer_id"]: r for r in records if r["status"] == "retained"}
    assert len(mols) == result["retained_conformers"] == len(retained)
    assert all(m is not None and m.GetConformer().Is3D() for m in mols)
    for m in mols:
        assert m.GetProp("conformer_id") in retained
        assert retained[m.GetProp("conformer_id")]["minimization_return_codes"][-1] == 0
        assert not geometry_issues(m, 0, m.GetProp("state_smiles"))
    hexane = [m for m in mols if m.GetProp("candidate_id") == "hexane"]
    assert len(hexane) > 1
    assert not np.allclose(hexane[0].GetConformer().GetPositions(), hexane[1].GetConformer().GetPositions())
    with pytest.raises(FileExistsError):
        run(path, tmp_path / "out")


def test_nonconvergence_never_reported_as_prepared(monkeypatch):
    class Unconverged:
        def Minimize(self, **kwargs):
            return 1
        def CalcEnergy(self):
            return -1.0
    monkeypatch.setattr("navgen.ligand_preparation.AllChem.MMFFGetMoleculeForceField", lambda *a, **kw: Unconverged())
    state, records, _, kept = prepare_candidate(candidate("CCO"), num_confs=2)
    assert state["status"] == "no_usable_conformer"
    assert records and all(r["status"] == "optimization_not_converged" for r in records)
    assert not kept
