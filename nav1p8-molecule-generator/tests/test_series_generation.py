import csv
import json
from pathlib import Path

import pytest
from rdkit import Chem

from navgen.series_generation import Series, run, site_proposals, mol2mol_proposals

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/a803467_generation.json"


@pytest.fixture
def series():
    return Series(json.loads(CONFIG.read_text()))


def test_sites_reconstruct_seed_and_preserve_secondary_amide(series):
    assert series.assemble(series.original) == series.seed
    products = list(site_proposals(series, 1)) + list(site_proposals(series, 2))
    assert products
    assert all(series.classify(p["smiles"])[0] == "strict" for p in products)
    methylated = series.seed.replace("NC(=O)", "N(C)C(=O)")
    assert series.classify(methylated)[0] == "expanded_round2"


def test_wrong_sites_and_changed_rings_not_in_strict_library(series):
    # Move para chlorine to meta on the distal ring; add a new substituent on furan.
    wrong_position = series.seed.replace("c3ccc(Cl)cc3", "c3cc(Cl)ccc3")
    changed_furan = series.seed.replace("c2ccc(", "c2c(F)cc(")
    assert wrong_position != series.seed and changed_furan != series.seed
    assert series.classify(wrong_position)[0] == "expanded_round2"
    assert series.classify(changed_furan)[0] == "expanded_round2"
    assert series.classify(series.seed.replace(")o2", ")s2"))[0] == "outside_series"


def test_bridge_between_named_sites_is_not_a_substituent(series):
    mol = Chem.RWMol(series.core)
    a = mol.AddAtom(Chem.Atom(6))
    b = mol.AddAtom(Chem.Atom(6))
    mol.AddBond(series.anchors[2], a, Chem.BondType.SINGLE)
    mol.AddBond(series.anchors[3], b, Chem.BondType.SINGLE)
    mol.AddBond(a, b, Chem.BondType.SINGLE)
    product = mol.GetMol()
    Chem.SanitizeMol(product)
    assert not series.strict_match(product)


def test_symmetric_sites_are_deduplicated_without_losing_lineage(series, tmp_path):
    a, b = dict(series.original), dict(series.original)
    a[2] = "O"
    b[3] = "O"
    assert series.assemble(a) == series.assemble(b)
    summary = run(CONFIG, tmp_path / "run")
    assert summary["strict_unique"] > 0
    rows = list(csv.DictReader((tmp_path / "run/candidates.csv").open(encoding="utf-8-sig")))
    assert len(rows) == len({r["smiles"] for r in rows})
    assert any(len(json.loads(r["methods"])) > 1 for r in rows)
    events = [json.loads(l) for l in (tmp_path / "run/lineage.jsonl").read_text().splitlines()]
    assert all(e["parent_smiles"] == series.seed for e in events)
    assert all(series.strict_match(Chem.MolFromSmiles(r["smiles"])) for r in rows)
    assert not any(k in rows[0] for k in ("reward", "docking_score"))
    assert all(not m.GetConformer().Is3D() for m in Chem.SDMolSupplier(str(tmp_path / "run/candidates_2d.sdf")))
    with pytest.raises(FileExistsError):
        run(CONFIG, tmp_path / "run")


def test_mol2mol_rejects_old_parent_and_preserves_invalid_rows(series, tmp_path):
    path = tmp_path / "model.csv"
    path.write_text("SMILES,Input_SMILES\nCCO,CCO\n")
    with pytest.raises(ValueError, match="parent differs"):
        list(mol2mol_proposals(path, series.seed))
    path.write_text("SMILES,Input_SMILES\ninvalid," + series.seed + "\n")
    rows = list(mol2mol_proposals(path, series.seed))
    assert len(rows) == 1 and series.classify(rows[0]["smiles"])[0] == "invalid"
    result = run(CONFIG, tmp_path / "model_import", path)
    assert result["methods"]["mol2mol"]["statuses"] == {"invalid": 1}
    assert result["mol2mol_input_sha256"]


def test_budget_is_per_strategy_and_no_fallback_for_unrelated_seed(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["budget_per_method"] = 1
    config["blocks"] = str(ROOT / "data/round2_building_blocks.csv")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    summary = run(path, tmp_path / "run")
    assert all(s["processed_rows"] == 1 and s["truncated"] for s in summary["methods"].values())
    config["seed_smiles"] = "CCO"
    with pytest.raises(ValueError, match="reconstruct"):
        Series(config)
