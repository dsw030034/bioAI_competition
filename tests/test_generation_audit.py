import json
from pathlib import Path

import pytest
from rdkit import Chem

from navgen.generation_audit import diagnose, describe_sites, summarize
from navgen.series_generation import Series


@pytest.fixture
def series():
    path = Path(__file__).resolve().parents[1] / "configs/a803467_generation.json"
    return Series(json.loads(path.read_text()))


def test_symmetry_related_groups_have_identical_descriptions(series):
    a, b = dict(series.original), dict(series.original)
    a[2], b[3] = "O", "O"
    results = []
    for groups in (a, b):
        status, mol = series.classify(series.assemble(groups))
        explanation = diagnose(series, mol, status)
        assert explanation["constraint_reasons"] == []
        results.append(describe_sites(series, mol, explanation["core_match"]))
    assert results[0] == results[1]
    assert results[0]["changed_site_count"] == 1


def test_constraint_explanations_distinguish_site_nh_and_core_changes(series):
    cases = [(series.seed.replace("NC(=O)", "N(C)C(=O)"), {"amide_NH_changed"}),
             (series.seed.replace("c3ccc(Cl)cc3", "c3cc(Cl)ccc3"), {"offsite_substitution"}),
             (series.seed.replace(")o2", ")s2"), {"full_core_missing", "round2_core_missing"})]
    for smiles, expected in cases:
        status, mol = series.classify(smiles)
        assert set(diagnose(series, mol, status)["constraint_reasons"]) == expected


def test_bridge_explanation_and_new_fragment_not_confused_with_illegal_site(series):
    rw = Chem.RWMol(series.core)
    a, b = rw.AddAtom(Chem.Atom(6)), rw.AddAtom(Chem.Atom(6))
    rw.AddBond(series.anchors[2], a, Chem.BondType.SINGLE)
    rw.AddBond(series.anchors[3], b, Chem.BondType.SINGLE)
    rw.AddBond(a, b, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    status, mol = series.classify(Chem.MolToSmiles(mol))
    assert "bridge_or_fusion" in diagnose(series, mol, status)["constraint_reasons"]
    groups = dict(series.original)
    groups[2] = "SC"
    status, mol = series.classify(series.assemble(groups))
    assert status == "strict"
    diag = diagnose(series, mol, status)
    assert describe_sites(series, mol, diag["core_match"])["outside_enumerator_vocabulary"]


def test_single_and_pair_are_one_method_and_duplicates_are_not_unique_credit():
    events = [{"method": "site_single", "status": "strict"}, {"method": "site_pair", "status": "strict"},
              {"method": "mol2mol", "status": "strict"}]
    rows = [{"methods": ["site_enumeration", "mol2mol"], "scope": "strict", "changed_site_count": 1,
             "constraint_reasons": []}]
    result = summarize(events, rows)
    enum = next(r for r in result["contributions"] if r["method"] == "site_enumeration" and r["scope"] == "strict")
    assert enum["raw_rows"] == 2
    assert enum["unique_structures"] == 1 and enum["exclusive_structures"] == 0
