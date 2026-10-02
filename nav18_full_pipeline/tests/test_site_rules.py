import json
from pathlib import Path

from rdkit import Chem

from navgen.site_rules import load_site_profile, score_site_c


def test_site_c_proxy_is_interpretable_and_bounded():
    profile_path = Path(__file__).resolve().parents[1] / "data" / "nav18_site_c_rules.json"
    profile = load_site_profile(profile_path)
    result = score_site_c(Chem.MolFromSmiles("COc1ccccc1C(=O)Nc1ccccc1"), profile)

    assert 0.0 <= result["site_c_score"] <= 1.0
    assert 0.0 <= result["selectivity_proxy"] <= 1.0
    assert "Gln355" in json.dumps(result["evidence_residues"])
