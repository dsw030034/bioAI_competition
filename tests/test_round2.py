import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from navgen.shared_scoring import ProjectScorer
from navgen.reinvent import prepare_reinvent_config
from navgen.reaction_library import enumerate_library
from navgen.experiment import evaluate, read_reference
from navgen.site_rules import score_site_c

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/round2_local.json"
ANALOG = "COc1cc(NC(=O)c2ccc(-c3ccc(F)cc3)o2)cc(OC)c1"


def test_seed_constraints_and_failed_rewards():
    scorer = ProjectScorer(CONFIG)
    good = scorer.score(ANALOG)
    assert good["passed"] and good["reward"] > 0
    for smiles in ("CCO", "", "not_smiles", scorer.seeds[0], "CC.[Na+]"):
        assert scorer.score(smiles)["reward"] == 0
    assert good["similarity_to_seed"] > .35


def test_rl_bridge_from_unrelated_cwd(tmp_path):
    result = prepare_reinvent_config(tmp_path / "run with space", "dummy.prior", project_config=CONFIG)
    seeds = Path(result["input_smiles"]).read_text().strip()
    assert seeds == ProjectScorer(CONFIG).seeds[0]
    config = tomllib.loads(Path(result["staged_learning"]).read_text())
    endpoint = config["stage"][0]["scoring"]["component"][0]["ExternalProcess"]["endpoint"][0]
    args = endpoint["params"]
    proc = subprocess.run([args["executable"], *shlex.split(args["args"])],
                          cwd=tmp_path, input=ANALOG + "\n\nCCO\n", text=True, capture_output=True, check=True)
    values = json.loads(proc.stdout)["payload"]["nav18_reward"]
    assert len(values) == 3 and values[0] > 0 and values[1:] == [0, 0]
    assert config["diversity_filter"]["type"] == "PenalizeSameSmiles"
    with pytest.raises(FileExistsError):
        prepare_reinvent_config(tmp_path / "run with space", "dummy.prior", project_config=CONFIG)


def test_recombination_is_seed_linked():
    scorer = ProjectScorer(CONFIG)
    rows = list(enumerate_library(scorer.seeds, ROOT / "data/round2_building_blocks.csv"))
    assert len(rows) >= 30
    assert all(r["parent"] == scorer.seeds[0] for r in rows)
    assert any(scorer.score(r["smiles"])["passed"] for r in rows)
    from rdkit.Chem import rdChemReactions
    for row in rows:
        reaction = rdChemReactions.ReactionFromSmarts(row["reaction_smarts"])
        products = reaction.RunReactants(tuple(Chem.MolFromSmiles(s) for s in json.loads(row["reactants"])))
        assert row["smiles"] in {Chem.MolToSmiles(p[0]) for p in products}


def test_amide_not_basic_center():
    assert not score_site_c(Chem.MolFromSmiles("CC(=O)Nc1ccccc1"), {})["site_c_features"]["basic_center"]
    assert score_site_c(Chem.MolFromSmiles("CCNCC"), {})["site_c_features"]["basic_center"]


def test_common_evaluator_counts(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text("SMILES\n" + ANALOG + "\n" + ANALOG + "\nbad-smiles\nCCO\n")
    result = evaluate(CONFIG, path, tmp_path / "eval", "test", budget=3)
    assert result["raw_rows_evaluated"] == 3
    assert result["unique"] == 1 and result["valid"] == 2
    assert result["accepted_2d"] == 1


def test_exploration_requires_real_3d(tmp_path):
    seed = ProjectScorer(CONFIG).seeds[0]
    path = tmp_path / "ref.sdf"
    mol = Chem.MolFromSmiles(seed)
    AllChem.Compute2DCoords(mol)
    with Chem.SDWriter(str(path)) as w:
        w.write(mol)
    with pytest.raises(ValueError, match="3D"):
        read_reference(path, seed)
    with pytest.raises(ValueError, match="identity"):
        read_reference(path, "CCO")


def test_exploration_keeps_review_gate(tmp_path):
    seed = ProjectScorer(CONFIG).seeds[0]
    refpath = tmp_path / "ref.sdf"
    genpath = tmp_path / "generated.sdf"
    for smiles, path in ((seed, refpath), (ANALOG, genpath)):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
        with Chem.SDWriter(str(path)) as w:
            w.write(mol)
    result = evaluate(ROOT / "configs/round2_exploration.json", genpath, tmp_path / "out", "synthetic_fixture", reference=refpath)
    assert result["accepted_2d"] == 1
    rows = list(csv.DictReader((tmp_path / "out/candidates.csv").open()))
    assert rows[0]["ready_for_docking"] == "False"
    assert 0 <= float(rows[0]["shape_tanimoto"]) <= 1
