from pathlib import Path

from navgen.generation import generate_candidates_for_seed
from navgen.pipeline import run_generation_pipeline


def test_a803467_generation_produces_seed_analogues():
    seed = "COc1cc(NC(=O)c2ccc(-c3ccc(Cl)cc3)o2)cc(OC)c1"
    records = generate_candidates_for_seed(seed, max_candidates=20)

    assert records
    assert all(record["seed"] == seed for record in records)
    assert all(record["source"] == "seed_analogue" for record in records)
    assert all(record["canonical_smiles"] != seed for record in records)
    assert any("aryl_Cl_to_F" in record["route"] for record in records)


def test_pipeline_generates_candidates(tmp_path):
    config_path = Path(__file__).resolve().parents[1] / "configs" / "demo.json"
    output_dir = tmp_path / "demo_run"
    result = run_generation_pipeline(config_path, output_dir)

    assert output_dir.joinpath("candidates.csv").exists()
    assert output_dir.joinpath("molecules.sqlite").exists()
    assert len(result["accepted"]) > 0
    assert "protein_word_score" in result["accepted"][0]
    assert output_dir.joinpath("manifest.json").read_text(encoding="utf-8").find('"protein_word_scoring": true') >= 0
