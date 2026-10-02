import json
from pathlib import Path
import sqlite3

import pytest
from rdkit import Chem

from navgen.conformers import embed
from navgen.pipeline import run


def test_reproducible_pipeline_and_provenance(config, tmp_path):
    first = run(config)
    out = Path(config["output_dir"])
    assert first["accepted"] > 10
    assert first["validity"] == 1
    assert first["stop_reason"] == "generator_exhausted"
    assert first["novelty"] is None
    original = (out / "candidates.csv").read_bytes()
    with sqlite3.connect(out / "molecules.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == first["attempts"]
        assert db.execute("SELECT COUNT(*) FROM molecules WHERE passed=1").fetchone()[0] == first["accepted"]
        payload = json.loads(db.execute("SELECT payload FROM molecules WHERE passed=1 LIMIT 1").fetchone()[0])
        assert payload["route"]["reactants"]
    config["output_dir"] = str(tmp_path / "second")
    second = run(config)
    assert first == second
    assert original == (tmp_path / "second/candidates.csv").read_bytes()
    with pytest.raises(FileExistsError):
        run(config)


def test_import_counts_invalid_duplicate_and_novelty(config, tmp_path):
    pool = tmp_path / "clm.csv"
    pool.write_text("SMILES,NLL\nCCO,1\nOCC,2\nbad-smiles,3\n", encoding="utf-8")
    config.update(backend="smiles", input_smiles=str(pool), reference_file=config["seeds_file"])
    stats = run(config)
    assert stats["attempts"] == 3
    assert stats["valid"] == 2
    assert stats["unique_valid"] == 1
    assert stats["duplicates"] == 1
    assert stats["uniqueness"] == .5
    assert stats["accepted"] == 0
    assert stats["novelty"]["molecule_novel_fraction"] is None


def test_candidate_and_attempt_budget(config):
    config["max_candidates"] = 2
    report = run(config)
    assert report["accepted"] == 2
    assert report["stop_reason"] == "candidate_limit"


def test_attempt_budget(config):
    config["max_attempts"] = 1
    report = run(config)
    assert report["attempts"] == 1
    assert report["stop_reason"] == "attempt_limit"


def test_conformer_export(config):
    config["conformers"].update(enabled=True, max_molecules=2, num_confs=3, max_iterations=1000)
    run(config)
    out = Path(config["output_dir"])
    statuses = json.loads((out / "conformer_status.json").read_text())
    assert len(statuses) == 2
    assert all(row["status"] == "ok" for row in statuses)
    mols = list(Chem.SDMolSupplier(str(out / "best_conformers.sdf"), removeHs=False))
    assert len(mols) == 2
    assert all(m is not None and m.GetConformer().Is3D() and m.HasProp("candidate_id") for m in mols)


def test_embedding_failure_is_missing(monkeypatch):
    monkeypatch.setattr("navgen.conformers.AllChem.EmbedMultipleConfs", lambda *a, **k: [])
    mol, status = embed("CCO", 42)
    assert mol is None
    assert status["status"] == "embedding_failed"
    assert status["energy"] is None

