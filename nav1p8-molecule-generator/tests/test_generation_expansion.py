import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from rdkit import Chem

from navgen.expansion_blocks import build_blocks, cross_proposals
from navgen.generation_expansion import run, vocabulary_evidence
from navgen.generation_audit import method_family
from navgen.series_generation import Series, run as original_run


ROOT = Path(__file__).resolve().parents[1]


def test_cross_products_reconstruct_tertiary_amide_and_keep_both_block_sources(tmp_path):
    path = tmp_path / "blocks.csv"
    path.write_text("id,role,smiles\na1,acid,CC(=O)O\nb1,amine,CNC\n")
    parent = "CC(=O)N(C)C"
    blocks, diagnostics = build_blocks(path, [{"candidate_id": "parent", "scope": "expanded_round2", "smiles": parent}], ["expanded_round2"])
    assert not diagnostics and len(blocks) == 2
    assert all(any(s["type"] == "retrosplit_reviewed_candidate" for s in b["sources"]) for b in blocks)
    records = list(cross_proposals(blocks, parent))
    assert len(records) == 1
    assert records[0]["smiles"] == Chem.MolToSmiles(Chem.MolFromSmiles(parent))
    assert records[0]["acid_block_id"] and records[0]["amine_block_id"]
    assert records[0]["reactants"] == ["CC(=O)O", "CNC"]


def test_expansion_preserves_all_old_scopes_lineage_and_invalid_sampling(tmp_path):
    config = json.loads((ROOT / "configs/a803467_generation.json").read_text())
    config["sites"][0]["fragments"] = ["Cl", "F"]
    for site in config["sites"][1:]:
        site["fragments"] = ["OC", "O"]
    blocks = tmp_path / "blocks.csv"
    blocks.write_text("id,role,smiles\na1,acid,O=C(O)c1ccc(-c2ccc(Cl)cc2)o1\nb1,amine,Nc1cc(OC)cc(OC)c1\n")
    config["blocks"] = str(blocks)
    old_config = tmp_path / "original.json"
    old_config.write_text(json.dumps(config))
    seed = config["seed_smiles"]
    methylated = seed.replace("NC(=O)", "N(C)C(=O)")
    previous_csv = tmp_path / "old.csv"
    with previous_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SMILES", "Input_SMILES"])
        writer.writerows([["CCO", seed], [methylated, seed]])
    baseline = tmp_path / "baseline"
    original_run(old_config, baseline, previous_csv)
    expansion = dict(config, budgets={"site_single": 100, "site_pair": 100, "site_triple": 100,
                                     "amide_cross_product": 100, "mol2mol": 100},
                     derive_blocks_from_scopes=["strict", "expanded_round2", "seed_copy"],
                     retain_scopes=["strict", "expanded_round2", "outside_series"], review_context="test review")
    config_path = tmp_path / "expansion.json"
    config_path.write_text(json.dumps(expansion))
    model_csv = tmp_path / "new.csv"
    with model_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SMILES", "Input_SMILES", "sampling_batch"])
        writer.writerows([["CCC", seed, 1], ["CCC", seed, 2], ["invalid", seed, 2]])
    output = tmp_path / "expanded"
    summary = run(config_path, baseline, output, model_csv, images=False)
    rows = json.loads((output / "audit_records.json").read_text(encoding="utf8"))
    assert {r["scope"] for r in rows} == {"strict", "expanded_round2", "outside_series"}
    old_events = [json.loads(line) for line in (baseline / "lineage.jsonl").read_text(encoding="utf8").splitlines()]
    events = [json.loads(line) for line in (output / "lineage.jsonl").read_text(encoding="utf8").splitlines()]
    assert len({r["event_id"] for r in events}) == len(events)
    assert {r["candidate_id"] for r in old_events if r["status"] in expansion["retain_scopes"]}.issubset({r["candidate_id"] for r in rows})
    assert sum(r["status"] == "invalid" for r in events) == 1
    assert summary["methods_new_run"]["mol2mol"]["raw_rows"] == 3
    growth = json.loads((output / "mol2mol_batch_growth.json").read_text())
    assert growth[0]["new_vs_previous_batches"] == 1 and growth[1]["new_vs_previous_batches"] == 0
    outside = list(Chem.SDMolSupplier(str(output / "outside_candidates_2d.sdf")))
    assert len(outside) == 2 and all(not m.GetConformer().Is3D() for m in outside)
    assert any(r.get("changed_site_count") == 3 for r in rows)
    with pytest.raises(FileExistsError):
        run(config_path, baseline, output, model_csv, images=False)
    # Explicit budgets may not silently drop model rows or the deterministic space.
    expansion["budgets"]["site_triple"] = 1
    config_path.write_text(json.dumps(expansion))
    # This fixture has exactly one triple assignment, so force a smaller single-site budget instead.
    expansion["budgets"]["site_single"] = 1
    config_path.write_text(json.dumps(expansion))
    with pytest.raises(ValueError, match="no silent truncation"):
        run(config_path, baseline, tmp_path / "overflow", images=False)
    assert not (tmp_path / "overflow").exists()


def test_unobserved_fragment_not_silently_added_and_method_families_stable():
    config = json.loads((ROOT / "configs/a803467_generation.json").read_text())
    base = Series(config)
    new = json.loads(json.dumps(config))
    new["sites"][0]["fragments"].append("CCCCCCCC")
    with pytest.raises(ValueError, match="lacks site-specific reviewed evidence"):
        vocabulary_evidence(base, Series(new), [])
    assert method_family("site_triple") == "site_enumeration"
    assert method_family("amide_cross_product") == "amide_recombination"


def test_remote_batch_merge_checks_hash_parent_and_increments(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("batch_runner_test", ROOT / "scripts/run_mol2mol_batches.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for batch, samples in ((1, ["CCO", "CCC"]), (2, ["CCC", "invalid"])):
        directory = tmp_path / f"batch_{batch:02d}"
        directory.mkdir()
        path = directory / "sampling_mol2mol.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["SMILES", "Input_SMILES"])
            writer.writerows([[s, "CC"] for s in samples])
        (directory / "manifest.json").write_text(json.dumps({"status": "completed", "output_sha256": module.sha(path),
                                                           "random_seed": 10 + batch, "samples": 2}))
    result = module.collect(tmp_path, "CC", {"CCO"}, [11, 12], 2)
    assert result["raw_rows"] == 4 and result["new_vs_baseline"] == 1
    assert result["batch_statistics"][1]["invalid_rows"] == 1
    assert result["batch_statistics"][1]["new_vs_previous_batches_excluding_seed"] == 0
    with pytest.raises(ValueError, match="parent mismatch"):
        module.collect(tmp_path, "CO", set(), [11, 12], 2)
    (tmp_path / "batch_02/sampling_mol2mol.csv").write_text("changed")
    with pytest.raises(ValueError, match="modified batch"):
        module.collect(tmp_path, "CC", set(), [11, 12], 2)
