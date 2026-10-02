from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
try:
    import tomllib
except ImportError:
    import tomli as tomllib

from navgen.config import load_config, validate
from navgen.feedback import select_seeds
from navgen.io import write_json
from navgen.pipeline import run
from navgen.reinvent import prepare


@pytest.mark.parametrize("section,key,value", [
    ("filters", "mw", [500, 100]), ("filters", "similarity_min", 1.1),
    ("weights", "selectivity", .2), ("weights", "qed", float("nan")),
    ("conformers", "num_confs", 0),
])
def test_configuration_fails_closed(config, section, key, value):
    config[section][key] = value
    with pytest.raises(ValueError):
        validate(config)


def test_unknown_config_key(tmp_path):
    path = tmp_path / "config.json"
    write_json(path, {"qedd": 0.6})
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)


def test_reinvent_config_and_real_subprocess_protocol(config, tmp_path):
    prior = tmp_path / "dummy.prior"
    prior.write_bytes(b"TEST ONLY - not model weights")
    output = tmp_path / "with spaces"
    prepare(config, prior, output, num_smiles=12)
    sampling = tomllib.loads((output / "sampling.toml").read_text(encoding="utf-8"))
    rl = tomllib.loads((output / "staged_learning.toml").read_text(encoding="utf-8"))
    assert sampling["parameters"]["num_smiles"] == 12
    assert rl["learning_strategy"]["type"] == "dap"
    endpoint = rl["stage"][0]["scoring"]["component"][0]["ExternalProcess"]["endpoint"][0]
    import shlex
    proc = subprocess.run([endpoint["params"]["executable"], *shlex.split(endpoint["params"]["args"])],
                          input="CCO\ninvalid\n", capture_output=True, text=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["version"] == 1
    assert payload["payload"]["valid"] == [True, False]
    assert len(payload["payload"][endpoint["params"]["property"]]) == 2


def make_feedback(config, tmp_path, source="free_energy"):
    config.update(state="inactivated", protocol="matched_assay_v1")
    run(config)
    candidates = list(csv.DictReader(open(Path(config["output_dir"]) / "candidates.csv", encoding="utf-8")))
    rows = []
    for index, candidate in enumerate(candidates[:5]):
        rows.append({"candidate_id": candidate["candidate_id"], "status": "ok" if index else "embedding_failed",
                     "species": config["species"], "state": config["state"], "protocol": config["protocol"],
                     "label_source": source, "ddg_kcal_mol": 100 if index == 0 else index,
                     "nav18_docking": -8 - index})
    path = tmp_path / "feedback.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path, rows


def test_feedback_skips_failure_and_orders_correct_sign(config, tmp_path):
    feedback, rows = make_feedback(config, tmp_path)
    result = select_seeds(Path(config["output_dir"]) / "molecules.sqlite", feedback, tmp_path / "next.smi",
                          config, top_k=2, exploration=0, max_similarity_allowed=1)
    assert result["selected"] == 2
    ids = [r["candidate_id"] for r in result["selected_records"]]
    assert ids == [rows[4]["candidate_id"], rows[3]["candidate_id"]]
    assert rows[0]["candidate_id"] not in ids


def test_docking_is_triage_not_selectivity(config, tmp_path):
    feedback, _ = make_feedback(config, tmp_path, source="docking")
    database = Path(config["output_dir"]) / "molecules.sqlite"
    with pytest.raises(ValueError, match="not docking"):
        select_seeds(database, feedback, tmp_path / "bad.smi", config)
    result = select_seeds(database, feedback, tmp_path / "triage.smi", config, ranking="docking_triage")
    assert result["selected"] > 0
    assert result["selectivity_evidence"] is False

