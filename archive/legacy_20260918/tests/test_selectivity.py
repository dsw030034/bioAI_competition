import csv
from copy import deepcopy

import pytest

from navgen.chemistry import molecule
from navgen.selectivity import SelectivityModel, train


@pytest.fixture
def training_csv(tmp_path):
    # Synthetic labels test software only; these are not biological training data.
    structures = ["c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "COc1ccccc1",
                  "c1ccncc1", "Cc1ccncc1", "Oc1ccncc1", "COc1ccncc1",
                  "C1CCCCC1", "CC1CCCCC1", "OC1CCCCC1", "COC1CCCCC1",
                  "c1ccsc1", "Cc1ccsc1", "Oc1ccsc1", "COc1ccsc1"]
    path = tmp_path / "synthetic_test_labels.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["smiles", "ddg_kcal_mol", "species", "state", "protocol", "label_source"])
        writer.writeheader()
        for i, smiles in enumerate(structures):
            assert molecule(smiles)
            writer.writerow({"smiles": smiles, "ddg_kcal_mol": i / 8 - 1,
                             "species": "Homo sapiens", "state": "test_state",
                             "protocol": "synthetic_software_test", "label_source": "experiment"})
    return path


def test_scaffold_holdout_and_model_context(training_csv, config, tmp_path):
    output = tmp_path / "model.joblib"
    report = train(training_csv, output)
    assert not set(report["train_scaffolds"]) & set(report["test_scaffolds"])
    assert report["n_train"] + report["n_test"] == 16
    config.update(state="test_state", protocol="synthetic_software_test")
    model = SelectivityModel(output, config)
    prediction = model.predict("c1ccccc1")
    assert prediction["training_similarity"] == 1
    assert prediction["ddg_std"] >= 0
    assert prediction["uncertainty_calibrated"] is False
    config["state"] = "mismatched"
    with pytest.raises(ValueError, match="context mismatch"):
        SelectivityModel(output, config)


def test_no_docking_truth(training_csv, tmp_path):
    training_csv.write_text(training_csv.read_text().replace("experiment", "docking"))
    with pytest.raises(ValueError, match="not selectivity truth"):
        train(training_csv, tmp_path / "model.joblib")


def test_no_mixed_species(training_csv, tmp_path):
    training_csv.write_text(training_csv.read_text().replace("Homo sapiens", "Mus musculus", 1))
    with pytest.raises(ValueError, match="human data"):
        train(training_csv, tmp_path / "model.joblib")


def test_ood_reward_and_active_learning(training_csv, config, tmp_path):
    from navgen.active_learning import select
    from navgen.generation import read_seeds
    from navgen.scoring import Scorer
    output = tmp_path / "model.joblib"
    train(training_csv, output)
    config.update(state="test_state", protocol="synthetic_software_test")
    config["selectivity"].update(model=str(output), ood_similarity_min=1)
    config["weights"]["selectivity"] = .2
    config["filters"].update(mw=[1, 1000], tpsa=[0, 200], logp=[-10, 10], qed_min=0,
                              similarity_min=0, sa_max=10, exclude_seeds=False)
    scorer = Scorer(config, read_seeds(config["seeds_file"]))
    row = scorer.score(scorer.seeds[0])
    assert row["selectivity_status"] == "out_of_domain"
    assert row["reward"] == 0
    assert row["ddg_pred"] is not None
    pool = tmp_path / "pool.smi"
    pool.write_text("c1ccccc1\n" + scorer.seeds[0] + "\n")
    result = select(config, pool, tmp_path / "oracle.csv", count=2, diversity=1)
    assert result["selected"] == 2  # OOD molecules may be sent to the oracle.

