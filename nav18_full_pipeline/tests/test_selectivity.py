import pandas as pd
import pytest

from navgen.selectivity import (
    SelectivityModel,
    load_paired_labels,
    murcko_scaffold,
    train_selectivity_model,
)


def _synthetic_labels(tmp_path):
    entries = [
        ("O=C(Nc1ccccc1)c1ccccc1", 0.0),
        ("O=C(Nc1ccc(Cl)cc1)c1ccccc1", -0.6),
        ("O=C(Nc1ccc(F)cc1)c1ccccc1", 0.2),
        ("O=C(Nc1cccc(C)c1)c1ccccc1", 0.0),
        ("COc1ccc(NC(=O)c2ccccc2)cc1", 0.1),
        ("O=C(Nc1ccccc1)c1ccco1", 0.0),
        ("O=C(Nc1ccc(Cl)cc1)c1ccco1", -0.6),
        ("O=C(Nc1ccc(F)cc1)c1ccco1", 0.2),
        ("O=C(Nc1ccccc1)c1ccc(-c2ccccc2)o1", 0.1),
        ("O=C(Nc1ccccc1)c1ccccn1", 0.1),
        ("O=C(Nc1ccc(Cl)cc1)c1ccccn1", -0.5),
        ("O=C(Nc1ccc(F)cc1)c1ccccn1", 0.3),
        ("O=C(Nc1ccccc1)c1cccnc1", 0.1),
        ("O=C(Nc1ccccc1)c1cccs1", 0.0),
    ]
    df = pd.DataFrame(
        {
            "smiles": [s for s, _ in entries],
            "ddg_kcal_mol": [v for _, v in entries],
            "species": "Homo sapiens",
            "state": "inactivated",
            "protocol": "test_protocol",
            "label_source": "experiment",
        }
    )
    path = tmp_path / "paired_labels.csv"
    df.to_csv(path, index=False)
    return path


def test_load_paired_labels_rejects_wrong_species(tmp_path):
    path = _synthetic_labels(tmp_path)
    df = pd.read_csv(path)
    df["species"] = "Rattus norvegicus"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_paired_labels(path)


def test_load_paired_labels_rejects_multiple_contexts(tmp_path):
    path = _synthetic_labels(tmp_path)
    df = pd.read_csv(path)
    df.loc[0, "state"] = "activated"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_paired_labels(path)


def test_train_returns_model_with_scaffold_cv(tmp_path):
    path = _synthetic_labels(tmp_path)
    model, metrics = train_selectivity_model(path)
    assert isinstance(model, SelectivityModel)
    assert metrics["n_scaffolds"] >= 3
    assert metrics["cv"] is not None
    assert "mae_kcal_mol" in metrics["cv"]
    assert model.metadata()["state"] == "inactivated"


def test_predict_and_reward_are_bounded(tmp_path):
    path = _synthetic_labels(tmp_path)
    model, _ = train_selectivity_model(path)
    mean, std = model.predict("O=C(Nc1ccccc1)c1ccccc1")
    assert isinstance(mean, float) and isinstance(std, float)
    reward = model.selectivity_reward("O=C(Nc1ccccc1)c1ccccc1")
    assert 0.0 <= reward <= 1.0


def test_context_mismatch_rejected_on_load(tmp_path):
    path = _synthetic_labels(tmp_path)
    model, _ = train_selectivity_model(path)
    saved = model.save(tmp_path / "model.joblib")
    with pytest.raises(ValueError):
        SelectivityModel.load(saved, state="activated")


def test_ood_detection_for_dissimilar_molecule(tmp_path):
    path = _synthetic_labels(tmp_path)
    model, _ = train_selectivity_model(path)
    # 乙酸与酰胺训练集相似性极低，应判定为域外。
    assert model.is_ood("CC(=O)O")


def test_murcko_scaffold_is_nonempty():
    assert murcko_scaffold("O=C(Nc1ccccc1)c1ccccc1") != ""
