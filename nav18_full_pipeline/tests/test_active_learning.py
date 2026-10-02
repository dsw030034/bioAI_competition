import pandas as pd

from navgen.active_learning import load_pool, select_oracle
from navgen.selectivity import train_selectivity_model


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


def test_select_oracle_selects_and_queues_ood(tmp_path):
    model, _ = train_selectivity_model(_synthetic_labels(tmp_path))
    pool = [
        "O=C(Nc1ccccc1)c1ccccc1",
        "O=C(Nc1ccc(Cl)cc1)c1ccccc1",
        "O=C(Nc1ccc(F)cc1)c1ccccc1",
        "CC(=O)O",  # 域外分子
    ]
    result = select_oracle(model, pool, count=2, beta=1.0)
    assert len(result["selected"]) <= 2
    assert result["n_ood"] >= 1
    assert any(record["ood"] for record in result["pending_queue"])


def test_load_pool_smi(tmp_path):
    pool = tmp_path / "pool.smi"
    pool.write_text("O=C(Nc1ccccc1)c1ccccc1 name1\nO=C(Nc1ccccc1)c1ccco1 name2\n", encoding="utf-8")
    smiles = load_pool(pool)
    assert smiles == ["O=C(Nc1ccccc1)c1ccccc1", "O=C(Nc1ccccc1)c1ccco1"]
