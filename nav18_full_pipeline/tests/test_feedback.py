import pandas as pd
import pytest

from navgen.feedback import (
    RANKING_DOCKING_TRIAGE,
    build_next_seeds,
    feedback_from_scored_tsv,
    load_feedback,
    rank_feedback,
)


def _feedback_frame():
    return pd.DataFrame(
        {
            "candidate_id": ["m1", "m2", "m3", "m4"],
            "smiles": [
                "O=C(Nc1ccccc1)c1ccccc1Cl",
                "O=C(Nc1ccccc1)c1ccccc1F",
                "O=C(Nc1ccccc1)c1ccccc1",
                "O=C(Nc1ccccc1)Oc1ccccc1",
            ],
            "status": ["ok", "ok", "docking_failed", "ok"],
            "nav18_docking": [-7.2, -7.2, -5.0, -6.9],
            "nav15_docking": [-7.0, -7.1, -5.1, -6.8],
            "heavy_atoms": [13, 12, 13, 12],
            "ddg_kcal_mol": [0.2, 0.1, None, -0.3],
            "species": "Homo sapiens",
            "state": "inactivated",
            "protocol": "test_protocol",
            "label_source": "experiment",
        }
    )


def test_rank_feedback_excludes_non_ok_status():
    ranked = rank_feedback(_feedback_frame(), RANKING_DOCKING_TRIAGE)
    assert "m3" not in set(ranked["candidate_id"])


def test_docking_triage_orders_by_normalized_affinity():
    ranked = rank_feedback(_feedback_frame(), RANKING_DOCKING_TRIAGE)
    # -nav18/heavy_atoms: m1 = 7.2/13≈0.554, m2 = 7.2/12=0.6, m4 = 6.9/12=0.575
    # m2 应排最前。
    assert ranked.iloc[0]["candidate_id"] == "m2"


def test_build_next_seeds_dedup_similar_neighbours():
    frame = pd.DataFrame(
        {
            "smiles": ["O=C(Nc1ccccc1)c1ccccc1Cl", "O=C(Nc1ccccc1)c1ccccc1F"] * 6,
            "status": ["ok"] * 12,
            "nav18_docking": [-7.2] * 12,
            "heavy_atoms": [13] * 12,
        }
    )
    ranked = rank_feedback(frame, RANKING_DOCKING_TRIAGE)
    seeds, audit = build_next_seeds(ranked, top_k=8, explore_ratio=0.0, similarity_cap=0.5)
    # 相似分子（仅 Cl/F 之差）应被去近邻，最终保留更少种子。
    assert 0 < len(seeds) < len(ranked)
    assert audit["n_input"] == len(ranked)


def test_feedback_from_scored_tsv_maps_columns(tmp_path):
    tsv = tmp_path / "subtype_scores.tsv"
    tsv.write_text(
        "label\tsmiles\tscore_nav18\tscore_nav15\tdelta_sel\tqed\tsa\tmembrane\tcombined\n"
        "m1\tO=C(Nc1ccccc1)c1ccccc1Cl\t-7.2\t-7.0\t+0.2\t0.5\t2.5\t0.8\t0.6\n",
        encoding="utf-8",
    )
    frame = feedback_from_scored_tsv(tsv)
    assert list(frame["status"]) == ["ok"]
    assert float(frame.iloc[0]["nav18_docking"]) == -7.2
    assert int(frame.iloc[0]["heavy_atoms"]) > 0


def test_load_feedback_rejects_unknown_status(tmp_path):
    frame = _feedback_frame()
    frame.loc[0, "status"] = "bogus"
    path = tmp_path / "fb.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_feedback(path)


def test_load_feedback_resolves_smiles_from_database(tmp_path):
    import sqlite3

    db = tmp_path / "molecules.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE molecules (candidate_id TEXT PRIMARY KEY, smiles TEXT)")
    conn.execute("INSERT INTO molecules VALUES ('m1', 'O=C(Nc1ccccc1)c1ccccc1Cl')")
    conn.commit()
    conn.close()

    fb = tmp_path / "fb.csv"
    pd.DataFrame({"candidate_id": ["m1"], "status": ["ok"]}).to_csv(fb, index=False)
    frame = load_feedback(fb, database=db)
    assert frame.iloc[0]["smiles"] == "O=C(Nc1ccccc1)c1ccccc1Cl"
