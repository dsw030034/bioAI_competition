from __future__ import annotations

import random
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None

from .generation import canonical_smiles
from .selectivity import fingerprint, tanimoto

# status == "ok" 的分子才参与排序；其余状态（embedding_failed / docking_failed /
# missing）不参与排序或训练，仅保留在审计里。
USABLE_STATUS = {"ok"}
KNOWN_STATUS = {"ok", "embedding_failed", "docking_failed", "missing"}
REQUIRED_FEEDBACK_COLUMNS = ["smiles", "status"]

# 外部真值 / docking 初筛各自的排序字段。
RANKING_DDG = "ddg"
RANKING_DOCKING_TRIAGE = "docking_triage"


def load_feedback(path: str | Path, database: str | Path | None = None) -> pd.DataFrame:
    """读取并校验 feedback CSV。要求 smiles（或 candidate_id + database）；status 可选列。

    当 feedback 文件只有 candidate_id 而无 smiles 时，用 database（molecules.sqlite）
    反查 smiles；两者都缺则报错。
    """
    frame = pd.read_csv(path)
    if "smiles" not in frame.columns:
        if "candidate_id" not in frame.columns or database is None:
            raise ValueError(
                "feedback file must contain a 'smiles' column, or a 'candidate_id' "
                "column together with --database"
            )
        frame["smiles"] = frame["candidate_id"].astype(str).map(_smiles_lookup(Path(database)))
        frame = frame.dropna(subset=["smiles"])
    if "status" not in frame.columns:
        frame["status"] = "ok"
    frame["smiles"] = frame["smiles"].astype(str).str.strip()
    frame["status"] = frame["status"].astype(str).str.strip()
    unknown = set(frame["status"].unique()) - KNOWN_STATUS
    if unknown:
        raise ValueError(f"feedback status contains unknown values: {sorted(unknown)}")
    if Chem is not None:
        valid = frame["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
        frame = frame[valid]
    return frame.reset_index(drop=True)


def feedback_from_scored_tsv(path: str | Path) -> pd.DataFrame:
    """把 docking+打分/score.py 的 subtype_scores.tsv 转成 feedback 行。

    score.py 输出列：label, smiles, score_nav18, score_nav15, delta_sel, qed, sa,
    membrane, combined。这里把 score_nav18/score_nav15 映射为 nav18_docking /
    nav15_docking，重原子数用 RDKit 现算，ddg 留空 → 只适配 docking_triage 排序。
    """
    frame = pd.read_csv(path, sep="\t")
    if "smiles" not in frame.columns:
        raise ValueError("scored TSV must contain a 'smiles' column")
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        smiles = str(row["smiles"]).strip()
        label = str(row.get("label", "")).strip()
        rows.append(
            {
                "candidate_id": label or None,
                "smiles": smiles,
                "status": "ok",
                "nav18_docking": _as_float(row.get("score_nav18")),
                "nav15_docking": _as_float(row.get("score_nav15")),
                "heavy_atoms": _heavy_atoms(smiles),
                "ddg_kcal_mol": None,
                "species": "Homo sapiens",
                "state": None,
                "protocol": None,
                "label_source": "docking",
            }
        )
    return pd.DataFrame(rows)


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _smiles_lookup(database: Path) -> dict[str, str]:
    if not database.exists():
        raise FileNotFoundError(f"database not found: {database}")
    conn = sqlite3.connect(database)
    try:
        rows = conn.execute("SELECT candidate_id, smiles FROM molecules").fetchall()
    finally:
        conn.close()
    return {str(cid): str(smiles) for cid, smiles in rows if smiles}


def _heavy_atoms(smiles: str) -> int:
    if Chem is None:
        return len([ch for ch in smiles if ch.isupper()])
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def rank_feedback(frame: pd.DataFrame, ranking: str) -> pd.DataFrame:
    """给每个 status==ok 的行计算排序分数（越大越优先）。"""
    ok = frame[frame["status"] == "ok"].copy()
    if ok.empty:
        return ok.assign(rank_score=pd.Series(dtype=float))
    if ranking == RANKING_DDG:
        if "ddg_kcal_mol" not in ok.columns or ok["ddg_kcal_mol"].isna().all():
            raise ValueError("ddg ranking requires non-null ddg_kcal_mol column")
        ok["rank_score"] = pd.to_numeric(ok["ddg_kcal_mol"], errors="coerce")
    elif ranking == RANKING_DOCKING_TRIAGE:
        for col in ("nav18_docking", "heavy_atoms"):
            if col not in ok.columns or ok[col].isna().all():
                raise ValueError(f"docking_triage ranking requires non-null {col} column")
        nav18 = pd.to_numeric(ok["nav18_docking"], errors="coerce")
        heavy = pd.to_numeric(ok["heavy_atoms"], errors="coerce").replace(0, 1.0)
        ok["rank_score"] = -nav18 / heavy  # 更负的 Nav1.8 docking / 尺寸 更优先
    else:
        raise ValueError(f"unknown ranking mode: {ranking!r}")
    ok = ok.dropna(subset=["rank_score"]).sort_values("rank_score", ascending=False).reset_index(drop=True)
    return ok


def build_next_seeds(
    ranked: pd.DataFrame,
    top_k: int,
    explore_ratio: float = 0.2,
    similarity_cap: float = 0.5,
    seed: int = 0,
) -> tuple[list[str], dict[str, Any]]:
    """从已排序的 ok 分子中挑下一轮种子：默认 80% top-k + 20% 随机探索，再做指纹去近邻。

    候选不足时如实输出较少分子，不靠重复补足。
    """
    if ranked.empty:
        return [], {"n_input": 0, "n_selected": 0, "explore_pool": 0, "similarity_cap": similarity_cap}

    rng = random.Random(seed)
    n_top = max(0, round(top_k * (1.0 - explore_ratio)))
    n_explore = max(0, top_k - n_top)

    greedy = ranked.iloc[:n_top]["smiles"].tolist()
    remaining = ranked.iloc[n_top:]["smiles"].tolist()
    explore_pool = list(remaining)
    if explore_pool:
        rng.shuffle(explore_pool)
        greedy += explore_pool[:n_explore]

    # 指纹去近邻：顺序保留，与已保留集合的最大 Tanimoto 超过上限则跳过。
    kept: list[str] = []
    kept_fps = []
    for smiles in greedy:
        fp = fingerprint(smiles)
        if any(tanimoto(fp, existing) > similarity_cap for existing in kept_fps):
            continue
        kept.append(smiles)
        kept_fps.append(fp)

    audit = {
        "n_input": int(len(ranked)),
        "n_selected": len(kept),
        "requested_top_k": top_k,
        "explore_ratio": explore_ratio,
        "n_greedy_top": n_top,
        "n_explore_candidates": n_explore,
        "similarity_cap": similarity_cap,
        "random_seed": seed,
    }
    return kept, audit


def write_seeds(smiles_list: list[str], output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, smiles in enumerate(smiles_list):
        try:
            smiles = canonical_smiles(smiles)
        except ValueError:
            continue
        lines.append(f"{smiles} round2_seed_{i}")
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out
