from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .selectivity import SelectivityModel, fingerprint, tanimoto


def load_pool(path: str | Path) -> list[str]:
    pool_path = Path(path)
    if not pool_path.exists():
        raise FileNotFoundError(f"pool file not found: {pool_path}")
    if pool_path.suffix.lower() in (".csv", ".tsv"):
        sep = "\t" if pool_path.suffix.lower() == ".tsv" else ","
        with pool_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            column = next((c for c in ("smiles", "SMILES") if c in (reader.fieldnames or [])), None)
            if column is None:
                raise ValueError("pool CSV must contain a smiles/SMILES column")
            return [row[column].strip() for row in reader if row.get(column, "").strip()]
    return [line.split()[0].strip() for line in pool_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def select_oracle(
    model: SelectivityModel,
    pool_smiles: list[str],
    count: int,
    beta: float = 1.0,
    similarity_cap: float = 0.5,
) -> dict[str, Any]:
    """按 UCB = 预测均值 + β×树间标准差 选样，保留多样性；域外分子进入待测队列。

    返回 selected（待送 oracle 的分子）与 pending（域外待测队列）及逐分子打分。
    程序只负责挑样，不自动启动 docking/FEP 或把预测当标签。
    """
    scored = []
    pending = []
    for smiles in pool_smiles:
        try:
            mean, std = model.predict(smiles)
            ood = model.is_ood(smiles)
        except ValueError:
            continue  # 无效 SMILES 直接跳过
        ucb = mean + beta * std
        record = {
            "smiles": smiles,
            "predicted_ddg_mean": round(mean, 4),
            "tree_std": round(std, 4),
            "ucb": round(ucb, 4),
            "max_similarity_to_train": round(model.max_similarity_to_train(smiles), 4),
            "ood": bool(ood),
        }
        if ood:
            pending.append(record)
        else:
            scored.append(record)

    scored.sort(key=lambda r: r["ucb"], reverse=True)

    selected: list[dict[str, Any]] = []
    kept_fps = []
    for record in scored:
        if len(selected) >= count:
            break
        fp = fingerprint(record["smiles"])
        if any(tanimoto(fp, existing) > similarity_cap for existing in kept_fps):
            continue
        selected.append(record)
        kept_fps.append(fp)

    return {
        "selected": selected,
        "pending_queue": pending,
        "n_pool": len(pool_smiles),
        "n_in_domain": len(scored),
        "n_ood": len(pending),
        "beta": beta,
        "similarity_cap": similarity_cap,
    }


def write_oracle_batch(result: dict[str, Any], output_csv: str | Path, pending_csv: str | Path | None = None) -> Path:
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["smiles", "predicted_ddg_mean", "tree_std", "ucb", "max_similarity_to_train", "ood"]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in result["selected"]:
            writer.writerow(record)
    if pending_csv is not None:
        pending_path = Path(pending_csv)
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        with pending_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in result["pending_queue"]:
                writer.writerow(record)
    return out
