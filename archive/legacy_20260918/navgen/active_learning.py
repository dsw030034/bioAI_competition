"""Diverse oracle batch selection from model uncertainty and OOD candidates."""

import math

from .chemistry import fingerprint, max_similarity, molecule
from .generation import read_seeds
from .io import read_smiles, write_csv
from .scoring import Scorer


def select(config, pool, output, count=32, beta=1.0, diversity=0.85):
    from pathlib import Path
    if not config["selectivity"]["model"]:
        raise ValueError("Active learning requires a trained selectivity model")
    if count < 1 or not math.isfinite(beta) or beta < 0 or not 0 <= diversity <= 1:
        raise ValueError("Invalid active learning parameters")
    if Path(output).exists():
        raise FileExistsError(output)
    scorer = Scorer(config, read_seeds(config["seeds_file"]))
    rows, seen = [], set()
    for raw in read_smiles(pool):
        row = scorer.score(raw)
        if not row["valid"] or set(row["reasons"]) - {"selectivity_out_of_domain"}:
            continue
        if row["candidate_id"] in seen:
            continue
        seen.add(row["candidate_id"])
        row["acquisition"] = row["ddg_pred"] + beta * row["ddg_std"]
        rows.append(row)
    rows.sort(key=lambda r: (-r["acquisition"], r["candidate_id"]))
    chosen, fps = [], []
    for row in rows:
        mol = molecule(row["smiles"])
        if not fps or max_similarity(mol, fps) <= diversity:
            chosen.append(row)
            fps.append(fingerprint(mol))
        if len(chosen) >= count:
            break
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, chosen, ["candidate_id", "smiles", "acquisition", "ddg_pred", "ddg_std",
                               "training_similarity", "selectivity_status"])
    return {"eligible": len(rows), "selected": len(chosen), "requested": count,
            "uncertainty": "heuristic, uncalibrated; selection requests new oracle labels"}

