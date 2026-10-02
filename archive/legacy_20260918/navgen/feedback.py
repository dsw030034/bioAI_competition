"""Validated downstream feedback -> diverse top-k and exploration seeds."""

import csv
import json
import math
from pathlib import Path
import random
import sqlite3

from .chemistry import fingerprint, max_similarity, molecule
from .io import write_json


def select_seeds(database, feedback, output, config, top_k=20, exploration=0.2, max_similarity_allowed=0.85, ranking="ddg"):
    if top_k < 1 or not 0 <= exploration <= 1 or not 0 <= max_similarity_allowed <= 1:
        raise ValueError("Invalid top_k/exploration/diversity threshold")
    if ranking not in {"ddg", "docking_triage"}:
        raise ValueError("ranking must be ddg or docking_triage")
    output = Path(output)
    if output.exists() or output.with_suffix(".feedback.json").exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    eligible, audit, seen = [], [], set()
    with sqlite3.connect(f"{Path(database).resolve().as_uri()}?mode=ro", uri=True) as conn:
        with open(feedback, encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"candidate_id", "status", "species", "state", "protocol", "label_source"}
            required.add("ddg_kcal_mol" if ranking == "ddg" else "nav18_docking")
            if not required <= set(reader.fieldnames or []):
                raise ValueError(f"Feedback requires {sorted(required)}")
            for row in reader:
                identifier = row["candidate_id"]
                if identifier in seen:
                    raise ValueError(f"Duplicate feedback candidate: {identifier}")
                seen.add(identifier)
                record = conn.execute("SELECT payload,passed FROM molecules WHERE candidate_id=?", (identifier,)).fetchone()
                if not record or not record[1]:
                    raise ValueError(f"Unknown or rejected candidate: {identifier}")
                for key in ("species", "state", "protocol"):
                    if row[key] != config[key] or row[key] == "unspecified":
                        raise ValueError(f"Feedback context missing/mismatch: {key}")
                if row["status"] not in {"ok", "embedding_failed", "docking_failed", "missing"}:
                    raise ValueError(f"Unknown feedback status: {row['status']}")
                audit.append(row)
                if row["status"] != "ok":
                    continue  # Missing truth is never a zero-valued label.
                if ranking == "ddg" and row["label_source"] not in {"experiment", "free_energy"}:
                    raise ValueError("Top-k selectivity feedback requires experiment/free_energy labels, not docking deltas")
                if ranking == "docking_triage" and row["label_source"] != "docking":
                    raise ValueError("docking_triage requires label_source=docking")
                payload = json.loads(record[0])
                value = (float(row["ddg_kcal_mol"]) if ranking == "ddg"
                         else -float(row["nav18_docking"]) / payload["heavy_atoms"])
                if not math.isfinite(value):
                    raise ValueError("Feedback value must be finite")
                eligible.append({**payload, "feedback_priority": value})
    eligible.sort(key=lambda r: (-r["feedback_priority"], r["candidate_id"]))
    exploit_count = top_k - int(top_k * exploration)
    chosen, fps = [], []
    for row in eligible:
        if len(chosen) >= exploit_count:
            break
        mol = molecule(row["smiles"])
        if not fps or max_similarity(mol, fps) <= max_similarity_allowed:
            chosen.append({**row, "selection": "top_k"})
            fps.append(fingerprint(mol))
    selected_ids = {r["candidate_id"] for r in chosen}
    remaining = [r for r in eligible if r["candidate_id"] not in selected_ids]
    random.Random(config["seed"]).shuffle(remaining)
    for row in remaining:
        if len(chosen) >= top_k:
            break
        mol = molecule(row["smiles"])
        if not fps or max_similarity(mol, fps) <= max_similarity_allowed:
            chosen.append({**row, "selection": "exploration"})
            fps.append(fingerprint(mol))
    if not chosen:
        raise ValueError("No successfully evaluated molecules available; failed/missing scores are not feedback")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in chosen:
            handle.write(row["smiles"] + "\t" + row["candidate_id"] + "\n")
    report = {"requested": top_k, "selected": len(chosen), "eligible": len(eligible),
              "seed": config["seed"], "selected_records": chosen, "feedback": audit,
              "ranking": ranking, "selectivity_evidence": ranking == "ddg",
              "priority_definition": "DG15-DG18" if ranking == "ddg" else "-Nav1.8 docking / heavy_atoms (triage only)"}
    write_json(output.with_suffix(".feedback.json"), report)
    return report
