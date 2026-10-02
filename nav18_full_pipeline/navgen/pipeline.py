from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - fallback for minimal environments
    Chem = None

from .config import FilterConfig, load_config
from .conformers import export_conformers
from .evaluation import evaluate_library
from .generation import generate_candidates_for_seed, load_seed_smiles
from .protein_rules import ProteinWordScorer, load_protein_sequence
from .scoring import score_candidate
from .selectivity import load_model_from_config
from .site_rules import load_site_profile, score_site_c


def _init_sqlite(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS molecules (
            candidate_id TEXT PRIMARY KEY,
            seed TEXT,
            source TEXT,
            route TEXT,
            smiles TEXT,
            reward REAL,
            qed REAL,
            mw REAL,
            tpsa REAL,
            logp REAL,
            passed INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS attempts (
            candidate_id TEXT,
            seed TEXT,
            source TEXT,
            route TEXT,
            smiles TEXT,
            passed INTEGER,
            reason TEXT
        )
        """
    )
    return conn


def _row_from_candidate(record: dict[str, Any], metrics: dict[str, Any], passed: bool) -> dict[str, Any]:
    return {
        "candidate_id": record["candidate_id"],
        "seed": record["seed"],
        "source": record["source"],
        "route": record["route"],
        "smiles": record["canonical_smiles"],
        "reward": float(metrics.get("reward", 0.0)),
        "qed": float(metrics.get("qed", 0.0)),
        "mw": float(metrics.get("mw", 0.0)),
        "tpsa": float(metrics.get("tpsa", 0.0)),
        "logp": float(metrics.get("logp", 0.0)),
        "protein_word_score": float(metrics.get("protein_word_score", 0.0)),
        "matched_rule_count": int(metrics.get("matched_rule_count", 0)),
        "protein_word_coverage": float(metrics.get("protein_word_coverage", 0.0)),
        "matched_rules": json.dumps(metrics.get("matched_rules", []), ensure_ascii=False),
        "site_c_score": float(metrics.get("site_c_score", 0.0)),
        "fenestration_score": float(metrics.get("fenestration_score", 0.0)),
        "nav18_proxy_score": float(metrics.get("nav18_proxy_score", 0.0)),
        "nav15_counter_penalty": float(metrics.get("nav15_counter_penalty", 0.0)),
        "selectivity_proxy": float(metrics.get("selectivity_proxy", 0.0)),
        "reinvent_geometric_mean": float(metrics.get("reinvent_geometric_mean", 0.0)),
        "selectivity_mean": float(metrics.get("selectivity_mean", 0.0)) if metrics.get("selectivity_mean") is not None else None,
        "selectivity_std": float(metrics.get("selectivity_std", 0.0)) if metrics.get("selectivity_std") is not None else None,
        "selectivity_reward": float(metrics.get("selectivity_reward", 0.0)) if metrics.get("selectivity_reward") is not None else None,
        "selectivity_ood": bool(metrics.get("selectivity_ood", False)),
        "passed": int(bool(passed)),
    }


def run_generation_pipeline(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    cfg = load_config(config_path)
    filters = cfg["filter_config"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = load_seed_smiles(cfg["seed_file"])
    protein_scorer = None
    site_profile = load_site_profile(cfg["site_profile_file"]) if cfg.get("site_profile_file") else None
    selectivity_model = load_model_from_config(cfg)
    if cfg.get("protein_sequence_file"):
        protein_scorer = ProteinWordScorer(
            load_protein_sequence(cfg["protein_sequence_file"]),
            word_size=int(cfg.get("protein_word_size", 5)),
        )
        (output_dir / "protein_word_rules.json").write_text(
            json.dumps(
                {
                    "sequence_file": cfg["protein_sequence_file"],
                    "word_size": protein_scorer.word_size,
                    "method": "contiguous-word prototype; replace with protein-wordwise/ESM-2 rules when available",
                    "rules": protein_scorer.rules_as_dict(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    db_path = output_dir / "molecules.sqlite"
    conn = _init_sqlite(db_path)

    for seed in seeds:
        seed_mol = Chem.MolFromSmiles(seed) if Chem is not None else seed
        for record in generate_candidates_for_seed(seed, max_candidates=filters.max_candidates):
            mol = Chem.MolFromSmiles(record["canonical_smiles"]) if Chem is not None else record["canonical_smiles"]
            if Chem is not None and mol is None:
                rejected.append({**record, "reason": "invalid_smiles"})
                conn.execute(
                    "INSERT INTO attempts (candidate_id, seed, source, route, smiles, passed, reason) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (record["candidate_id"], seed, record["source"], record["route"], record["canonical_smiles"], 0, "invalid_smiles"),
                )
                continue
            protein_metrics = protein_scorer.score(record["canonical_smiles"]) if protein_scorer else None
            site_metrics = score_site_c(mol, site_profile) if site_profile else None
            passed, metrics = score_candidate(mol, seed_mol, filters, protein_metrics, site_metrics, selectivity_model)
            record.update({"metrics": metrics, "passed": passed})
            conn.execute(
                "INSERT INTO attempts (candidate_id, seed, source, route, smiles, passed, reason) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (record["candidate_id"], seed, record["source"], record["route"], record["canonical_smiles"], int(bool(passed)), ";".join(metrics.get("reasons", []))),
            )
            if passed:
                accepted.append(_row_from_candidate(record, metrics, True))
                conn.execute(
                    "INSERT OR REPLACE INTO molecules (candidate_id, seed, source, route, smiles, reward, qed, mw, tpsa, logp, passed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        record["candidate_id"],
                        seed,
                        record["source"],
                        record["route"],
                        record["canonical_smiles"],
                        float(metrics.get("reward", 0.0)),
                        float(metrics.get("qed", 0.0)),
                        float(metrics.get("mw", 0.0)),
                        float(metrics.get("tpsa", 0.0)),
                        float(metrics.get("logp", 0.0)),
                        1,
                    ),
                )
            else:
                rejected.append({**record, "reason": ";".join(metrics.get("reasons", []))})
                conn.execute(
                    "INSERT OR REPLACE INTO molecules (candidate_id, seed, source, route, smiles, reward, qed, mw, tpsa, logp, passed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        record["candidate_id"],
                        seed,
                        record["source"],
                        record["route"],
                        record["canonical_smiles"],
                        float(metrics.get("reward", 0.0)),
                        float(metrics.get("qed", 0.0)),
                        float(metrics.get("mw", 0.0)),
                        float(metrics.get("tpsa", 0.0)),
                        float(metrics.get("logp", 0.0)),
                        0,
                    ),
                )

    conn.commit()
    conn.close()

    accepted_df = __import__("pandas").DataFrame(accepted)
    rejected_df = __import__("pandas").DataFrame(rejected)
    if not accepted_df.empty:
        accepted_df = accepted_df.sort_values("reward", ascending=False).reset_index(drop=True)
    if not rejected_df.empty:
        rejected_df = rejected_df.sort_values("reward" if "reward" in rejected_df.columns else "canonical_smiles", ascending=False, kind="stable").reset_index(drop=True)

    accepted_path = output_dir / "candidates.csv"
    rejected_path = output_dir / "rejected.csv"
    accepted_df.to_csv(accepted_path, index=False)
    with (output_dir / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if not rejected_df.empty:
        rejected_df.to_csv(rejected_path, index=False)
    else:
        rejected_path.write_text("candidate_id,seed,source,route,smiles,reason\n", encoding="utf-8")

    accepted_smiles = [row["smiles"] for row in accepted]
    metrics = {
        "n_seeds": len(seeds),
        "n_candidates": len(accepted) + len(rejected),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "accepted_ratio": (len(accepted) / max(len(accepted) + len(rejected), 1)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protein_word_rules": len(protein_scorer.rules) if protein_scorer else 0,
        "site_profile": cfg.get("site_profile_file"),
        "site_c_scoring": bool(site_profile),
        "selectivity_model": bool(selectivity_model),
        "library_evaluation": evaluate_library(accepted_smiles),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    manifest = {
        "config": str(config_path),
        "output_dir": str(output_dir),
        "seed_file": str(cfg["seed_file"]),
        "filters": filters.as_dict(),
        "version": "0.1.0",
        "status": "ready",
        "protein_word_scoring": bool(protein_scorer),
        "selectivity_model": selectivity_model.metadata() if selectivity_model else None,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    conformers = export_conformers(accepted, output_dir)
    return {
        "accepted": accepted,
        "rejected": rejected,
        "db": str(db_path),
        "metrics": metrics,
        "conformers": conformers,
    }
