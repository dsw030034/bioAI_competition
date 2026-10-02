from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None

from .pipeline import run_generation_pipeline
from .reinvent import prepare_reinvent_config
from .config import load_config
from .protein_rules import ProteinWordScorer, load_protein_sequence
from .site_rules import load_site_profile, score_site_c
from .scoring import score_candidate
from .selectivity import load_model_from_config, train_selectivity_model, SelectivityModel
from .feedback import (
    RANKING_DDG,
    RANKING_DOCKING_TRIAGE,
    build_next_seeds,
    feedback_from_scored_tsv,
    load_feedback,
    rank_feedback,
    write_seeds,
)
from .active_learning import load_pool, select_oracle, write_oracle_batch
from .evaluation import evaluate_library


def _generate(args: argparse.Namespace) -> int:
    result = run_generation_pipeline(args.config, args.output)
    print(json.dumps({"accepted": len(result["accepted"]), "rejected": len(result["rejected"])}, indent=2))
    return 0


def _prepare_reinvent(args: argparse.Namespace) -> int:
    result = prepare_reinvent_config(
        args.output,
        args.prior,
        device=args.device,
        num_smiles=args.num_smiles,
        project_config=args.config,
        python_path=args.python,
    )
    print(json.dumps(result, indent=2))
    return 0


def _score_stdin(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else {"filter_config": None}
    filters = cfg["filter_config"]
    protein_scorer = None
    site_profile = None
    if args.config and cfg.get("protein_sequence_file"):
        protein_scorer = ProteinWordScorer(load_protein_sequence(cfg["protein_sequence_file"]))
    if args.config and cfg.get("site_profile_file"):
        site_profile = load_site_profile(cfg["site_profile_file"])
    selectivity_model = load_model_from_config(cfg) if args.config else None
    lines = [line.strip() for line in sys.stdin if line.strip()]
    rewards = []
    for line in lines:
        smiles = line
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                smiles = str(parsed.get("smiles", parsed.get("SMILES", "")))
        except json.JSONDecodeError:
            pass
        if not smiles:
            rewards.append(0.0)
            continue
        mol = Chem.MolFromSmiles(smiles) if Chem is not None else smiles
        if Chem is not None and mol is None:
            rewards.append(0.0)
            continue
        protein_metrics = protein_scorer.score(smiles) if protein_scorer else None
        site_metrics = score_site_c(mol, site_profile) if site_profile else None
        _, metrics = score_candidate(mol, None, filters or None, protein_metrics, site_metrics, selectivity_model)
        rewards.append(float(metrics["reward"]))
    print(json.dumps({"version": 1, "payload": {"nav18_reward": rewards}}))
    return 0


def _score_reinvent_file(args: argparse.Namespace) -> int:
    import pandas as pd

    cfg = load_config(args.config)
    filters = cfg["filter_config"]
    protein_scorer = (
        ProteinWordScorer(load_protein_sequence(cfg["protein_sequence_file"]))
        if cfg.get("protein_sequence_file")
        else None
    )
    site_profile = load_site_profile(cfg["site_profile_file"]) if cfg.get("site_profile_file") else None
    selectivity_model = load_model_from_config(cfg)
    frame = pd.read_csv(args.input)
    smiles_column = args.smiles_column
    if smiles_column not in frame.columns:
        raise ValueError(f"Input file does not contain SMILES column: {smiles_column}")
    rows = []
    for value in frame[smiles_column].fillna(""):
        smiles = str(value).strip()
        mol = Chem.MolFromSmiles(smiles) if Chem is not None and smiles else None
        if mol is None:
            rows.append({"smiles": smiles, "project_reward": 0.0, "valid": False})
            continue
        protein_metrics = protein_scorer.score(smiles) if protein_scorer else None
        site_metrics = score_site_c(mol, site_profile) if site_profile else None
        _, metrics = score_candidate(mol, None, filters, protein_metrics, site_metrics, selectivity_model)
        rows.append({"smiles": smiles, "project_reward": metrics["reward"], "valid": True, **metrics})
    scored = pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)
    print(json.dumps({"input_rows": len(frame), "output": str(args.output)}, indent=2))
    return 0


def _train_selectivity(args: argparse.Namespace) -> int:
    model, metrics = train_selectivity_model(args.data)
    out = model.save(args.output)
    sidecar = Path(args.output).with_suffix(".metrics.json")
    sidecar.write_text(
        json.dumps({**model.metadata(), **metrics}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"model": str(out), "metrics": {**model.metadata(), **metrics}}, indent=2, ensure_ascii=False))
    return 0


def _feedback(args: argparse.Namespace) -> int:
    if args.scores:
        frame = feedback_from_scored_tsv(args.scores)
        ranking = RANKING_DOCKING_TRIAGE
    else:
        if not args.feedback:
            raise SystemExit("feedback requires --feedback or --scores")
        frame = load_feedback(args.feedback, database=args.database)
        ranking = args.ranking
    ranked = rank_feedback(frame, ranking)
    seeds, audit = build_next_seeds(
        ranked,
        args.top_k,
        explore_ratio=args.explore_ratio,
        similarity_cap=args.similarity_cap,
    )
    out = write_seeds(seeds, args.output)
    audit_path = Path(args.output).with_suffix(".audit.json")
    audit_path.write_text(
        json.dumps({"output": str(out), "ranking": ranking, **audit}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(
        {"n_ranked": int(len(ranked)), "n_seeds": len(seeds), "output": str(out), "audit": str(audit_path)},
        indent=2,
    ))
    return 0


def _select_oracle(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    model = load_model_from_config(cfg)
    if model is None:
        raise SystemExit("config does not enable a selectivity model (selectivity_model + selectivity_weight)")
    pool = load_pool(args.pool)
    result = select_oracle(model, pool, args.count, beta=args.beta, similarity_cap=args.similarity_cap)
    out = write_oracle_batch(result, args.output, pending_csv=args.pending)
    print(json.dumps({
        "n_pool": result["n_pool"],
        "n_in_domain": result["n_in_domain"],
        "n_ood": result["n_ood"],
        "n_selected": len(result["selected"]),
        "output": str(out),
        "pending": args.pending,
    }, indent=2))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    import pandas as pd

    input_path = Path(args.input)
    if input_path.suffix.lower() in (".smi", ".txt"):
        smiles = [line.split()[0].strip() for line in input_path.read_text(encoding="utf-8").splitlines()
                  if line.strip() and not line.startswith("#")]
    else:
        frame = pd.read_csv(input_path)
        if args.smiles_column not in frame.columns:
            raise ValueError(f"input CSV missing SMILES column: {args.smiles_column}")
        smiles = frame[args.smiles_column].astype(str).tolist()

    reference = None
    if args.reference:
        reference = load_pool(args.reference)

    metrics = evaluate_library(smiles, reference_smiles=reference)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="navgen")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", help="Generate and score candidate molecules")
    gen.add_argument("--config", required=True, help="JSON configuration path")
    gen.add_argument("--output", required=True, help="Directory for generated outputs")
    gen.set_defaults(func=_generate)

    prep = subparsers.add_parser("prepare-reinvent", help="Create REINVENT4 sampling and learning config stubs")
    prep.add_argument("--config", help="Project config file", default=None)
    prep.add_argument("--prior", required=True, help="Mol2Mol prior file path")
    prep.add_argument("--output", required=True, help="Output directory for reinforcement-learning config")
    prep.add_argument("--device", default="cpu")
    prep.add_argument("--num-smiles", type=int, default=1000)
    prep.add_argument("--python", default="python", help="Python executable in the scoring environment")
    prep.set_defaults(func=_prepare_reinvent)

    score = subparsers.add_parser("score-stdin", help="Score SMILES for REINVENT4 ExternalProcess")
    score.add_argument("--config", help="Project JSON configuration for the shared scorer")
    score.set_defaults(func=_score_stdin)

    score_file = subparsers.add_parser("score-reinvent-file", help="Apply the project scorer to a REINVENT4 CSV")
    score_file.add_argument("--config", required=True, help="Project JSON configuration")
    score_file.add_argument("--input", required=True, help="REINVENT4 CSV output")
    score_file.add_argument("--output", required=True, help="Scored CSV destination")
    score_file.add_argument("--smiles-column", default="SMILES")
    score_file.set_defaults(func=_score_reinvent_file)

    train_sel = subparsers.add_parser("train-selectivity", help="Train a random-forest paired-ΔΔG selectivity model")
    train_sel.add_argument("--data", required=True, help="Paired ΔΔG labels CSV")
    train_sel.add_argument("--output", required=True, help="Deployment model destination (.joblib)")
    train_sel.set_defaults(func=_train_selectivity)

    feedback = subparsers.add_parser("feedback", help="Rank scored molecules and write next-round seeds")
    feedback.add_argument("--config", help="Project config (context/state reference)", default=None)
    feedback.add_argument("--database", help="molecules.sqlite (reserved for candidate_id lookup)", default=None)
    feedback.add_argument("--feedback", help="Feedback CSV", default=None)
    feedback.add_argument("--scores", help="docking+打分 subtype_scores.tsv (docking triage)", default=None)
    feedback.add_argument("--output", required=True, help="Next-round seeds .smi destination")
    feedback.add_argument("--top-k", type=int, default=20)
    feedback.add_argument("--ranking", choices=[RANKING_DDG, RANKING_DOCKING_TRIAGE], default=RANKING_DDG)
    feedback.add_argument("--explore-ratio", type=float, default=0.2)
    feedback.add_argument("--similarity-cap", type=float, default=0.5)
    feedback.set_defaults(func=_feedback)

    oracle = subparsers.add_parser("select-oracle", help="Active-learning: pick molecules for the expensive oracle")
    oracle.add_argument("--config", required=True, help="Config with an enabled selectivity model")
    oracle.add_argument("--pool", required=True, help="Pool SMILES file (SMI or CSV)")
    oracle.add_argument("--output", required=True, help="Selected batch CSV destination")
    oracle.add_argument("--count", type=int, default=32)
    oracle.add_argument("--beta", type=float, default=1.0)
    oracle.add_argument("--similarity-cap", type=float, default=0.5)
    oracle.add_argument("--pending", default=None, help="Out-of-domain pending-queue CSV destination")
    oracle.set_defaults(func=_select_oracle)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a candidate library (distribution/scaffold/novelty)")
    evaluate.add_argument("--input", required=True, help="Candidate CSV/SMI")
    evaluate.add_argument("--smiles-column", default="smiles")
    evaluate.add_argument("--reference", default=None, help="Optional reference SMILES set for novelty")
    evaluate.add_argument("--output", required=True, help="Metrics JSON destination")
    evaluate.set_defaults(func=_evaluate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
