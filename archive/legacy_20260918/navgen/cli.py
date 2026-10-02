import argparse
import json
import sys

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Nav1.8 reference-guided molecule generation")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate/import, filter, evaluate and export a batch")
    generate.add_argument("--config", required=True)
    generate.add_argument("--output", help="Override output directory")
    score = commands.add_parser("score-stdin", help="REINVENT4 ExternalProcess JSON scorer")
    score.add_argument("--config", required=True)
    fit = commands.add_parser("train-selectivity", help="Fit a scaffold-validated direct ddG regressor")
    fit.add_argument("--data", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--seed", type=int, default=20260918)
    feedback = commands.add_parser("feedback", help="Select next-round seeds from validated paired truth")
    feedback.add_argument("--config", required=True)
    feedback.add_argument("--database", required=True)
    feedback.add_argument("--feedback", required=True)
    feedback.add_argument("--output", required=True)
    feedback.add_argument("--top-k", type=int, default=20)
    feedback.add_argument("--exploration", type=float, default=0.2)
    feedback.add_argument("--max-similarity", type=float, default=0.85)
    feedback.add_argument("--ranking", choices=["ddg", "docking_triage"], default="ddg")
    prepare = commands.add_parser("prepare-reinvent", help="Write Mol2Mol sampling and RL TOML files")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--prior", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--device", default="cpu")
    prepare.add_argument("--python", help="Absolute scoring Python executable (navgen installed)")
    prepare.add_argument("--num-smiles", type=int, default=1000)
    active = commands.add_parser("select-oracle", help="Choose a diverse batch for expensive new labels")
    active.add_argument("--config", required=True)
    active.add_argument("--pool", required=True)
    active.add_argument("--output", required=True)
    active.add_argument("--count", type=int, default=32)
    active.add_argument("--beta", type=float, default=1.0)
    active.add_argument("--max-similarity", type=float, default=0.85)
    args = parser.parse_args(argv)
    try:
        if args.command == "train-selectivity":
            from .selectivity import train
            result = train(args.data, args.output, args.seed)
        else:
            config = load_config(args.config)
            if args.command == "generate":
                from pathlib import Path
                from .pipeline import run
                if args.output:
                    config["output_dir"] = str(Path(args.output).resolve())
                report = run(config)
                result = {k: report[k] for k in ("attempts", "accepted", "validity", "uniqueness", "stop_reason")}
                result["output_dir"] = config["output_dir"]
            elif args.command == "score-stdin":
                from .reinvent import external_scores
                result = external_scores(config, sys.stdin.read().splitlines())
            elif args.command == "prepare-reinvent":
                from .reinvent import prepare
                result = prepare(config, args.prior, args.output, args.device, args.python, args.num_smiles)
            elif args.command == "feedback":
                from .feedback import select_seeds
                report = select_seeds(args.database, args.feedback, args.output, config,
                                      args.top_k, args.exploration, args.max_similarity, args.ranking)
                result = {k: report[k] for k in ("selected", "requested", "eligible")}
            elif args.command == "select-oracle":
                from .active_learning import select
                result = select(config, args.pool, args.output, args.count, args.beta, args.max_similarity)
        print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError) as error:
        print(f"navgen: {error}", file=sys.stderr)
        return_code = 2
        raise SystemExit(return_code) from error
