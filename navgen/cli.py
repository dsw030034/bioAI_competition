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
    from .shared_scoring import ProjectScorer
    scorer = ProjectScorer(args.config)
    rewards = []
    for line in sys.stdin.read().splitlines():
        smiles = line.strip()
        try:
            parsed = json.loads(smiles)
            if isinstance(parsed, dict):
                smiles = str(parsed.get("smiles", parsed.get("SMILES", "")))
        except json.JSONDecodeError:
            pass
        rewards.append(scorer.score(smiles)["reward"])
    print(json.dumps({"version": 1, "payload": {"nav18_reward": rewards}}, allow_nan=False))
    return 0


def _score_reinvent_file(args: argparse.Namespace) -> int:
    import pandas as pd

    from .shared_scoring import ProjectScorer
    scorer = ProjectScorer(args.config)
    frame = pd.read_csv(args.input)
    smiles_column = args.smiles_column
    if smiles_column not in frame.columns:
        raise ValueError(f"Input file does not contain SMILES column: {smiles_column}")
    rows = []
    for value in frame[smiles_column].fillna(""):
        smiles = str(value).strip()
        metrics = scorer.score(smiles)
        rows.append({"smiles": smiles, "project_reward": metrics["reward"], **metrics})
    scored = pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)
    print(json.dumps({"input_rows": len(frame), "output": str(args.output)}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="navgen")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", help="Generate and score candidate molecules")
    gen.add_argument("--config", required=True, help="JSON configuration path")
    gen.add_argument("--output", required=True, help="Directory for generated outputs")
    gen.set_defaults(func=_generate)

    prep = subparsers.add_parser("prepare-reinvent", help="Create REINVENT4 sampling and learning config stubs")
    prep.add_argument("--config", help="Project config file", required=True)
    prep.add_argument("--prior", required=True, help="Mol2Mol prior file path")
    prep.add_argument("--output", required=True, help="Output directory for reinforcement-learning config")
    prep.add_argument("--device", default="cpu")
    prep.add_argument("--num-smiles", type=int, default=1000)
    prep.add_argument("--python", default=None, help="Python executable in the scoring environment")
    prep.set_defaults(func=_prepare_reinvent)

    score = subparsers.add_parser("score-stdin", help="Score SMILES for REINVENT4 ExternalProcess")
    score.add_argument("--config", required=True, help="Project JSON configuration for the shared scorer")
    score.set_defaults(func=_score_stdin)

    score_file = subparsers.add_parser("score-reinvent-file", help="Apply the project scorer to a REINVENT4 CSV")
    score_file.add_argument("--config", required=True, help="Project JSON configuration")
    score_file.add_argument("--input", required=True, help="REINVENT4 CSV output")
    score_file.add_argument("--output", required=True, help="Scored CSV destination")
    score_file.add_argument("--smiles-column", default="SMILES")
    score_file.set_defaults(func=_score_reinvent_file)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
