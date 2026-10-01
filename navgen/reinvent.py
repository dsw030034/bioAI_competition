from __future__ import annotations

from pathlib import Path
import json
import shlex
import sys

from .config import load_config
from .generation import load_seed_smiles


def _toml_string(value: str | Path) -> str:
    return json.dumps(str(value))


def prepare_reinvent_config(
    output_dir: str | Path,
    prior_path: str,
    device: str = "cpu",
    num_smiles: int = 1000,
    project_config: str | Path | None = None,
    python_path: str | None = None,
) -> dict[str, str]:
    if not project_config:
        raise ValueError("project_config is required: never substitute a demo seed")
    config = load_config(project_config)
    seeds = load_seed_smiles(config["seed_file"])
    output_root = Path(output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    sampling = output_root / "sampling.toml"
    learning = output_root / "staged_learning.toml"
    input_smiles = output_root / "mol2mol_inputs.smi"
    input_smiles.write_text("\n".join(seeds) + "\n", encoding="utf-8")
    snapshot = {k: v for k, v in config.items() if k != "filter_config"}
    snapshot["seed_file"] = str(input_smiles)
    score_config = output_root / "project_config.json"
    score_config.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    prior = Path(prior_path).resolve()
    sampling_output = output_root / "sampling_mol2mol.csv"
    executable = str(Path(python_path or sys.executable).resolve())
    entry = Path(__file__).resolve().parents[1] / "scripts" / "score_bridge.py"
    score_args = shlex.join([entry.as_posix(), "--config", score_config.as_posix()])
    score_command = shlex.join([executable, str(entry), "--config", str(score_config)])

    sampling.write_text(
        "\n".join(
            [
                'run_type = "sampling"',
                f'device = "{device}"',
                "",
                "[parameters]",
                f"model_file = {_toml_string(prior)}",
                f"smiles_file = {_toml_string(input_smiles)}",
                'sample_strategy = "multinomial"',
                "temperature = 1.0",
                f"output_file = {_toml_string(sampling_output)}",
                f"num_smiles = {num_smiles}",
                "unique_molecules = false",
                "randomize_smiles = true",
            ]
        ),
        encoding="utf-8",
    )
    learning.write_text(
        "\n".join(
            [
                'run_type = "staged_learning"',
                f'device = "{device}"',
                f'tb_logdir = {_toml_string(output_root / "tensorboard")}',
                "",
                "[parameters]",
                f"prior_file = {_toml_string(prior)}",
                f"agent_file = {_toml_string(prior)}",
                f"summary_csv_prefix = {_toml_string(output_root / 'rl_run')}",
                f"smiles_file = {_toml_string(input_smiles)}",
                "batch_size = 64",
                "randomize_smiles = true",
                "",
                "[learning_strategy]",
                'type = "dap"',
                "sigma = 64",
                "rate = 0.0001",
                "",
                "[diversity_filter]",
                'type = "PenalizeSameSmiles"',
                "bucket_size = 25",
                "minscore = 0.4",
                "penalty_multiplier = 0.5",
                "",
                "[[stage]]",
                f"chkpt_file = {_toml_string(output_root / 'stage1.chkpt')}",
                'termination = "simple"',
                "max_score = 0.95",
                "min_steps = 25",
                "max_steps = 100",
                "",
                "[stage.scoring]",
                'type = "geometric_mean"',
                "",
                "[[stage.scoring.component]]",
                "[stage.scoring.component.QED]",
                "[[stage.scoring.component.QED.endpoint]]",
                'name = "QED"',
                "weight = 1.0",
                "",
                "[[stage.scoring.component]]",
                "[stage.scoring.component.MolecularWeight]",
                "[[stage.scoring.component.MolecularWeight.endpoint]]",
                'name = "MW"',
                "weight = 1.0",
                'transform.type = "double_sigmoid"',
                "transform.low = 180.0",
                "transform.high = 550.0",
                "transform.coef_div = 550.0",
                "transform.coef_si = 20.0",
                "transform.coef_se = 20.0",
                "",
                "[[stage.scoring.component]]",
                "[stage.scoring.component.SlogP]",
                "[[stage.scoring.component.SlogP.endpoint]]",
                'name = "LogP"',
                "weight = 1.0",
                'transform.type = "double_sigmoid"',
                "transform.low = -0.5",
                "transform.high = 5.0",
                "transform.coef_div = 5.0",
                "transform.coef_si = 10.0",
                "transform.coef_se = 10.0",
            ]
        ),
        encoding="utf-8",
    )
    # One shared project scorer; do not double-count QED/MW/logP components.
    header = learning.read_text(encoding="utf-8").split("[[stage.scoring.component]]", 1)[0]
    learning.write_text(header + "\n".join([
        "[[stage.scoring.component]]",
        "[stage.scoring.component.ExternalProcess]",
        "[[stage.scoring.component.ExternalProcess.endpoint]]",
        'name = "A803467 constrained reward"', "weight = 1.0",
        f"params.executable = {_toml_string(executable)}",
        f"params.args = {_toml_string(score_args)}",
        'params.property = "nav18_reward"', "",
    ]), encoding="utf-8")
    return {
        "sampling": str(sampling),
        "staged_learning": str(learning),
        "input_smiles": str(input_smiles),
        "project_score_command": score_command,
    }
