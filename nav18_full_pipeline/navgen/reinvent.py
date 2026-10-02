from __future__ import annotations

from pathlib import Path
import json


def _toml_string(value: str | Path) -> str:
    return json.dumps(str(value))


def prepare_reinvent_config(
    output_dir: str | Path,
    prior_path: str,
    device: str = "cpu",
    num_smiles: int = 1000,
    project_config: str | Path | None = None,
    python_path: str = "python",
) -> dict[str, str]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    sampling = output_root / "sampling.toml"
    learning = output_root / "staged_learning.toml"
    input_smiles = output_root / "mol2mol_inputs.smi"
    input_smiles.write_text("O=C(Nc1ccccc1)c1ccccc1\n", encoding="utf-8")
    prior = Path(prior_path).resolve()
    sampling_output = output_root / "sampling_mol2mol.csv"
    score_command = f"{python_path} -m navgen score-stdin"
    if project_config:
        score_command += f" --config {_toml_string(Path(project_config).resolve())}"

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
                "unique_molecules = true",
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
                'type = "IdenticalMurckoScaffold"',
                "bucket_size = 25",
                "minscore = 0.4",
                "",
                "[[stage]]",
                f"chkpt_file = {_toml_string(output_root / 'stage1.chkpt')}",
                'termination = "simple"',
                "max_score = 0.75",
                "min_steps = 5",
                "max_steps = 50",
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
    return {
        "sampling": str(sampling),
        "staged_learning": str(learning),
        "input_smiles": str(input_smiles),
        "project_score_command": score_command,
    }
