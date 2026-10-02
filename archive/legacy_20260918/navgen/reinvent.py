"""REINVENT4 Mol2Mol sampling + DAP RL configuration and ExternalProcess bridge.

Interface checked against MolecularAI/REINVENT4 commit
ee0d56f4a07472bbb622cd0858184d06f11bff5d (2026-09-18).
"""

import json
from pathlib import Path
import shlex
import sys

from .generation import read_seeds
from .io import write_json
from .scoring import Scorer


def external_scores(config, smiles):
    scorer = Scorer(config, read_seeds(config["seeds_file"]))
    results = [scorer.score(s.strip()) for s in smiles]
    return {"version": 1, "payload": {
        "nav18_reward": [r["reward"] for r in results],
        "valid": [r["valid"] for r in results],
        "passed": [r["passed"] for r in results],
        "selectivity_status": [r.get("selectivity_status", "not_available") for r in results],
    }}


def prepare(config, prior, output, device="cpu", python_executable=None, num_smiles=1000):
    prior = Path(prior).resolve()
    if not prior.is_file():
        raise FileNotFoundError(f"Supply a real REINVENT4 Mol2Mol prior: {prior}")
    if num_smiles < 1:
        raise ValueError("num_smiles must be positive")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seeds = output / "seeds.smi"
    seeds.write_text("\n".join(read_seeds(config["seeds_file"])) + "\n", encoding="utf-8")
    score_config = output / "score_config.json"
    write_json(score_config, config)
    # JSON quoted strings are also valid TOML basic strings; forward slashes
    # and shlex.join keep subprocess arguments safe on both Windows and Linux.
    def quote(value):
        return json.dumps(str(value).replace("\\", "/"), ensure_ascii=False)
    sampling = f'''run_type = "sampling"
device = {quote(device)}
json_out_config = {quote(output / "sampling_resolved.json")}

[parameters]
model_file = {quote(prior)}
smiles_file = {quote(seeds)}
output_file = {quote(output / "sampling.csv")}
sample_strategy = "multinomial"
temperature = 1.0
num_smiles = {num_smiles}
unique_molecules = false
randomize_smiles = true
'''
    args = shlex.join(["-m", "navgen", "score-stdin", "--config", score_config.as_posix()])
    rl = f'''run_type = "staged_learning"
device = {quote(device)}
json_out_config = {quote(output / "rl_resolved.json")}
tb_logdir = {quote(output / "tensorboard")}

[parameters]
summary_csv_prefix = {quote(output / "rl")}
prior_file = {quote(prior)}
agent_file = {quote(prior)}
smiles_file = {quote(seeds)}
batch_size = 64
sample_strategy = "multinomial"
randomize_smiles = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001

[diversity_filter]
type = "IdenticalMurckoScaffold"
bucket_size = 25
minscore = 0.4

[[stage]]
chkpt_file = {quote(output / "nav18_agent.chkpt")}
termination = "simple"
max_score = 0.95
min_steps = 25
max_steps = 100

[stage.scoring]
type = "arithmetic_mean"

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "Nav1.8 constrained reward"
weight = 1.0
params.executable = {quote(python_executable or sys.executable)}
params.args = {json.dumps(args, ensure_ascii=False)}
params.property = "nav18_reward"
'''
    (output / "sampling.toml").write_text(sampling, encoding="utf-8")
    (output / "staged_learning.toml").write_text(rl, encoding="utf-8")
    return {"sampling": str(output / "sampling.toml"), "rl": str(output / "staged_learning.toml"),
            "selectivity_enabled": bool(config["selectivity"]["model"]),
            "num_smiles_note": "REINVENT Mol2Mol samples this many per input seed"}

