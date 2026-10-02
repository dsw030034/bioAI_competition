from copy import deepcopy
import math
from pathlib import Path

from .io import read_json


DEFAULTS = {
    "seed": 20260918,
    "target": "Nav1.8", "counter_target": "Nav1.5", "site": "C",
    "species": "Homo sapiens", "state": "unspecified", "protocol": "unspecified",
    "seeds_file": None, "building_blocks_file": None, "output_dir": "outputs/run",
    "backend": "reaction", "input_smiles": None, "reference_file": None,
    "max_attempts": 10000, "max_candidates": 1000,
    "filters": {
        "similarity_min": 0.35, "qed_min": 0.35, "sa_max": 5.0,
        "mw": [180.0, 550.0], "logp": [-0.5, 5.0], "tpsa": [15.0, 120.0],
        "heavy_atoms_max": 40, "rotatable_bonds_max": 10, "pains": True,
        "exclude_seeds": True, "required_smarts": None,
    },
    "membrane": {"logp": [1.0, 3.5], "tpsa": [30.0, 90.0], "mw": [250.0, 450.0]},
    "weights": {"similarity": 0.30, "qed": 0.20, "sa": 0.15,
                "membrane": 0.25, "size": 0.10, "selectivity": 0.0},
    "selectivity": {"model": None, "ood_similarity_min": 0.35,
                    "uncertainty_penalty": 1.0, "reward_scale": 2.0},
    "conformers": {"enabled": False, "num_confs": 5, "max_molecules": 100,
                   "max_iterations": 300},
}


def load_config(path):
    path = Path(path).resolve()
    raw = read_json(path)
    config = deepcopy(DEFAULTS)
    for key, value in raw.items():
        if key not in config:
            raise ValueError(f"Unknown configuration key: {key}")
        if isinstance(config[key], dict):
            if not isinstance(value, dict) or set(value) - set(config[key]):
                raise ValueError(f"Unknown or invalid options in {key}")
            config[key].update(value)
        else:
            config[key] = value
    for key in ("seeds_file", "building_blocks_file", "output_dir", "input_smiles", "reference_file"):
        if config[key]:
            config[key] = str((path.parent / config[key]).resolve())
    if config["selectivity"]["model"]:
        config["selectivity"]["model"] = str((path.parent / config["selectivity"]["model"]).resolve())
    validate(config)
    return config


def validate(c):
    if c["backend"] not in {"reaction", "smiles", "hybrid"}:
        raise ValueError("backend must be reaction, smiles or hybrid")
    if not c["seeds_file"]:
        raise ValueError("seeds_file is required")
    if c["backend"] in {"smiles", "hybrid"} and not c["input_smiles"]:
        raise ValueError("This backend requires input_smiles")
    if c["backend"] in {"reaction", "hybrid"} and not c["building_blocks_file"]:
        raise ValueError("Reaction generation requires building_blocks_file")
    for group in ("filters", "membrane"):
        for key in ("mw", "logp", "tpsa"):
            bounds = c[group][key]
            if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or bounds[0] >= bounds[1]:
                raise ValueError(f"{group}.{key}: expected finite [low, high], low < high")
    for key in ("max_attempts", "max_candidates"):
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 <= c["seed"] < 2**31:
        raise ValueError("seed must be in [0, 2**31)")
    for key in ("similarity_min", "qed_min"):
        if not 0 <= c["filters"][key] <= 1:
            raise ValueError(f"filters.{key} must be in [0,1]")
    if not 1 <= c["filters"]["sa_max"] <= 10:
        raise ValueError("sa_max must be in [1,10]")
    weights = c["weights"]
    if not all(math.isfinite(v) and v >= 0 for v in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("weights must be finite, nonnegative, with positive sum")
    if weights["selectivity"] > 0 and not c["selectivity"]["model"]:
        raise ValueError("Positive selectivity weight requires a trained model; missing predictions are not zero")
    s = c["selectivity"]
    if not 0 <= s["ood_similarity_min"] <= 1 or not math.isfinite(s["reward_scale"]) or s["reward_scale"] <= 0:
        raise ValueError("Invalid selectivity range or reward_scale")
    if not math.isfinite(s["uncertainty_penalty"]) or s["uncertainty_penalty"] < 0:
        raise ValueError("uncertainty_penalty must be finite and nonnegative")
    for key in ("num_confs", "max_molecules", "max_iterations"):
        if not isinstance(c["conformers"][key], int) or c["conformers"][key] < 1:
            raise ValueError(f"conformers.{key} must be positive")

