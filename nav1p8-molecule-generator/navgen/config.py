from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FilterConfig:
    similarity_min: float = 0.35
    qed_min: float = 0.35
    sa_max: float = 5.0
    mw_min: float = 180.0
    mw_max: float = 550.0
    clogp_min: float = -0.5
    clogp_max: float = 5.0
    tpsa_min: float = 15.0
    tpsa_max: float = 120.0
    required_smarts: list[str] = field(default_factory=lambda: ["C(=O)N"])
    num_confs: int = 5
    max_candidates: int = 200
    protein_word_weight: float = 0.2
    site_c_weight: float = 0.0
    selectivity_proxy_weight: float = 0.0
    reinvent_metric_weight: float = 0.0

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "FilterConfig":
        filters = mapping.get("filters", {})
        config = cls()
        for name, value in filters.items():
            if hasattr(config, name):
                setattr(config, name, value)
        if "required_smarts" in filters and isinstance(filters["required_smarts"], str):
            config.required_smarts = [filters["required_smarts"]]
        return config

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    if "filters" not in cfg:
        cfg["filters"] = {}
    cfg["filters"] = {**FilterConfig().as_dict(), **cfg.get("filters", {})}
    cfg["seed_file"] = str((config_path.parent / cfg.get("seed_file", "data/seeds.smi")).resolve())
    cfg["output_dir"] = str((config_path.parent / cfg.get("output_dir", "outputs")).resolve())
    if cfg.get("protein_sequence_file"):
        cfg["protein_sequence_file"] = str((config_path.parent / cfg["protein_sequence_file"]).resolve())
    if cfg.get("site_profile_file"):
        cfg["site_profile_file"] = str((config_path.parent / cfg["site_profile_file"]).resolve())
    cfg["filter_config"] = FilterConfig.from_mapping(cfg)
    return cfg
