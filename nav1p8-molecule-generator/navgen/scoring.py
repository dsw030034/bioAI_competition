from __future__ import annotations

import math
from typing import Any

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors
    from rdkit.Contrib.SA_Score import sascorer
except ImportError:  # pragma: no cover - fallback for minimal environments
    Chem = None
    Crippen = None
    Descriptors = None
    QED = None
    rdMolDescriptors = None
    sascorer = None

from .config import FilterConfig
from .generation import pairwise_similarity


def _double_sigmoid(value: float, low: float, high: float) -> float:
    if value <= low:
        return max(0.0, min(1.0, (value - low) / max(high - low, 1e-6)))
    if value >= high:
        return max(0.0, min(1.0, (high - value) / max(high - low, 1e-6)))
    return 1.0


def reinvent_native_metrics(metrics: dict[str, float], cfg: FilterConfig) -> dict[str, float]:
    """Mirror the REINVENT4 QED/MW/LogP component semantics locally."""
    mw_score = _double_sigmoid(metrics["mw"], cfg.mw_min, cfg.mw_max)
    logp_score = _double_sigmoid(metrics["logp"], cfg.clogp_min, cfg.clogp_max)
    qed_score = max(0.0, min(1.0, metrics["qed"]))
    geometric_mean = max(0.0, qed_score * mw_score * logp_score) ** (1.0 / 3.0)
    return {
        "reinvent_qed_score": float(qed_score),
        "reinvent_mw_score": float(mw_score),
        "reinvent_logp_score": float(logp_score),
        "reinvent_geometric_mean": float(geometric_mean),
    }


def _string_heuristic_metrics(smiles: str) -> dict[str, float]:
    carbon = smiles.count("C") + smiles.count("c")
    oxygen = smiles.count("O") + smiles.count("o")
    nitrogen = smiles.count("N") + smiles.count("n")
    heavy = len([ch for ch in smiles if ch.isupper()])
    mw = 12.0 * carbon + 16.0 * oxygen + 14.0 * nitrogen + 1.0 * (len(smiles) - heavy)
    logp = min(5.0, max(-0.5, (carbon - oxygen) / 6.0))
    tpsa = max(15.0, min(120.0, 20.0 + 10.0 * oxygen + 5.0 * nitrogen))
    qed = max(0.0, min(1.0, 0.5 + (mw / 500.0) - abs(logp - 2.0) / 10.0))
    sa = 2.5 + abs(logp - 2.0) + max(0.0, heavy - 20) * 0.05
    return {
        "mw": float(mw),
        "logp": float(logp),
        "tpsa": float(tpsa),
        "qed": float(qed),
        "sa": float(sa),
        "num_hba": float(oxygen + nitrogen),
        "num_hbd": float(nitrogen),
        "num_rotatable_bonds": float(max(0, heavy // 10)),
        "heavy_atoms": float(heavy),
    }


def descriptor_dict(mol: Chem.Mol | str) -> dict[str, float]:
    if mol is None:
        raise ValueError("Molecule is None")
    if Chem is None:
        smiles = str(mol)
        return _string_heuristic_metrics(smiles)
    return {
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "qed": float(QED.qed(mol)),
        "sa": float(sascorer.calculateScore(mol)),
        "num_hba": float(rdMolDescriptors.CalcNumHBA(mol)),
        "num_hbd": float(rdMolDescriptors.CalcNumHBD(mol)),
        "num_rotatable_bonds": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "heavy_atoms": float(mol.GetNumAtoms()),
    }


def required_smarts_present(mol: Chem.Mol | str, required_smarts: list[str]) -> bool:
    if not required_smarts:
        return True
    if Chem is None:
        text = str(mol)
        return any(smarts.replace(" ", "") in text for smarts in required_smarts)
    for smarts in required_smarts:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return True
    return False


def _string_similarity(smiles_a: str, smiles_b: str) -> float:
    a_set = set(smiles_a)
    b_set = set(smiles_b)
    if not a_set and not b_set:
        return 1.0
    return len(a_set & b_set) / max(len(a_set | b_set), 1)


def score_candidate(
    mol: Chem.Mol | str,
    seed_mol: Chem.Mol | str | None = None,
    cfg: FilterConfig | None = None,
    protein_word_metrics: dict[str, Any] | None = None,
    site_metrics: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    cfg = cfg or FilterConfig()
    metrics = descriptor_dict(mol)
    passed = True
    reasons: list[str] = []

    if metrics["mw"] < cfg.mw_min or metrics["mw"] > cfg.mw_max:
        passed = False
        reasons.append("mw")
    if metrics["logp"] < cfg.clogp_min or metrics["logp"] > cfg.clogp_max:
        passed = False
        reasons.append("logp")
    if metrics["tpsa"] < cfg.tpsa_min or metrics["tpsa"] > cfg.tpsa_max:
        passed = False
        reasons.append("tpsa")
    if metrics["qed"] < cfg.qed_min:
        passed = False
        reasons.append("qed")
    if metrics["sa"] > cfg.sa_max:
        passed = False
        reasons.append("sa")
    if not required_smarts_present(mol, cfg.required_smarts):
        passed = False
        reasons.append("required_smarts")

    if seed_mol is not None:
        seed_text = str(seed_mol)
        candidate_text = str(mol)
        similarity = _string_similarity(seed_text, candidate_text) if Chem is None else pairwise_similarity(Chem.MolToSmiles(seed_mol), Chem.MolToSmiles(mol))
        metrics["similarity_to_seed"] = float(similarity)
        if similarity < cfg.similarity_min:
            passed = False
            reasons.append("similarity")
    else:
        metrics["similarity_to_seed"] = None

    metrics["reward"] = max(0.0, float(metrics["qed"]))
    membrane_score = 1.0 if cfg.clogp_min <= metrics["logp"] <= cfg.clogp_max and cfg.tpsa_min <= metrics["tpsa"] <= cfg.tpsa_max else 0.5
    metrics["reward"] = 0.7 * metrics["qed"] + 0.3 * membrane_score
    protein_word_metrics = protein_word_metrics or {
        "protein_word_score": 0.0,
        "matched_rule_count": 0,
        "matched_rules": [],
        "protein_word_coverage": 0.0,
    }
    metrics.update(protein_word_metrics)
    metrics["reward"] = (
        (1.0 - cfg.protein_word_weight) * metrics["reward"]
        + cfg.protein_word_weight * metrics["protein_word_score"]
    )
    site_metrics = site_metrics or {
        "site_c_score": 0.0,
        "fenestration_score": 0.0,
        "nav18_proxy_score": 0.0,
        "nav15_counter_penalty": 0.0,
        "selectivity_proxy": 0.0,
        "site_c_features": {},
        "proxy_warning": "Site C proxy disabled",
    }
    metrics.update(site_metrics)
    metrics["reward"] = (
        (1.0 - cfg.site_c_weight - cfg.selectivity_proxy_weight) * metrics["reward"]
        + cfg.site_c_weight * metrics["nav18_proxy_score"]
        + cfg.selectivity_proxy_weight * metrics["selectivity_proxy"]
    )
    metrics.update(reinvent_native_metrics(metrics, cfg))
    if cfg.reinvent_metric_weight:
        metrics["reward"] = (
            (1.0 - cfg.reinvent_metric_weight) * metrics["reward"]
            + cfg.reinvent_metric_weight * metrics["reinvent_geometric_mean"]
        )
    metrics["passed"] = passed
    metrics["reasons"] = reasons
    if not passed:
        metrics["reward"] = 0.0
    return passed, metrics
