from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, QED
except ImportError:  # pragma: no cover
    Chem = None
    Crippen = Descriptors = QED = None

from .selectivity import fingerprint, murcko_scaffold, tanimoto

_MAX_RESERVOIR = 2000
_MAX_INTRA_PAIRS = 10_000
_REFERENCE_NOVELTY_CUTOFF = 0.4


def _parse_smiles(smiles: str) -> str | None:
    if Chem is None:
        return smiles.strip() if smiles.strip() else None
    mol = Chem.MolFromSmiles(smiles.strip())
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def evaluate_library(
    smiles_iter: Iterable[str],
    reference_smiles: Iterable[str] | None = None,
    passed_mask: Iterable[bool] | None = None,
    max_reservoir: int = _MAX_RESERVOIR,
    max_intra_pairs: int = _MAX_INTRA_PAIRS,
) -> dict[str, Any]:
    """计算候选库评价指标：validity / uniqueness / 分布 / 骨架数 / 批内相似性 / 参考新颖性。

    - validity = 可解析 / 原始
    - uniqueness = 唯一可解析 / 可解析
    - 分布用最多 max_reservoir 个分子的均匀蓄水池样本。
    """
    raw = list(smiles_iter)
    n_raw = len(raw)
    parsed = []
    passed_flags: list[bool] = []
    seen: set[str] = set()
    unique_smiles: list[str] = []
    for i, smiles in enumerate(raw):
        canon = _parse_smiles(str(smiles))
        if canon is None:
            continue
        parsed.append(canon)
        if passed_mask is not None:
            passed_flags.append(bool(list(passed_mask)[i]))
        if canon not in seen:
            seen.add(canon)
            unique_smiles.append(canon)

    n_parsed = len(parsed)
    n_unique = len(unique_smiles)
    metrics: dict[str, Any] = {
        "n_raw": n_raw,
        "n_valid": n_parsed,
        "n_unique": n_unique,
        "validity": n_parsed / n_raw if n_raw else 0.0,
        "uniqueness": n_unique / n_parsed if n_parsed else 0.0,
    }
    if passed_mask is not None and n_parsed:
        metrics["n_passed"] = int(sum(passed_flags))
        metrics["pass_rate_2d"] = metrics["n_passed"] / n_parsed

    metrics.update(_distribution(unique_smiles, max_reservoir))
    metrics.update(_scaffold_and_similarity(unique_smiles, max_reservoir, max_intra_pairs))
    if reference_smiles is not None:
        metrics.update(_reference_novelty(unique_smiles, reference_smiles, max_reservoir))
    else:
        metrics["reference_novelty"] = None
    return metrics


def _distribution(unique_smiles: list[str], max_reservoir: int) -> dict[str, Any]:
    sample = _reservoir_sample(unique_smiles, max_reservoir)
    if not sample or Chem is None:
        return {"distribution": None}
    mws, logps, tpsas, qeds = [], [], [], []
    for smiles in sample:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mws.append(Descriptors.MolWt(mol))
        logps.append(Crippen.MolLogP(mol))
        tpsas.append(Descriptors.TPSA(mol))
        qeds.append(QED.qed(mol))
    if not mws:
        return {"distribution": None}

    def _quantiles(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=float)
        return {
            "min": float(arr.min()),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
        }

    return {
        "distribution": {
            "n_sampled": len(mws),
            "mw": _quantiles(mws),
            "logp": _quantiles(logps),
            "tpsa": _quantiles(tpsas),
            "qed": _quantiles(qeds),
        }
    }


def _scaffold_and_similarity(unique_smiles: list[str], max_reservoir: int, max_intra_pairs: int) -> dict[str, Any]:
    scaffolds = {murcko_scaffold(s) for s in unique_smiles if murcko_scaffold(s)}
    sample = _reservoir_sample(unique_smiles, max_reservoir)
    intra = _mean_intra_tanimoto(sample, max_intra_pairs)
    return {
        "n_murcko_scaffolds": len(scaffolds),
        "intra_batch_tanimoto_mean": intra,
    }


def _reference_novelty(
    unique_smiles: list[str],
    reference_smiles: Iterable[str],
    max_reservoir: int,
) -> dict[str, Any]:
    ref_smiles = []
    for s in reference_smiles:
        canon = _parse_smiles(str(s))
        if canon is not None:
            ref_smiles.append(canon)
    if not ref_smiles:
        return {"reference_novelty": {"error": "no valid reference molecules"}}

    ref_fps = [fingerprint(s) for s in ref_smiles]
    sample = _reservoir_sample(unique_smiles, max_reservoir)
    nearest = []
    novel_count = 0
    for smiles in sample:
        fp = fingerprint(smiles)
        sim = max((tanimoto(fp, ref) for ref in ref_fps), default=0.0)
        nearest.append(sim)
        if sim < _REFERENCE_NOVELTY_CUTOFF:
            novel_count += 1

    ref_scaffolds = {murcko_scaffold(s) for s in ref_smiles if murcko_scaffold(s)}
    query_scaffolds = {murcko_scaffold(s) for s in sample if murcko_scaffold(s)}
    novel_scaffolds = query_scaffolds - ref_scaffolds

    return {
        "reference_novelty": {
            "n_reference": len(ref_smiles),
            "n_sampled": len(sample),
            "mean_nearest_tanimoto": float(np.mean(nearest)) if nearest else 0.0,
            "fraction_novel_lt_0_4": (novel_count / len(sample)) if sample else 0.0,
            "n_novel_scaffolds": len(novel_scaffolds),
        }
    }


def _reservoir_sample(items: list[str], k: int) -> list[str]:
    if len(items) <= k:
        return list(items)
    rng = random.Random(0)
    reservoir = items[:k]
    for i in range(k, len(items)):
        j = rng.randrange(i + 1)
        if j < k:
            reservoir[j] = items[i]
    return reservoir


def _mean_intra_tanimoto(smiles: list[str], max_pairs: int) -> float | None:
    if len(smiles) < 2 or Chem is None:
        return None
    fps = [fingerprint(s) for s in smiles]
    rng = random.Random(0)
    indices = list(range(len(fps)))
    pairs = [(i, j) for i in indices for j in indices if i < j]
    if len(pairs) > max_pairs:
        pairs = rng.sample(pairs, max_pairs)
    sims = [tanimoto(fps[i], fps[j]) for i, j in pairs]
    return float(np.mean(sims)) if sims else None
