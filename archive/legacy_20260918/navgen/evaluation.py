"""Bounded-memory distributions and novelty, with explicit denominators."""

import random

import numpy as np
from rdkit import DataStructs

from .chemistry import canonical, fingerprint, molecule, scaffold
from .io import read_smiles


def distribution(values):
    if not values:
        return None
    hist, edges = np.histogram(values, bins=10)
    return {"n": len(values), "quantiles": dict(zip(
        ["min", "p10", "p25", "median", "p75", "p90", "max"],
        [float(x) for x in np.quantile(values, [0, .1, .25, .5, .75, .9, 1])])),
        "histogram": {"counts": hist.tolist(), "edges": edges.tolist()}}


def evaluate(connection, stats, config):
    import json
    rng = random.Random(config["seed"])
    sampled = []
    sample_size = 2000
    # Reservoir sampling avoids first-generated / highest-score bias.
    for i, (payload,) in enumerate(connection.execute("SELECT payload FROM molecules WHERE passed=1 ORDER BY candidate_id")):
        record = json.loads(payload)
        if len(sampled) < sample_size:
            sampled.append(record)
        else:
            j = rng.randrange(i + 1)
            if j < sample_size:
                sampled[j] = record
    n_accepted = connection.execute("SELECT COUNT(*) FROM molecules WHERE passed=1").fetchone()[0]
    n_scaffolds = connection.execute("SELECT COUNT(DISTINCT scaffold) FROM molecules WHERE passed=1").fetchone()[0]
    fps = [fingerprint(molecule(r["smiles"])) for r in sampled]
    pairs = []
    if len(fps) > 1:
        for _ in range(min(10000, len(fps) * (len(fps) - 1) // 2)):
            i, j = rng.sample(range(len(fps)), 2)
            pairs.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    report = {
        **stats, "accepted": n_accepted,
        "validity": stats["valid"] / stats["attempts"] if stats["attempts"] else None,
        "uniqueness": stats["unique_valid"] / stats["valid"] if stats["valid"] else None,
        "acceptance_per_unique_valid": n_accepted / stats["unique_valid"] if stats["unique_valid"] else None,
        "accepted_scaffold_count": n_scaffolds, "distribution_sample_size": len(sampled),
        "distribution_sampling": "uniform reservoir, at most 2000 accepted molecules",
        "distributions": {k: distribution([r[k] for r in sampled])
                          for k in ("mw", "logp", "tpsa", "qed", "sa", "similarity", "membrane_proxy")},
        "pairwise_tanimoto": distribution(pairs),
        "pair_sampling": "uniform distinct-index pairs sampled with replacement, at most 10000",
        "novelty": None,
        "synthesis": {"sa_distribution": distribution([r["sa"] for r in sampled]),
                      "retrosynthesis_pass_rate": None, "routes_are_template_hypotheses": True},
        "selectivity": {"model_enabled": bool(config["selectivity"]["model"]),
                        "experimental_validation": False, "paired_error_correlation": None},
        "physical_membrane_metrics": {"burial_kl": None, "fenestration_pass_rate": None},
    }
    if config["reference_file"]:
        references = set()
        for raw in read_smiles(config["reference_file"]):
            smi = canonical(raw)
            if smi is None:
                raise ValueError(f"Invalid reference SMILES: {raw}")
            references.add(smi)
        if not references:
            raise ValueError("Empty novelty reference set")
        ref_mols = [molecule(s) for s in sorted(references)]
        ref_scaffolds = {scaffold(m) for m in ref_mols}
        ref_fps = [fingerprint(m) for m in ref_mols]
        report["novelty"] = {
            "reference_count": len(references), "evaluated_count": len(sampled),
            "molecule_novel_fraction": sum(r["smiles"] not in references for r in sampled) / len(sampled) if sampled else None,
            "scaffold_novel_fraction": sum(r["scaffold"] not in ref_scaffolds for r in sampled) / len(sampled) if sampled else None,
            "nearest_reference_tanimoto": distribution([max(DataStructs.BulkTanimotoSimilarity(fp, ref_fps)) for fp in fps]),
        }
    return report

