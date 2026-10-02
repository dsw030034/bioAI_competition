"""Common, budgeted evaluation of local SMILES and 3D exploration SDF outputs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolAlign, rdShapeHelpers
from rdkit.Chem.Scaffolds import MurckoScaffold
from .generation import candidate_id
from .shared_scoring import ProjectScorer


def read_reference(path, expected_smiles):
    mols = list(Chem.SDMolSupplier(str(path), removeHs=False))
    if len(mols) != 1 or mols[0] is None:
        raise ValueError("Reference SDF must contain exactly one valid molecule")
    mol = mols[0]
    actual = Chem.MolToSmiles(Chem.RemoveHs(mol))
    expected = Chem.MolToSmiles(Chem.MolFromSmiles(expected_smiles))
    if actual != expected:
        raise ValueError(f"Reference identity differs from configured seed: {actual}")
    if not mol.GetNumConformers() or not mol.GetConformer().Is3D():
        raise ValueError("Reference must have explicit 3D coordinates")
    if not all(math.isfinite(v) for p in mol.GetConformer().GetPositions() for v in p):
        raise ValueError("Nonfinite reference coordinates")
    return mol


def inputs(path):
    path = Path(path)
    if path.suffix.lower() == ".sdf":
        for mol in Chem.SDMolSupplier(str(path), removeHs=False):
            yield (Chem.MolToSmiles(Chem.RemoveHs(mol)), mol) if mol else ("", None)
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = {k.lower(): k for k in reader.fieldnames or []}
            key = columns.get("smiles", columns.get("canonical_smiles"))
            if key is None:
                raise ValueError("CSV requires SMILES or canonical_smiles")
            for row in reader:
                yield row[key].strip(), None
    else:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.startswith("#"):
                yield line.split()[0] if line.strip() else "", None


def evaluate(config, input_path, output, method, budget=300, reference=None):
    if budget < 1:
        raise ValueError("budget must be positive")
    scorer = ProjectScorer(config)
    mode = scorer.config.get("generation_task", "local")
    if mode not in {"local", "scaffold_exploration"}:
        raise ValueError("Unknown generation_task")
    ref = read_reference(reference, scorer.seeds[0]) if reference else None
    if mode == "scaffold_exploration" and ref is None:
        raise ValueError("Exploration evaluation requires an identity-checked 3D reference")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rows, seen, accepted_scaffolds = [], set(), set()
    writer = Chem.SDWriter(str(out / "accepted_generated_poses.sdf"))
    try:
        import itertools
        for index, (smiles, mol3d) in enumerate(itertools.islice(inputs(input_path), budget)):
            score = scorer.score(smiles)
            mol = Chem.MolFromSmiles(smiles) if score["valid"] else None
            canonical = Chem.MolToSmiles(mol) if mol else ""
            duplicate = bool(canonical and canonical in seen)
            if canonical:
                seen.add(canonical)
            row = {"input_index": index, "method": method, "smiles": canonical,
                   "raw_sample_index": mol3d.GetProp("raw_sample_index") if mol3d is not None and mol3d.HasProp("raw_sample_index") else None,
                   "candidate_id": candidate_id(canonical) if canonical else None,
                   **score, "duplicate": duplicate, "shape_tanimoto": None,
                   "geometry_status": "not_evaluated", "pharmacophore_status": "not_evaluated",
                   "electrostatics_status": "not_evaluated"}
            # Shape is a screen, not binding or pharmacophore validation. Preserve raw pose.
            if ref is not None and mol3d is not None and mol3d.GetNumConformers() and mol3d.GetConformer().Is3D():
                try:
                    probe = Chem.RemoveHs(Chem.Mol(mol3d))
                    target = Chem.RemoveHs(Chem.Mol(ref))
                    rdMolAlign.GetCrippenO3A(probe, target).Align()
                    row["shape_tanimoto"] = 1.0 - rdShapeHelpers.ShapeTanimotoDist(probe, target)
                    row["geometry_status"] = "shape_only_unrelaxed"
                except (RuntimeError, ValueError):
                    row["geometry_status"] = "alignment_failed"
            row["accepted_2d"] = bool(score["passed"] and not duplicate)
            # Exploration remains a review queue until geometry/ESP/pharmacophore QA.
            row["ready_for_docking"] = bool(row["accepted_2d"] and mode == "local")
            if row["accepted_2d"]:
                accepted_scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
                if mol3d:
                    mol3d.SetProp("candidate_id", row["candidate_id"])
                    writer.write(mol3d)
            rows.append(row)
    finally:
        writer.close()
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["candidate_id", "smiles"]
    for name, selected in (("all_scored.csv", rows), ("candidates.csv", [r for r in rows if r["accepted_2d"]])):
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            csvwriter = csv.DictWriter(handle, fieldnames=fields)
            csvwriter.writeheader()
            csvwriter.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in selected)
    summary = {"method": method, "task": mode, "raw_rows_evaluated": len(rows),
               "valid": sum(r["valid"] for r in rows), "unique": len(seen),
               "accepted_2d": sum(r["accepted_2d"] for r in rows), "scaffolds": len(accepted_scaffolds),
               "budget": budget, "input_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
               "config": {k: v for k, v in scorer.config.items() if k != "filter_config"},
               "reference_sha256": hashlib.sha256(Path(reference).read_bytes()).hexdigest() if reference else None,
               "validity_scope": "input-file rows only; upstream filtering must be reported separately",
               "physical_validation": "not performed; no activity/selectivity conclusions"}
    import numpy as np
    summary["validity_in_input"] = summary["valid"] / len(rows) if rows else None
    summary["uniqueness_in_valid_input"] = len(seen) / summary["valid"] if summary["valid"] else None
    summary["accepted_distributions"] = {}
    for field in ("mw", "logp", "tpsa", "qed", "sa", "similarity_to_seed", "shape_tanimoto"):
        values = [r[field] for r in rows if r["accepted_2d"] and r.get(field) is not None]
        summary["accepted_distributions"][field] = dict(zip(
            ("min", "p25", "median", "p75", "max"),
            np.quantile(values, [0, .25, .5, .75, 1]).tolist())) if values else None
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--reference")
    a = parser.parse_args()
    print(json.dumps(evaluate(a.config, a.input, a.output, a.method, a.budget, a.reference), indent=2))


if __name__ == "__main__":
    main()
