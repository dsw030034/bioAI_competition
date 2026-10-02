"""Direct paired-difference regression; no subtraction of docking predictions.

Sign convention: ddg = DG(Nav1.5) - DG(Nav1.8), kcal/mol; larger favors Nav1.8.
Random-forest tree spread is a heuristic, NOT a calibrated confidence interval.
"""

import csv
import math
from pathlib import Path

import joblib
import numpy as np
from rdkit import DataStructs
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from .chemistry import canonical, fingerprint, molecule, scaffold
from .io import file_hash, write_json

CONTEXT = ("species", "state", "protocol")
ALLOWED_LABEL_SOURCES = {"experiment", "free_energy"}


def features(smiles):
    return np.asarray([np.asarray(fingerprint(molecule(s)), dtype=np.float32) for s in smiles])


def train(data_file, output, seed=20260918):
    rows = []
    seen = set()
    contexts = set()
    with open(data_file, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"smiles", "ddg_kcal_mol", "label_source", *CONTEXT}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Training CSV requires {sorted(required)}")
        for row in reader:
            smi = canonical(row["smiles"])
            if not smi or "." in smi or smi in seen:
                raise ValueError("Invalid, disconnected or duplicate training structure; aggregate matched replicates first")
            if row["label_source"] not in ALLOWED_LABEL_SOURCES:
                raise ValueError("Only experiment/free_energy labels are supported; docking is not selectivity truth")
            context = tuple(row[k].strip() for k in CONTEXT)
            if context[0] != "Homo sapiens" or any(v in {"", "unspecified"} for v in context):
                raise ValueError("Require human data with explicit state and matched assay protocol")
            value = float(row["ddg_kcal_mol"])
            if not math.isfinite(value):
                raise ValueError("Nonfinite label")
            contexts.add(context)
            seen.add(smi)
            rows.append((smi, value))
    if len(contexts) != 1:
        raise ValueError("Train one model per species/state/protocol; mixed contexts are not comparable")
    if len(rows) < 12:
        raise ValueError("At least 12 unique molecules are required even for a smoke-test model")
    smiles, y = zip(*rows)
    y = np.asarray(y)
    groups = [scaffold(molecule(s)) for s in smiles]
    if len(set(groups)) < 3:
        raise ValueError("At least 3 distinct Murcko scaffolds required for held-out validation")
    x = features(smiles)
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(x, y, groups))
    if len(train_idx) < 4 or len(test_idx) < 2:
        raise ValueError("Scaffold split too small; collect more diverse paired data")
    model = RandomForestRegressor(n_estimators=128, min_samples_leaf=2, random_state=seed, n_jobs=1)
    model.fit(x[train_idx], y[train_idx])
    prediction = model.predict(x[test_idx])
    report = {
        "n_train": len(train_idx), "n_test": len(test_idx),
        "train_scaffolds": sorted({groups[i] for i in train_idx}),
        "test_scaffolds": sorted({groups[i] for i in test_idx}),
        "mae": float(mean_absolute_error(y[test_idx], prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y[test_idx], prediction))),
        "constant_baseline_mae": float(np.mean(np.abs(y[test_idx] - y[train_idx].mean()))),
        "test_predictions": [{"smiles": smiles[i], "observed": float(y[i]), "predicted": float(p)}
                             for i, p in zip(test_idx, prediction)],
        "context": dict(zip(CONTEXT, next(iter(contexts)))),
        "label_definition": "DG_Nav1.5 - DG_Nav1.8 (kcal/mol); positive favors Nav1.8",
        "uncertainty": "uncalibrated tree standard deviation",
        "data_sha256": file_hash(data_file), "seed": seed,
        "deployment_fit": "all rows after independent scaffold holdout evaluation",
    }
    model.fit(x, y)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".metrics.json").exists():
        raise FileExistsError(f"Refusing to overwrite model: {output}")
    joblib.dump({"format_version": 1, "model": model, "smiles": list(smiles),
                 "context": report["context"], "target": "Nav1.8", "counter_target": "Nav1.5",
                 "site": "C", "report": report}, output)
    write_json(output.with_suffix(".metrics.json"), report)
    return report


class SelectivityModel:
    def __init__(self, path, config):
        # joblib/pickle is executable: only load artifacts you created or trust.
        data = joblib.load(path)
        if data["format_version"] != 1:
            raise ValueError("Unsupported model format")
        for key in (*CONTEXT,):
            if data["context"][key] != config[key]:
                raise ValueError(f"Selectivity model context mismatch: {key}")
        for key in ("target", "counter_target", "site"):
            if data[key] != config[key]:
                raise ValueError(f"Selectivity model mismatch: {key}")
        self.model = data["model"]
        self.reference_fps = [fingerprint(molecule(s)) for s in data["smiles"]]

    def predict(self, smiles):
        x = features([smiles])
        predictions = np.asarray([tree.predict(x)[0] for tree in self.model.estimators_])
        similarity = max(DataStructs.BulkTanimotoSimilarity(fingerprint(molecule(smiles)), self.reference_fps))
        return {"ddg_pred": float(predictions.mean()), "ddg_std": float(predictions.std()),
                "training_similarity": float(similarity), "uncertainty_calibrated": False}

