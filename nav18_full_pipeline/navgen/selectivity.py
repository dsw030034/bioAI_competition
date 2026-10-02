from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:  # pragma: no cover - fallback for minimal environments
    Chem = None
    AllChem = None
    DataStructs = None
    MurckoScaffold = None

# 配对 ΔΔG 标签约定（与 README 一致）：
#   ΔΔG = DG(Nav1.5) − DG(Nav1.8)，单位 kcal/mol，正值偏向 Nav1.8。
# species 仅允许 Homo sapiens；单个训练文件只允许一个 (state, protocol) 测定上下文；
# label_source 限 experiment / free_energy（禁用 docking，避免把静态对接差当选择性真值）。
REQUIRED_LABEL_COLUMNS = ["smiles", "ddg_kcal_mol", "species", "state", "protocol", "label_source"]
ALLOWED_SPECIES = {"Homo sapiens"}
ALLOWED_LABEL_SOURCES = {"experiment", "free_energy"}

_FP_N_BITS = 2048
_FP_RADIUS = 2


def fingerprint(smiles: str) -> np.ndarray:
    """Morgan/ECFP 指纹，用于回归特征与域外相似性。"""
    if Chem is None:
        raise RuntimeError("RDKit is required for selectivity fingerprints")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for fingerprint: {smiles}")
    arr = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(
        AllChem.GetMorganFingerprintAsBitVect(mol, _FP_RADIUS, nBits=_FP_N_BITS), arr
    )
    return arr.astype(np.float32)


def murcko_scaffold(smiles: str) -> str:
    """Murcko 骨架规范 SMILES，用于骨架留出划分与骨架数统计。"""
    if Chem is None:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold, canonical=True)


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.dot(a, b))
    union = float(np.sum(a) + np.sum(b) - inter)
    return inter / union if union > 0 else 0.0


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def load_paired_labels(path: str | Path) -> pd.DataFrame:
    """读取并校验配对 ΔΔG 标签 CSV。任何违反物种/上下文/来源约束都直接抛错。"""
    frame = pd.read_csv(path)
    missing = [col for col in REQUIRED_LABEL_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"paired labels file is missing columns: {', '.join(missing)}")
    species = set(frame["species"].dropna().astype(str).unique())
    if not species.issubset(ALLOWED_SPECIES):
        raise ValueError(f"species must be one of {sorted(ALLOWED_SPECIES)}, got {sorted(species)}")
    if "Homo sapiens" not in species:
        raise ValueError("paired labels require species = Homo sapiens")

    contexts = frame[["state", "protocol"]].dropna().drop_duplicates()
    if len(contexts) != 1:
        raise ValueError("a single paired-labels file must describe exactly one (state, protocol) context")

    sources = set(frame["label_source"].dropna().astype(str).unique())
    if not sources.issubset(ALLOWED_LABEL_SOURCES):
        raise ValueError(
            f"label_source must be one of {sorted(ALLOWED_LABEL_SOURCES)}, got {sorted(sources)}"
        )

    cleaned = frame.dropna(subset=["smiles", "ddg_kcal_mol", "species", "state", "protocol", "label_source"]).copy()
    cleaned["smiles"] = cleaned["smiles"].astype(str).str.strip()
    cleaned["ddg_kcal_mol"] = pd.to_numeric(cleaned["ddg_kcal_mol"], errors="coerce")
    cleaned = cleaned.dropna(subset=["ddg_kcal_mol"])
    if Chem is not None:
        valid = cleaned["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
        cleaned = cleaned[valid]
    if cleaned.empty:
        raise ValueError("paired labels file contains no valid labeled molecules")
    return cleaned.reset_index(drop=True)


def train_selectivity_model(
    data_path: str | Path,
    n_estimators: int = 200,
    random_state: int = 0,
) -> tuple["SelectivityModel", dict[str, Any]]:
    """训练随机森林预测配对 ΔΔG，按 Murcko 骨架留出验证，并与常数基线比较。

    返回 (deployment_model, metrics)。部署模型用全部输入重训；验证误差来自骨架留出，
    避免同一骨架同时出现在训练与验证两侧造成的乐观估计。
    """
    frame = load_paired_labels(data_path)
    smiles = frame["smiles"].tolist()
    y = frame["ddg_kcal_mol"].to_numpy(dtype=float)
    fps = np.vstack([fingerprint(s) for s in smiles])
    scaffolds = [murcko_scaffold(s) or s for s in smiles]

    n_unique = len(set(smiles))
    n_scaffolds = len(set(scaffolds))
    warnings: list[str] = []
    if n_unique < 12:
        warnings.append(f"only {n_unique} unique structures (< 12 smoke-test floor)")
    if n_scaffolds < 3:
        warnings.append(f"only {n_scaffolds} scaffolds (< 3 smoke-test floor)")

    metrics: dict[str, Any] = {
        "n_molecules": int(len(smiles)),
        "n_unique_structures": n_unique,
        "n_scaffolds": n_scaffolds,
        "warnings": warnings,
    }

    if n_scaffolds >= 2:
        logo = LeaveOneGroupOut()
        preds = np.full_like(y, np.nan, dtype=float)
        for train_idx, test_idx in logo.split(fps, y, groups=scaffolds):
            fold_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
            fold_model.fit(fps[train_idx], y[train_idx])
            preds[test_idx] = fold_model.predict(fps[test_idx])
        valid = ~np.isnan(preds)
        metrics["cv"] = {
            "strategy": "leave-one-scaffold-out",
            "mae_kcal_mol": float(mean_absolute_error(y[valid], preds[valid])),
            "rmse_kcal_mol": float(mean_squared_error(y[valid], preds[valid]) ** 0.5),
            "constant_baseline_mae_kcal_mol": float(mean_absolute_error(y[valid], np.full(valid.sum(), y.mean()))),
            "constant_baseline_rmse_kcal_mol": float(mean_squared_error(y[valid], np.full(valid.sum(), y.mean())) ** 0.5),
        }
    else:
        # 只有 1 个骨架时无法做骨架留出；仍重训部署模型，但不声称得到无交叉验证误差。
        metrics["cv"] = None

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(fps, y)

    state = str(frame["state"].iloc[0])
    protocol = str(frame["protocol"].iloc[0])
    label_source = sorted(sources)[0] if (sources := set(frame["label_source"].astype(str))) else None
    wrapped = SelectivityModel(
        model=model,
        state=state,
        protocol=protocol,
        label_source=label_source,
        train_smiles=smiles,
        train_fps=fps,
    )
    return wrapped, metrics


@dataclass
class SelectivityModel:
    """包装随机森林 + 训练上下文元数据，提供预测、域外检测与奖励映射。

    奖励采用 README 第 6 节的约定：`sigmoid((预测均值 − λ×树间标准差) / scale)`。
    树间标准差是未校准启发式，不能当置信区间。
    """

    model: RandomForestRegressor
    state: str
    protocol: str
    label_source: str | None
    train_smiles: list[str]
    train_fps: np.ndarray
    ood_similarity_min: float = 0.35
    uncertainty_penalty: float = 1.0
    reward_scale: float = 2.0

    def predict(self, smiles: str) -> tuple[float, float]:
        fp = fingerprint(smiles)
        tree_preds = np.array([tree.predict(fp.reshape(1, -1))[0] for tree in self.model.estimators_])
        return float(tree_preds.mean()), float(tree_preds.std(ddof=0))

    def max_similarity_to_train(self, smiles: str) -> float:
        fp = fingerprint(smiles)
        return max((tanimoto(fp, ref) for ref in self.train_fps), default=0.0)

    def is_ood(self, smiles: str) -> bool:
        return self.max_similarity_to_train(smiles) < self.ood_similarity_min

    def selectivity_reward(self, smiles: str) -> float:
        mean, std = self.predict(smiles)
        return float(_sigmoid((mean - self.uncertainty_penalty * std) / self.reward_scale))

    def matches_context(self, state: str | None, protocol: str | None) -> bool:
        if state is not None and state != self.state:
            return False
        if protocol is not None and protocol != self.protocol:
            return False
        return True

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out)
        return out

    @classmethod
    def load(cls, path: str | Path, state: str | None = None, protocol: str | None = None) -> "SelectivityModel":
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Selectivity model not found: {model_path}")
        # joblib 反序列化：只加载自己训练或可信来源的文件。
        obj = joblib.load(model_path)
        if not isinstance(obj, SelectivityModel):
            raise TypeError(f"{model_path} does not contain a SelectivityModel")
        if not obj.matches_context(state, protocol):
            raise ValueError(
                f"model context mismatch: model is (state={obj.state!r}, protocol={obj.protocol!r}) "
                f"but config requests (state={state!r}, protocol={protocol!r})"
            )
        return obj

    def metadata(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "protocol": self.protocol,
            "label_source": self.label_source,
            "n_train_molecules": len(self.train_smiles),
            "ood_similarity_min": self.ood_similarity_min,
            "uncertainty_penalty": self.uncertainty_penalty,
            "reward_scale": self.reward_scale,
        }


def load_model_from_config(cfg: dict[str, Any]) -> "SelectivityModel | None":
    """从项目配置加载已训练的选择性模型；未配置模型或权重为 0 时返回 None。"""
    filters = cfg.get("filter_config")
    if filters is None:
        return None
    if not filters.selectivity_model or not filters.selectivity_weight:
        return None
    model = SelectivityModel.load(
        filters.selectivity_model,
        state=cfg.get("state"),
        protocol=cfg.get("protocol"),
    )
    model.ood_similarity_min = filters.selectivity_ood_min
    model.uncertainty_penalty = filters.selectivity_uncertainty_penalty
    model.reward_scale = filters.selectivity_reward_scale
    return model
