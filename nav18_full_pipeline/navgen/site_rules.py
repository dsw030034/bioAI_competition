from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
except ImportError:  # pragma: no cover
    Chem = None
    Crippen = Descriptors = Lipinski = rdMolDescriptors = None


def load_site_profile(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path)
    if not profile_path.exists():
        raise FileNotFoundError(f"Site profile not found: {profile_path}")
    with profile_path.open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    if profile.get("site") != "Nav1.8 Site C":
        raise ValueError("Site profile must describe Nav1.8 Site C")
    return profile


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _fallback(smiles: str) -> dict[str, Any]:
    aromatic = smiles.count("c") >= 6
    carbonyl = "C(=O)" in smiles
    basic = "N" in smiles and "[nH]" not in smiles
    halogen = any(token in smiles for token in ("Cl", "Br", "F", "I"))
    score = 0.25 * carbonyl + 0.25 * aromatic + 0.15 * basic + 0.1 * halogen
    return {
        "site_c_score": round(_bounded(score), 4),
        "fenestration_score": round(_bounded(0.5 * aromatic + 0.5 * halogen), 4),
        "nav18_proxy_score": round(_bounded(score), 4),
        "nav15_counter_penalty": round(_bounded(0.5 * basic + 0.5 * aromatic), 4),
        "selectivity_proxy": round(_bounded(score - 0.35 * (0.5 * basic + 0.5 * aromatic)), 4),
        "site_c_features": {"carbonyl_anchor": carbonyl, "aromatic_hydrophobe": aromatic, "basic_center": basic, "halogen": halogen},
        "proxy_warning": "2D motif proxy; not docking, binding affinity, or experimental selectivity",
    }


def score_site_c(mol: Any, profile: dict[str, Any]) -> dict[str, Any]:
    if Chem is None or isinstance(mol, str):
        return _fallback(str(mol))
    if mol is None:
        raise ValueError("Molecule is None")

    carbonyl = mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[#6,#7,#8]"))
    amide = mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[NX3]"))
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    basic = any(atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0 for atom in mol.GetAtoms())
    halogen = any(atom.GetAtomicNum() in (9, 17, 35, 53) for atom in mol.GetAtoms())
    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rotors = Lipinski.NumRotatableBonds(mol)

    hydrophobic_fit = _bounded(1.0 - abs(logp - 3.0) / 4.0)
    size_fit = _bounded(1.0 - abs(mw - 360.0) / 260.0)
    fenestration = _bounded(0.45 * min(aromatic_rings / 2.0, 1.0) + 0.2 * halogen + 0.2 * hydrophobic_fit + 0.15 * _bounded(rotors / 8.0))
    site_c = _bounded(0.25 * carbonyl + 0.15 * amide + 0.2 * min(aromatic_rings / 2.0, 1.0) + 0.2 * hydrophobic_fit + 0.2 * size_fit)
    counter_penalty = _bounded(0.45 * basic + 0.3 * min(aromatic_rings / 2.0, 1.0) + 0.25 * hydrophobic_fit)
    nav18 = _bounded(0.65 * site_c + 0.35 * fenestration)
    selectivity = _bounded(nav18 - 0.35 * counter_penalty)
    return {
        "site_c_score": round(site_c, 4),
        "fenestration_score": round(fenestration, 4),
        "nav18_proxy_score": round(nav18, 4),
        "nav15_counter_penalty": round(counter_penalty, 4),
        "selectivity_proxy": round(selectivity, 4),
        "site_c_features": {
            "carbonyl_anchor": bool(carbonyl),
            "amide": bool(amide),
            "aromatic_rings": int(aromatic_rings),
            "basic_center": bool(basic),
            "halogen": bool(halogen),
            "logp": round(float(logp), 3),
            "mw": round(float(mw), 3),
            "rotatable_bonds": int(rotors),
        },
        "proxy_warning": "2D motif proxy; not docking, binding affinity, or experimental selectivity",
        "evidence_residues": profile.get("nav18_residues", []),
    }
