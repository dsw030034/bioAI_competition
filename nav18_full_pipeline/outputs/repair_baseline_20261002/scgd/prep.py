# -*- coding: utf-8 -*-
"""分子准备：Meeko 配体 → PDBQT；RDKit 自定义 AD4 原子类型受体 → PDBQT。

不依赖 AutoDockTools：受体 PDBQT 由 RDKit 生成 3D 构象 + Gasteiger 电荷 +
AD4 原子类型（C→'A'/'C'、N→'NA'/'N'、O→'OA'、S→'S'、H→'HD'），
并裁到对接盒中心半径 R 内。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdPartialCharges
from meeko import MoleculePreparation, PDBQTWriterLegacy

from . import config

# AD4 原子类型 → 元素（供接触分析 / 重原子判定复用）
AD4_TO_ELEMENT = {
    'A': 'C', 'C': 'C', 'N': 'N', 'NA': 'N', 'O': 'O', 'OA': 'O',
    'S': 'S', 'Cl': 'Cl', 'F': 'F', 'Br': 'Br', 'I': 'I', 'P': 'P',
    'HD': 'H', 'HS': 'H',
}


# ---------------------------------------------------------------------------
# 配体准备（Meeko）
# ---------------------------------------------------------------------------
def prep_ligand(smiles: str, out_pdbqt: Path | None = None,
                seed: int = 42) -> Path:
    """SMILES → 3D → Meeko PDBQT。返回 PDBQT 路径。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'无效 SMILES：{smiles}')
    nheavy = mol.GetNumAtoms()
    mol = Chem.AddHs(mol)
    embedded = False
    for s in range(seed, seed + 38):
        if AllChem.EmbedMolecule(mol, randomSeed=s) == 0:
            embedded = True
            break
    if not embedded:
        raise RuntimeError(f'EmbedMolecule 失败：{smiles}')
    AllChem.MMFFOptimizeMolecule(mol)

    setups = MoleculePreparation().prepare(mol)
    if isinstance(setups, (list, tuple)):
        setups = setups[0]
    pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups)
    if not ok:
        raise RuntimeError(f'Meeko 写 PDBQT 失败：{err}')

    out_pdbqt = Path(out_pdbqt) if out_pdbqt else Path(f'{_slug(smiles)}.pdbqt')
    out_pdbqt.write_text(pdbqt)
    return out_pdbqt


def _slug(smiles: str, n: int = 8) -> str:
    import hashlib
    return hashlib.md5(smiles.encode()).hexdigest()[:n]


# ---------------------------------------------------------------------------
# 受体准备（RDKit 自定义 AD4）
# ---------------------------------------------------------------------------
def _ad_type(atom) -> str:
    an = atom.GetAtomicNum()
    if an == 6:
        return 'A' if atom.GetIsAromatic() else 'C'
    if an == 7:
        if atom.GetIsAromatic() or atom.GetHybridization() == Chem.HybridizationType.SP2:
            return 'NA'
        return 'N'
    if an == 8:
        return 'OA'
    if an == 16:
        return 'S'
    return atom.GetSymbol()


def prep_receptor(protein_pdb: Path, out_pdbqt: Path | None = None,
                  center: tuple[float, float, float] | None = None,
                  radius: float = 28.0) -> Path:
    """蛋白 PDB → 受体 PDBQT（AddHs + Gasteiger + AD4 类型，裁到中心 R 内）。

    center 默认用 BOX_CENTER；合并非极性 H 电荷到碳（AD4 惯例）。
    """
    center = np.array(config.BOX_CENTER if center is None else center, dtype=float)
    mol = Chem.MolFromPDBFile(str(protein_pdb), removeHs=True, sanitize=True)
    if mol is None:
        raise ValueError(f'RDKit 无法解析蛋白：{protein_pdb}')
    molH = Chem.AddHs(mol, addCoords=True)
    rdPartialCharges.ComputeGasteigerCharges(molH)

    charges = {a.GetIdx(): float(a.GetDoubleProp('_GasteigerCharge'))
               for a in molH.GetAtoms()}
    for a in molH.GetAtoms():
        if a.GetAtomicNum() == 1:
            nbrs = list(a.GetNeighbors())
            if nbrs and nbrs[0].GetAtomicNum() == 6:
                charges[nbrs[0].GetIdx()] += charges[a.GetIdx()]
                charges[a.GetIdx()] = None     # 丢弃该 H

    conf = molH.GetConformer()
    serial = 0
    out = []
    for atom in molH.GetAtoms():
        q = charges[atom.GetIdx()]
        if q is None:
            continue
        p = conf.GetAtomPosition(atom.GetIdx())
        if np.linalg.norm([p.x - center[0], p.y - center[1], p.z - center[2]]) > radius:
            continue
        an = atom.GetAtomicNum()
        if an == 1:
            atype, name = 'HD', 'HD'
        else:
            atype = _ad_type(atom)
            ri = atom.GetPDBResidueInfo()
            name = ri.GetName().strip() if ri else atom.GetSymbol()
        ri = atom.GetPDBResidueInfo()
        resname = ri.GetResidueName().strip() if ri else 'UNK'
        chain = ri.GetChainId().strip() if ri else 'A'
        resnum = ri.GetResidueNumber() if ri else 0
        serial += 1
        out.append(
            f"ATOM  {serial:5d} {name:<4s} {resname:>3s} {chain:1s}{resnum:4d}    "
            f"{p.x:8.3f}{p.y:8.3f}{p.z:8.3f}{1.00:6.2f}{0.00:6.2f}    {q:6.3f} {atype:<2s}"
        )

    out_pdbqt = Path(out_pdbqt) if out_pdbqt else protein_pdb.with_suffix('.pdbqt')
    out_pdbqt.write_text('\n'.join(out) + '\n')
    return out_pdbqt
