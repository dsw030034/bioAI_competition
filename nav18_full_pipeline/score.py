#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nav1.8 亚型选择性 docking + 打分（Nav1.8 vs Nav1.5 反筛）。

对每个 SMILES：
  1. 对接 Nav1.8（7WE4 链A，Site C 盒）→ score_nav18（kcal/mol，更负更强）
  2. 对接 Nav1.5（6LQA 链B，quinidine 盒）→ score_nav15
  3. 选择性 Δ = score_nav15 − score_nav18（>0 偏向 Nav1.8）
  4. 叠加 2D 成药性（QED / SA / 膜区间）
  5. 综合打分 + 排序，输出 TSV

诚实边界：Δ 是 docking 打分差值，不是实验结合自由能或亚型选择性。
跨受体的 Δ 只能作探索性排序，不能当药效结论；不假定统一的固定误差范围。

依赖：RDKit + Meeko + PDBFixer/OpenMM + AutoDock Vina。
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 复用同目录 scgd 的配体/受体准备与对接原语（macOS 已验证可用）
_BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(_BASE))
from scgd import prep, docking  # noqa: E402
from scgd.prep import AD4_TO_ELEMENT  # noqa: E402

from rdkit import Chem  # noqa: E402
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors  # noqa: E402
try:
    from rdkit.Contrib.SA_Score import sascorer  # noqa: E402
except Exception:  # pragma: no cover
    sascorer = None

# ---------------------------------------------------------------------------
# 受体与对接盒（Nav1.8 Site C / Nav1.5 中央腔，共晶配体定位）
# ---------------------------------------------------------------------------
_RECEPTORS_DIR = _BASE / 'receptors'
RECEPTORS = {
    'nav18': {
        'pdb': _RECEPTORS_DIR / '7WE4.pdb',
        'chain': 'A',
        'center': (137.204, 136.509, 127.538),
        'size': 24.0,
    },
    'nav15': {
        'pdb': _RECEPTORS_DIR / '6LQA.pdb',
        'chain': 'B',
        'center': (128.691875, 125.578875, 137.582083),
        'size': 24.0,
    },
}

STANDARD_AA = set(
    'ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER '
    'THR TRP TYR VAL'.split()
)
VINA = None  # CLI / environment / PATH / sibling Windows binary
EXHAUSTIVENESS = 8          # 演示值；需要更稳排序可提到 16
NUM_MODES = 9  # Save alternatives; original scoring still takes top pose


# ---------------------------------------------------------------------------
# 受体清理 + 准备
# ---------------------------------------------------------------------------
def clean_chain(pdb_path: Path, chain: str, out_path: Path) -> Path:
    """抽取单链、标准残基、无 HETATM、altloc 取 ' '/A 的蛋白 PDB。"""
    lines = pdb_path.read_text(encoding='utf-8', errors='ignore').splitlines()
    kept = []
    for ln in lines:
        if not ln.startswith('ATOM'):
            continue
        if ln[21:22] != chain:
            continue
        if ln[16:17] not in (' ', 'A'):
            continue
        if ln[17:20].strip() not in STANDARD_AA:
            continue
        kept.append(ln)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(kept) + '\n', encoding='utf-8')
    return out_path


def prepare_receptors(workdir: Path, resume=False) -> dict[str, Path]:
    from scgd.receptors import prepare_receptor
    return {name: prepare_receptor(cfg['pdb'], Path(workdir) / name, cfg['chain'],
                                    cfg['center'], cfg['size'], resume)
            for name, cfg in RECEPTORS.items()}


def dock_box(ligand_pdbqt: Path, receptor_pdbqt: Path, out_pdbqt: Path,
             center: tuple, size: float, **options) -> Path:
    from scgd.runtime import dock_job
    options.setdefault('vina', VINA)
    options.setdefault('exhaustiveness', EXHAUSTIVENESS)
    options.setdefault('num_modes', NUM_MODES)
    return dock_job(ligand_pdbqt, receptor_pdbqt, out_pdbqt, center, size, **options)


def _membrane_score(mw: float, logp: float, tpsa: float) -> float:
    def ds(v, lo, hi):
        if v <= lo:
            return max(0.0, min(1.0, (v - lo) / max(hi - lo, 1e-6)))
        if v >= hi:
            return max(0.0, min(1.0, (hi - v) / max(hi - lo, 1e-6)))
        return 1.0
    return (ds(mw, 250, 450) + ds(logp, 1.0, 3.5) + ds(tpsa, 30, 90)) / 3.0


def descriptors_2d(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'无效 SMILES：{smiles}')
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    qed = QED.qed(mol)
    sa = sascorer.calculateScore(mol) if sascorer else float('nan')
    hba = rdMolDescriptors.CalcNumHBA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return {
        'mw': mw, 'logp': logp, 'tpsa': tpsa, 'qed': qed, 'sa': sa,
        'hba': hba, 'hbd': hbd, 'rot': rot,
        'membrane': _membrane_score(mw, logp, tpsa),
    }


# ---------------------------------------------------------------------------
# 打分
# ---------------------------------------------------------------------------
@dataclass
class Scored:
    label: str
    smiles: str
    score_nav18: float
    score_nav15: float
    delta: float                     # score_nav15 − score_nav18，>0 偏 Nav1.8
    qed: float
    sa: float
    membrane: float
    combined: float

    def row(self) -> list[str]:
        return [
            self.label, self.smiles,
            f'{self.score_nav18:.2f}', f'{self.score_nav15:.2f}',
            f'{self.delta:+.2f}',
            f'{self.qed:.2f}', f'{self.sa:.2f}', f'{self.membrane:.2f}',
            f'{self.combined:.3f}',
        ]


def _aff_norm(score: float) -> float:
    """Vina kcal/mol → [0,1]：−5 记 0 分，−10 记 1 分。"""
    return float(np.clip((-score - 5.0) / 5.0, 0.0, 1.0))


def _sel_norm(delta: float) -> float:
    """Δ (kcal/mol) → [0,1]：每 3 kcal/mol 差分映射到满幅，居中于 0。"""
    return float(np.clip(delta / 3.0, -1.0, 1.0) * 0.5 + 0.5)


def _sa_norm(sa: float) -> float:
    return float(np.clip((5.0 - sa) / 3.0, 0.0, 1.0))


def combined_score(score_nav18: float, delta: float, qed: float, sa: float,
                   membrane: float,
                   w_aff=0.40, w_sel=0.30, w_qed=0.15, w_sa=0.07,
                   w_mem=0.08) -> float:
    """综合分 = 结合力 + 亚型选择性 + 成药性（权重可调）。

    权重默认偏重"结合力 + 选择性"两项目标；QED/SA/膜只作软约束。
    """
    return (w_aff * _aff_norm(score_nav18)
            + w_sel * _sel_norm(delta)
            + w_qed * qed
            + w_sa * _sa_norm(sa)
            + w_mem * membrane)


def score_one(label: str, smiles: str, receptors: dict[str, Path],
              tmp: Path, *, preparation_seed=20261002, num_confs=20,
              resume=False, **dock_options) -> Scored:
    import hashlib
    import math
    safe = hashlib.sha256(label.encode()).hexdigest()[:24]
    lig = prep.prep_ligand(smiles, tmp / f'lig_{safe}.pdbqt', seed=preparation_seed,
                          candidate_id=label, num_confs=num_confs, resume=resume)
    scores = {}
    for name in ('nav18', 'nav15'):
        pose = dock_box(lig, receptors[name], tmp / f'res_{safe}_{name}.pdbqt',
                        RECEPTORS[name]['center'], RECEPTORS[name]['size'],
                        resume=resume, **dock_options)
        scores[name] = docking.read_score(pose)
    s18, s15 = scores['nav18'], scores['nav15']
    if s18 is None or s15 is None or not all(math.isfinite(x) for x in (s18, s15)):
        raise RuntimeError(f'Nonfinite or missing Vina score: {label}')
    d2d = descriptors_2d(smiles)
    delta = s15 - s18
    comb = combined_score(s18, delta, d2d['qed'], d2d['sa'], d2d['membrane'])
    if not math.isfinite(comb):
        raise RuntimeError('Missing/nonfinite descriptor; combined score is not computed')
    return Scored(label, smiles, s18, s15, delta, d2d['qed'], d2d['sa'],
                  d2d['membrane'], comb)


HEADER = ['label', 'smiles', 'score_nav18', 'score_nav15', 'delta_sel',
          'qed', 'sa', 'membrane', 'combined']


def read_molecules(input_path: str, smiles_column: str = 'smiles',
                   label_column: str | None = None) -> list[tuple[str, str]]:
    from scgd.workflow import read_entries
    return [(r['candidate_id'], r['smiles']) for r in read_entries(input_path, smiles_column, label_column)]


def main() -> int:
    from scgd.workflow import main as run_workflow
    return run_workflow(sys.modules[__name__])


if __name__ == '__main__':
    raise SystemExit(main())
