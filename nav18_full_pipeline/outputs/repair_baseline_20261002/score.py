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
Vina 打分噪声约 ±0.5 kcal/mol，跨受体的 Δ 只能作探索性排序，不能当药效结论。

依赖：RDKit + Meeko + AutoDock Vina（macOS /opt/homebrew/bin/vina）。
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
VINA = '/opt/homebrew/bin/vina'
EXHAUSTIVENESS = 8          # 演示值；需要更稳排序可提到 16
NUM_MODES = 1


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


def prepare_receptors(workdir: Path) -> dict[str, Path]:
    """清理 + 准备两个亚型的受体 PDBQT，返回 {name: pdbqt_path}。"""
    out = {}
    for name, cfg in RECEPTORS.items():
        clean = clean_chain(cfg['pdb'], cfg['chain'],
                            workdir / f'{name}_clean.pdb')
        pdbqt = prep.prep_receptor(clean, workdir / f'{name}.pdbqt',
                                   center=cfg['center'], radius=28.0)
        out[name] = pdbqt
    return out


# ---------------------------------------------------------------------------
# 对接（自定义盒）
# ---------------------------------------------------------------------------
def dock_box(ligand_pdbqt: Path, receptor_pdbqt: Path, out_pdbqt: Path,
             center: tuple, size: float) -> Path:
    import subprocess
    cx, cy, cz = center
    cmd = [
        VINA, '--receptor', str(receptor_pdbqt), '--ligand', str(ligand_pdbqt),
        '--center_x', f'{cx:.3f}', '--center_y', f'{cy:.3f}',
        '--center_z', f'{cz:.3f}',
        '--size_x', f'{size:.1f}', '--size_y', f'{size:.1f}',
        '--size_z', f'{size:.1f}',
        '--exhaustiveness', str(EXHAUSTIVENESS), '--num_modes', str(NUM_MODES),
        '--out', str(out_pdbqt),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'Vina 失败：\n{proc.stderr}')
    return out_pdbqt


# ---------------------------------------------------------------------------
# 2D 成药性
# ---------------------------------------------------------------------------
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
              tmp: Path) -> Scored:
    safe = ''.join(c if c.isalnum() else '_' for c in label)
    lig = prep.prep_ligand(smiles, tmp / f'lig_{safe}.pdbqt')
    res18 = dock_box(lig, receptors['nav18'], tmp / f'res_{safe}_18.pdbqt',
                     RECEPTORS['nav18']['center'], RECEPTORS['nav18']['size'])
    res15 = dock_box(lig, receptors['nav15'], tmp / f'res_{safe}_15.pdbqt',
                     RECEPTORS['nav15']['center'], RECEPTORS['nav15']['size'])
    s18 = docking.read_score(res18)
    s15 = docking.read_score(res15)
    if s18 is None or s15 is None:
        raise RuntimeError(f'未能解析 Vina 打分：{label}')
    d2d = descriptors_2d(smiles)
    delta = s15 - s18
    comb = combined_score(s18, delta, d2d['qed'], d2d['sa'], d2d['membrane'])
    return Scored(label, smiles, s18, s15, delta, d2d['qed'], d2d['sa'],
                  d2d['membrane'], comb)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
HEADER = ['label', 'smiles', 'score_nav18', 'score_nav15', 'delta_sel',
          'qed', 'sa', 'membrane', 'combined']


def read_molecules(input_path: str, smiles_column: str = 'smiles',
                   label_column: str | None = None) -> list[tuple[str, str]]:
    """从 CSV 或 SMI 文件读取 (label, SMILES) 列表。

    CSV：SMILES 取 smiles_column；标签优先 label_column，否则 candidate_id（若存在），
    否则行号。SMI：每行第一个 token 为 SMILES，第二个为标签（可选）。
    """
    path = Path(input_path)
    suffix = path.suffix.lower()
    if suffix in ('.smi', '.txt'):
        out = []
        for i, line in enumerate(path.read_text(encoding='utf-8').splitlines()):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tokens = line.split()
            smi = tokens[0]
            label = tokens[1] if len(tokens) > 1 else f'mol_{i}'
            out.append((label, smi))
        return out

    rows = list(csv.DictReader(path.read_text(encoding='utf-8').splitlines()))
    if not rows or smiles_column not in rows[0]:
        raise ValueError(f'文件缺少 SMILES 列：{smiles_column}')
    out = []
    for i, row in enumerate(rows):
        smi = (row.get(smiles_column) or '').strip()
        if not smi:
            continue
        if label_column and row.get(label_column):
            label = row[label_column]
        elif row.get('candidate_id'):
            label = row['candidate_id']
        else:
            label = f'mol_{i}'
        out.append((str(label), smi))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='Nav1.8 亚型选择性 docking + 打分')
    parser.add_argument('--input', help='候选 CSV/SMI 文件（含 SMILES 列）；缺省用内置演示集')
    parser.add_argument('--smiles-column', default='smiles')
    parser.add_argument('--label-column', default=None)
    parser.add_argument('--exhaustiveness', type=int, default=None)
    parser.add_argument('--out', default=None, help='输出 TSV（缺省 run/subtype_scores.tsv）')
    args = parser.parse_args()

    global EXHAUSTIVENESS
    if args.exhaustiveness:
        EXHAUSTIVENESS = args.exhaustiveness

    workdir = Path(__file__).resolve().parent / 'run'
    tmp = workdir / 'tmp'
    tmp.mkdir(parents=True, exist_ok=True)

    print('== 准备受体 ==')
    receptors = prepare_receptors(workdir)
    for k, v in receptors.items():
        print(f'  {k}: {v}')

    if args.input:
        molecules = read_molecules(args.input, args.smiles_column, args.label_column)
        print(f'== 读取 {len(molecules)} 个分子：{args.input} ==')
    else:
        molecules = [
            ('A-803467', 'COc1cc(NC(=O)c2ccc(-c3ccc(Cl)cc3)o2)cc(OC)c1'),
            ('benzanilide-Cl', 'O=C(Nc1ccccc1)c1ccccc1Cl'),
            ('benzanilide-F', 'O=C(Nc1ccccc1)c1ccccc1F'),
            ('benzanilide-H', 'O=C(Nc1ccccc1)c1ccccc1'),
            ('benzanilide-OMe', 'COc1ccccc1C(=O)Nc1ccccc1'),
            ('benzanilide-OPh', 'O=C(Nc1ccccc1)Oc1ccccc1'),
        ]
        print('== 使用内置演示集 ==')

    results = []
    for label, smi in molecules:
        try:
            r = score_one(label, smi, receptors, tmp)
        except Exception as exc:  # 单个分子失败不中断整批
            print(f'[跳过] {str(label)[:16]:16s} {exc}')
            continue
        results.append(r)
        print(f'{str(label)[:16]:16s} Nav1.8={r.score_nav18:6.2f} '
              f'Nav1.5={r.score_nav15:6.2f} Δ={r.delta:+5.2f} '
              f'combined={r.combined:.3f}')

    results.sort(key=lambda r: -r.combined)
    tsv = Path(args.out) if args.out else workdir / 'subtype_scores.tsv'
    tsv.parent.mkdir(parents=True, exist_ok=True)
    lines = ['\t'.join(HEADER)]
    lines += ['\t'.join(r.row()) for r in results]
    tsv.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n== 结果已写 {tsv} ==')
    print('按 combined 降序（结合力+选择性+成药性）：')
    for r in results:
        print(f'  {str(r.label)[:16]:16s} combined={r.combined:.3f} '
              f'Δ={r.delta:+.2f}')


if __name__ == '__main__':
    main()
