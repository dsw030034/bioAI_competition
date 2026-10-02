# -*- coding: utf-8 -*-
"""对接：调用 AutoDock Vina，解析打分与对接构象。

设计为独立于配体来源：输入配体 PDBQT 与受体 PDBQT 路径，输出结果 PDBQT。
对接盒与 Vina 参数统一取自 config。
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .prep import AD4_TO_ELEMENT


def dock(ligand_pdbqt: Path, receptor_pdbqt: Path, out_pdbqt: Path,
         vina: str = None, exhaustiveness: int = None,
         num_modes: int = None) -> Path:
    """对接单个配体到受体，结果写入 out_pdbqt。返回 out_pdbqt 路径。"""
    vina = config.VINA if vina is None else vina
    exhaustiveness = config.VINA_EXHAUSTIVENESS if exhaustiveness is None else exhaustiveness
    num_modes = config.VINA_NUM_MODES if num_modes is None else num_modes
    cx, cy, cz = config.BOX_CENTER
    s = config.BOX_SIZE

    cmd = [
        vina, '--receptor', str(receptor_pdbqt), '--ligand', str(ligand_pdbqt),
        '--center_x', f'{cx:.2f}', '--center_y', f'{cy:.2f}', '--center_z', f'{cz:.2f}',
        '--size_x', f'{s:.1f}', '--size_y', f'{s:.1f}', '--size_z', f'{s:.1f}',
        '--exhaustiveness', str(exhaustiveness), '--num_modes', str(num_modes),
        '--out', str(out_pdbqt),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'Vina 失败：\n{proc.stderr}')
    return out_pdbqt


def read_score(result_pdbqt: Path) -> float | None:
    """读取 Vina 最佳打分（kcal/mol）。"""
    for ln in result_pdbqt.read_text().splitlines():
        if ln.startswith('REMARK VINA RESULT'):
            try:
                return float(ln.split()[3])
            except (IndexError, ValueError):
                return None
    return None


def read_ligand_heavy(result_pdbqt: Path, model: int = 1) -> list[np.ndarray]:
    """读取结果 PDBQT 中第 model 个构象的配体重原子坐标列表。"""
    atoms, cur = [], False
    for ln in result_pdbqt.read_text().splitlines():
        if ln.startswith(f'MODEL {model}'):
            cur = True
            continue
        if cur and ln.startswith('ENDMDL'):
            break
        if cur and (ln.startswith('ATOM') or ln.startswith('HETATM')):
            atype = ln[77:79].strip() if len(ln) >= 79 else ln.split()[-1]
            el = AD4_TO_ELEMENT.get(atype, atype)
            if el != 'H':
                atoms.append(np.array([float(ln[30:38]), float(ln[38:46]),
                                       float(ln[46:54])]))
    return atoms


@dataclass
class DockResult:
    """单分子在单个状态上的对接结果。"""
    label: str
    smiles: str
    state: str                     # 'down' | 'up'
    score: float | None = None
    ligand_heavy: list = field(default_factory=list)

    def __post_init__(self):
        if self.score is None:
            self.score = 0.0


@dataclass
class MoleculeResult:
    """单分子在 down/up 双态上的完整结果（对接 + 奖励）。"""
    label: str
    smiles: str
    score_down: float
    score_up: float
    kkgs_down: int
    kkgs_up: int

    @property
    def K_window(self) -> int:
        """状态选择窗口：KKGS接触(down) − KKGS接触(up)。"""
        return self.kkgs_down - self.kkgs_up

    @property
    def W_score(self) -> float:
        """Vina 打分窗口（对照量）：score_up − score_down。"""
        return self.score_up - self.score_down

    def to_row(self) -> list:
        return [self.label, self.smiles, f'{self.score_down:.2f}',
                f'{self.score_up:.2f}', f'{self.W_score:.2f}',
                str(self.kkgs_down), str(self.kkgs_up), f'{self.K_window:+d}']
