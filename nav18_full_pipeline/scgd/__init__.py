# -*- coding: utf-8 -*-
"""SCGD：State-Conditional Generative Design（状态条件生成式设计）。

把离子通道的构象态（静息 down / 激活 up）作为生成条件与奖励目标，
显式优化 use-dependent 状态选择窗口，而不是只优化静态单态打分。

靶点：Nav1.8（SCN10A）VSD2，基准分子 VX-548（suzetrigine）。
"""

__version__ = "0.1.0"

from .config import (
    VX548_SMILES,
    A803467_SMILES,
    KKGS_RESIDUES,
    BOX_CENTER,
    BOX_SIZE,
)

__all__ = [
    "VX548_SMILES",
    "A803467_SMILES",
    "KKGS_RESIDUES",
    "BOX_CENTER",
    "BOX_SIZE",
]
