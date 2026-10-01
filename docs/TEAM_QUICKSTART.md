# 队友更新与复现

截至 2026-10-01，项目已完成 A-803467 系列的多方法生成、结构审计和千级扩容，得到 **2,944 个非母体唯一结构**：主库 1,543、扩展库 1,072、系列外保留 329。所有旧 134 个结构保留。当前不按 docking 或活性分数排名，ShEPhERD 尚未运行。

## 获取更新和查看结果

已有仓库且没有本地未提交改动时：

```bash
git switch main
git pull --ff-only origin main
```

首次获取：

```bash
git clone https://github.com/GHrendu/nav1p8-molecule-generator.git
cd nav1p8-molecule-generator
```

有未提交工作时先自行提交或备份再同步，不要用 `reset --hard` 丢弃改动。

不安装环境也能直接查看 [共享结果](../results/a803467_20261001/README.md)：

- [候选总表](../results/a803467_20261001/library/all_retained.csv)：2,944 个唯一结构、SMILES、方法来源、新增标记和分类。
- [7 页代表结构对照图](../results/a803467_20261001/review_first/)：左边母体，右边候选；橙色标出变化。
- [分批采样增量](../results/a803467_20261001/library/mol2mol_batch_growth.csv)：5×200 次采样的新增收益。
- [三维代表试跑](../results/a803467_20261001/preparation_pilot/)：79 个构象；18/19 个输入完成，1 个立体化学待确认。

完整二维 SDF 和 247 页 PNG 可在本地从共享原始输入重建。模型权重、虚拟环境、SSH 凭据和全部本地 `outputs/` 不在这次提交中。

## 安装本地 CPU 环境

推荐 Python 3.10 或 3.11。本次实际使用 Python 3.10、RDKit 2026.3.6；如需逐结构精确复现，固定同一 RDKit 版本。

Windows PowerShell，在仓库根目录：

```powershell
py -3.10 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -e '.[dev]' 'rdkit==2026.3.6'
& ./.venv/Scripts/python.exe -m pytest -q
& ./.venv/Scripts/python.exe scripts/reproduce_shared_expansion.py --verify-only
& ./.venv/Scripts/python.exe scripts/reproduce_shared_expansion.py --output outputs/team_reproduced --full-gallery
```

Linux/macOS：

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]' 'rdkit==2026.3.6'
.venv/bin/python -m pytest -q
.venv/bin/python scripts/reproduce_shared_expansion.py --output outputs/team_reproduced --full-gallery
```

复现脚本先校验共享文件 SHA256，再重跑位点枚举、虚拟砌块组合，导入已经发布的真实 GPU 采样，逐个核对最终候选 ID、SMILES 和分类。**不需要 GPU、REINVENT4 权重或原服务器登录权限**。输出目录须不存在，避免覆盖历史记录；不加 `--full-gallery` 时只生成代表图和完整 SDF/表格。

本次提交前已在仅含 Git 待提交文件的干净目录实际验证：66 个共享文件的哈希校验通过，35 项测试通过，CPU 复现的 2,944 个候选 ID、SMILES 和分类与发布快照完全一致。

本次主要在 Windows 验证。PNG 脚本优先用微软雅黑；其他系统可能需要自行配置中文字体。用 SDF/SMILES 查看结构不依赖中文字体。

更多三维准备：

```powershell
& ./.venv/Scripts/python.exe -m navgen.ligand_preparation --selection outputs/team_reproduced/pilot_selection.json --output outputs/team_preparation
```

三维准备保留输入电荷和化学状态，未做 pH 预测或互变异构体枚举；未指定立体化学会暂停并标记。

## 当前代码入口

| 任务 | 入口 |
|---|---|
| 首轮小批次生成 | `python -m navgen.series_generation` |
| 首轮来源与约束审计 | `python -m navgen.generation_audit` |
| 词表与双侧砌块扩容 | `python -m navgen.generation_expansion` |
| 共享千级结果 CPU 复现 | `python scripts/reproduce_shared_expansion.py` |
| 独立 Mol2Mol GPU 分批采样 | `scripts/run_mol2mol_batches.py` |
| 输入状态三维准备 | `python -m navgen.ligand_preparation` |
| 全库 PNG 导出 | `scripts/export_structure_pngs.py` |

当前生成路线和数量解释见 [GENERATION_EXPANSION.md](GENERATION_EXPANSION.md)。旧 `navgen generate` 和 `ROUND2.md` 中的奖励流程为历史入口，不作为本轮默认流程。原 notebook、`version1.py` 和旧 Nav1.7 说明已从当前项目目录清理，可从 Git 历史查看。

Docking 实施代码与测试也已同步，但重对接验证仍未过门槛，不用于候选可靠排序。详见 [DOCKING_VALIDATION.md](../DOCKING_VALIDATION.md) 和[共享验证摘要](../results/docking_validation_20260930/README.md)。
