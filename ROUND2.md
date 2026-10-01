# 第二轮：局部优化与三维骨架探索

> 本文是阶段记录。最新已完成 2,944 个候选的扩容，见 [GENERATION_EXPANSION.md](docs/GENERATION_EXPANSION.md)；GitHub 上的共享结果和 CPU 复现步骤见 [TEAM_QUICKSTART.md](docs/TEAM_QUICKSTART.md)。

> 2026-10-01 验证更新：正确 A-803467 种子的 Mol2Mol GPU 采样已实际完成（100 条、61 个唯一结构），并与本地生成合并为 79 个严格主库候选＋38 个扩展候选。环境、原始结果与复现命令见 [REMOTE_GPU.md](docs/REMOTE_GPU.md)。ShEPhERD 仍未实际运行；下文“尚未执行 Mol2Mol”是上一阶段记录。

> 2026-09-30 当前执行范围：先做 **A-803467 单系列、多生成方法**，之后再扩展多种子。首轮暂不推进打分或 RL。本地枚举与反应重组统一入口、核心/位点复核及已运行结果见 [A803467_GENERATION.md](docs/A803467_GENERATION.md)；Mol2Mol 可使用远程 GPU，见 [MOL2MOL_REMOTE.md](docs/MOL2MOL_REMOTE.md)。下文保留此前 ROUND2 的阶段方案与当时验证记录，不代表所有分支已运行。

本轮实现位于 **nav1p8-molecule-generator 子目录**。父目录的旧包已移入 archive，以下命令仍以本子目录为工作目录。

```powershell
Set-Location D:/2026-2027_autumn/bioAI_competition/nav1p8-molecule-generator
$py = (Resolve-Path ../.venv/Scripts/python.exe).Path
```

## 本轮改变

- Mol2Mol 从配置的 seed_file 读取真实种子，不再写死苯甲酰苯胺；保存种子和配置快照。
- RL 配置使用真实 ExternalProcess 项目奖励。绝对路径 score_bridge.py 强制加载本 checkout，不受 REINVENT 工作目录/父目录同名包影响。
- 离线 REINVENT CSV 与 RL 共用 ProjectScorer：Morgan/ECFP4 相似性、核心约束、过滤失败奖励归零、排除种子原样复制。空行保留为零分，批次长度不变。
- 多样性过滤改为 PenalizeSameSmiles，避免固定 Murcko 核心的类似物全部占用同一个小桶。
- 新增真实酰胺反应重组基线，代替固定字符串拼接；完整记录反应物与模板。原 generate 命令保留兼容，仍为旧局部编辑器。
- 新增 ShEPhERD 官方现代 API 推理脚本，使用 mosesaq、三维表面/静电/药效团条件；固定原始采样预算，调用 shepherd-score 做重构、xTB 弛豫和条件相似性评价。
- 两条任务分开配置。骨架探索不强制酰胺或高二维相似度；统一评测输出将其保留为待三维审查队列。

新配置不启用未经验证的 site/fenestration/selectivity 手工代理，也不使用旧版 reinvent_native_metrics（其区间函数不等价于官方平滑 double sigmoid）。本轮 RL 奖励仍是二维化学约束，不宣称学习了 Nav1.8 活性。酰胺氮不再被 site proxy 一概误标为碱性中心，但该 proxy 仍非 pKa/选择性模型。

## 参考三维结构

原 `Nav1.8_structure/a-803467-s2785.SDF` 经 RDKit 检查为 **2D**，不能直接用于三维条件生成。

已从 RCSB 7WE4 下载并提取 `95T / auth chain A / auth residue 2004`：

- `data/reference_7we4/a803467_bound.sdf`：实验 EM 配体坐标；
- `data/reference_7we4/source_ligand.pdb`：原始对应 HETATM；
- `data/reference_7we4/provenance.json`：URL、哈希及身份校验。

使用 seed 模板赋键级并验证规范 SMILES 完全一致，重原子坐标逐项验证不变。原供应商文件未修改。重新抓取使用 `scripts/fetch_bound_reference.py`；已有目录拒绝覆盖。该脚本固定此 PDB/配体身份，不适用于任意结构。

## A. 本地运行片段重组基线

```powershell
& $py -m navgen.reaction_library --config configs/round2_local.json --blocks data/round2_building_blocks.csv --output outputs/reaction_trial/raw.csv --budget 300
& $py -m navgen.experiment --config configs/round2_local.json --input outputs/reaction_trial/raw.csv --output outputs/reaction_trial/evaluation --method amide_recombination --budget 300
```

示例砌块来自前期项目的 35 条人工结构列表，不是可采购目录；包括重建原种子的片段。反应重组目前每次替换酰胺一侧，不做任意多步组合。化学空间耗尽会如实报告，不靠重复补齐 300 个。原始 CSV 保留重复和种子，统一评价再去重、过滤。模板通过不保证产率/可合成性。

局部核心 SMARTS 为 `O=C(Nc1ccccc1)c1ccco1`：保留苯胺—酰胺—呋喃核心，允许远端变化；会有意排除部分核心变化产物。日后放宽此约束需作为不同实验组。

## B. Mol2Mol 与真实外部奖励

```powershell
& $py -m navgen prepare-reinvent --config configs/round2_local.json --prior D:/models/mol2mol_medium_similarity.prior --output outputs/mol2mol_trial --device cuda:0 --num-smiles 300 --python $py
# 在 REINVENT4 环境执行，路径改为实际绝对路径：
reinvent D:/2026-2027_autumn/bioAI_competition/nav1p8-molecule-generator/outputs/mol2mol_trial/sampling.toml
```

配置生成允许 prior 路径尚未存在，以便先准备 GPU 机器的运行文件；实际执行 REINVENT 前必须提供真实权重。生成配置不等于已执行采样。

```powershell
& $py -m navgen.experiment --config configs/round2_local.json --input outputs/mol2mol_trial/sampling_mol2mol.csv --output outputs/mol2mol_trial/evaluation --method mol2mol --budget 300
```

先验证冻结 prior 采样。需要 RL 时在 REINVENT 环境运行同目录 `staged_learning.toml`。默认至少 25 步，最多 100 步；这仍是实验参数，不是收敛保证。采样 `unique_molecules=false` 保留重复以便统计。Mol2Mol 的 num_smiles 按输入种子计，多个种子时需核对总数。

## C. ShEPhERD 预检与 GPU 小试验

```powershell
& $py scripts/run_shepherd.py --config configs/round2_exploration.json --reference data/reference_7we4/a803467_bound.sdf --reference-source "PDB 7WE4 95T A2004" --output outputs/shepherd_trial --preflight
```

预检检查参考身份、3D 标志、依赖与 xTB；不下载模型、不创建输出目录。缺依赖退出码为 2。

本项目 RDKit 环境不能直接混装上游依赖：ShEPhERD 官方要求 Python>=3.9,<3.12，RDKit>=2023.03,<2025.03，且包含 PyTorch/PyG、open3d、xTB。建议在独立 **Linux GPU 环境**按以下官方说明安装：

- https://github.com/coleygroup/shepherd （核对源码提交 `1707fddc3b8752c2e275562c1a8256d49ed401be`）
- https://github.com/coleygroup/shepherd-score
- https://xtb-docs.readthedocs.io/en/latest/setup.html

runner 直接引用 checkout，不必安装父目录的旧 navgen 包。模型首次由 `load_model('mosesaq')` 自动下载，版本记录到 manifest。为严格复现须另固定 shepherd-score 与模型缓存文件版本/哈希；当前脚本尚未锁定全部外部依赖。

在配置齐全的 GPU 环境，从本项目目录执行：

```bash
python scripts/run_shepherd.py --config configs/round2_exploration.json --reference data/reference_7we4/a803467_bound.sdf --reference-source 'PDB 7WE4 95T A2004' --output outputs/shepherd_trial --samples 100 --batch-size 5 --steps 400 --seed 20260930
```

原子数按参考含氢原子数的 −4/0/+4 小范围扫描；预算严格按原始尝试数。对参考仅做 xTB 单点电荷，不主动优化其重原子姿势。输出：

- `raw_samples.pkl`：保留原始尝试，仅加载自己可信生成的 pickle；
- `reference_centered.sdf`：实际使用的参考构象；
- `shepherd_summary.csv` / `shepherd_details.csv`：官方评价器提供的有效率、弛豫、形状/静电/药效团等指标；
- `generated_relaxed.sdf`：成功重构并弛豫的产物，携带 raw_sample_index；
- `manifest.json`：运行参数、参考校验、依赖版本、成功/失败状态。

**ShEPhERD 推理及官方三维评价尚未在本机执行验证**。脚本按官方示例与 API 源码实现，不能把单元测试当成 GPU 运行成功。生成坐标以参考质心为原点，不能未经对齐直接用于原受体坐标系；必须重新对接或应用正确变换。

随后统一做二维评测：

```bash
python -m navgen.experiment --config configs/round2_exploration.json --input outputs/shepherd_trial/generated_relaxed.sdf --output outputs/shepherd_trial/evaluation --method shepherd --reference data/reference_7we4/a803467_bound.sdf --budget 100
```

统一评价器仅补充 RDKit O3A 形状相似性，不冒充完整药效团/静电验证；应结合 shepherd_details.csv 按 raw_sample_index 查看上游评价。其 `accepted_2d` 不等于三维验证通过，探索组 `ready_for_docking` 保持 false，需完成姿势/应变/药效团审查后选取送 docking 的集合。该标记不表示已对接或具有活性。

## 对照与后续边界

先小试验，再把 Mol2Mol/ShEPhERD 扩至 100–300 次原始采样；示例枚举空间耗尽单独报告。统一输出 candidates.csv / all_scored.csv / metrics.json，检查唯一合格数、核心/骨架覆盖和物化性质，再按相同规则与相同预算选 docking 子集。不可用已过滤 SDF 行数代替生成尝试数；ShEPhERD 的分母要查 raw_samples 与上游 summary。

本轮尚未接入 PrexSyn/SynFormer 权重；反应重组提供本地可运行的合成约束基线。后续若引入合成投影，要对投影后新结构重算三维和 docking。未接入虚构的 Nav1.8 选择性模型。后续独立开展过受体准备与重对接验证，见 DOCKING_VALIDATION.md；当前生成工作不依赖 docking 通过。

测试：`& $py -m pytest -q`。新测试包含跨工作目录 RL 子进程、空行长度、seed 身份、核心约束、失败归零、模板回放、去重计数和二维参考拒绝。

## 本机验证记录

- 单元/集成测试：14 项通过。
- 反应试验：`outputs/round2_reaction/raw.csv`，35 次模板枚举，34 个唯一结构；严格核心配置下 17 个通过二维筛选，3 个 Murcko 骨架。
- 全部枚举产物已通过模板层 sanitize，此处不能把 100% 可解析率与未经筛选的语言模型原始有效率直接比较。
- `outputs/round2_reaction/evaluation_final/` 保存统一评价；`outputs/round2_mol2mol_prepared/` 保存配置，尚未执行 Mol2Mol 推理。
- ShEPhERD 参考身份/3D 检查通过；缺 torch、shepherd、shepherd_score、xTB，本机也未发现 nvidia-smi。没有输出伪造的 ShEPhERD 分子或性能结论。
