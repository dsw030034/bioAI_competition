# A-803467 千级生成扩容

2026-10-01：用户概览上一轮结构后决定全部先保留，并授权扩充位点词表、双侧反应组合和 Mol2Mol 分批采样。本轮已实际完成，不是只准备配置。没有新增 docking 奖励、活性排序或模型训练。

## 实际结果

旧批次 134 个非母体结构全部保留，合并去重后 **2,944 个**，净新增 **2,810 个**。

| 分类 | 旧批次 | 当前合并库 | 净新增 |
|---|---:|---:|---:|
| 严格主库 | 79 | 1,543 | 1,464 |
| ROUND2 扩展库 | 38 | 1,072 | 1,034 |
| 系列外保留结构 | 17 | 329 | 312 |

分类边界保持原定义，保留结构不等于活性或合成验证。旧结构标记为概览后保留，新结构仍为未审阅；不据原审阅意见将新结构自动标为通过。

本轮生成 5,399 条原始记录，加上历史 236 条，共 5,635 条；重复、母体重建和 1 条模型解析失败均在完整 lineage 中。最终候选数不包含母体和无效结构。

## 三种扩容如何实现

**位点枚举**：R1 从 7 种扩大到 11 种，R2/R3 各从 6 种扩大到 16 种。所有新增片段必须在已审阅主库的同一连接位置出现过，来源记录在 `vocabulary_evidence.csv`；不把模型提议当成实验 SAR。增加 `site_triple`，仍按单点、双点、三点分别保留操作来源。R2/R3 对称去重后，枚举得到 **1,495 个非母体唯一结构**（25 单点、270 双点、1,200 三点）。没有用重复操作凑唯一结构数。

**双侧酰胺重组**：在原有 35 个砌块基础上，从旧主库和扩展库作酰胺逆向拆分，合并为 **24 个酸＋66 个胺**；酸侧和胺侧两两组合，实际产生 **1,584 个不同产物，含 1 个母体**。其中主库 406 个、扩展 916 个、系列外 261 个。虚拟砌块记录原砌块 ID 或拆分来源候选 ID，产物记录两个砌块 ID 和反应模板。未验证采购、保护基、条件、收率或反应选择性。

**Mol2Mol**：远程 RTX 4090 完成 **5 批×200 条**，随机种子 20261011–20261015，保持原始官方 medium prior、温度 1.0、multinomial、保留重复。没有以本批次新候选作递归输入，五批仍以已知 A-803467 为输入。1,000 条中 999 条可解析；含母体的唯一结构 352 个，排除母体后 351 个，其中 299 个旧批次没有出现。再扣除本轮本地枚举/重组已覆盖结构，Mol2Mol 净补充 **250 个**。

| 批次 | 原始采样 | 相对旧批次及前批新增 | 再扣除本轮本地扩容后新增 |
|---|---:|---:|---:|
| 1 | 200 | 88 | 75 |
| 2 | 200 | 60 | 45 |
| 3 | 200 | 56 | 47 |
| 4 | 200 | 50 | 45 |
| 5 | 200 | 45 | 38 |

新增收益下降，但仍有结构增量。每批的主库/扩展/系列外和位点片段类型见 `mol2mol_batch_growth.json`。数量不作为三种方法活性质量的比较；原始记录口径和搜索空间不同。按家族统计时，单/双/三点合并为枚举，单侧/双侧反应合并为反应重组，细分方法仍在 lineage 中。

主库最终单点/双点/三点变化为 **27/296/1,220**。目前数量增长的主要来源是组合展开，并没有形成同等数量的独立药化假设。后续应分组比较，尤其保留单点变化便于归因。

## 结果在哪里看

主结果：`outputs/a803467_expanded_20261001/`。

- `扩容结果.md`、`summary.json`：数量、来源和参数概览。
- `structure_png/`：全部 2,944 个结构，247 页 PNG（主库 129 页、扩展库 90 页、系列外 28 页），可以直接用 Windows 照片打开。`index.csv` 按候选 ID 找页码；图上 `edits` 为改变位点数，编号不是排名。
- `review_first/`：21 个新增代表与母体并排对照，共 7 页；包含 3 个系列外边界例子。不是优选活性分子。
- `all_retained.csv`：唯一结构总表，包含新增标记、来源方法、位点变化和描述性理化性质。
- `candidates*`、`expanded_candidates*`、`outside_candidates*`：各层 CSV/SMI/二维 SDF。全库 SDF 仍然是二维坐标。
- `virtual_blocks.json/.csv`、`vocabulary_evidence.csv`、`lineage.jsonl`：砌块、片段和所有生成记录的追溯链。
- `method_contributions.csv`、`contributions.json`：合并库各方法的独有、共有和约束原因。

远程回传证据：`outputs/remote_mol2mol_scaleup_20261001/`，包含五批原始 CSV、模型/运行哈希、配置和日志，以及合并 CSV。合并文件 SHA256 为 `c5be2e8c9373bf77696da19e4b07d83d80f1034e6b96d287ad81ba531bbbfd97`，本地与远程 manifest 一致。远程结束后已检查 GPU 无计算进程、利用率 0%；实例本身未关闭。

## 三维代表试跑

`outputs/a803467_expanded_preparation_20261001/`：母体＋12 个新主库＋6 个新扩展结构，按来源/变化覆盖和指纹多样性选取。

19 个输入中，18 个在输入化学状态下完成准备，保留 **79 个三维构象**；1 个因存在未指定立体化学而暂停。9 个结构有化学状态审阅提示。没有自动决定 pH 质子化、互变异构体或未知立体化学；不是全库已经可直接 docking。

## 复现

本地子项目目录：

```powershell
& ../.venv/Scripts/python.exe -m navgen.generation_expansion --config configs/a803467_expansion.json --baseline outputs/a803467_with_mol2mol_20261001 --mol2mol-csv outputs/remote_mol2mol_scaleup_20261001/sampling_combined.csv --output outputs/new_expansion
& ../.venv/Scripts/python.exe scripts/export_structure_pngs.py --input outputs/new_expansion --output outputs/new_expansion/structure_png
& ../.venv/Scripts/python.exe -m navgen.ligand_preparation --selection outputs/new_expansion/pilot_selection.json --output outputs/new_expansion_preparation
```

输出目录必须为新目录。配置预算按细分方法显式设置，空间超预算时报错，不静默截断。本轮预算没有触发截断。每个历史事件 ID 增加源批次哈希前缀，保留原事件 ID；不覆盖旧库。

远程脚本 `scripts/run_mol2mol_batches.py` 与 `run_series_mol2mol.py` 放在同一目录，连同系列配置和历史结构 `baseline.smi` 上传：

```bash
/root/shared-nvme/nav18/env/bin/python run_mol2mol_batches.py --series a803467_generation.json --prior /root/shared-nvme/nav18/models/mol2mol_medium_similarity.prior --baseline baseline.smi --output /root/shared-nvme/nav18/runs/new_5x200
```

默认 5×200；`--resume` 只复用输入哈希、参数和输出哈希一致的完整批次。未完成/被修改的批次不会当作成功结果读取。

验证：本次项目全套 **35 项测试通过**；另读取并核对全部 2,944 条二维 SDF 的身份、分类与坐标类型，确认旧 134 个结构完整保留、5,635 个事件 ID 唯一、90 个虚拟砌块来源可追溯、五批远程 CSV 校验一致。图册及三维 SDF 验证记录见主结果目录 `validation.json`。

现在已达到本轮千级目标。后续宜先按单点/多点、片段类别、化学状态和合成可行性整理代表集；是否继续增加采样、改变种子或接入 ShEPhERD，应依据仍未覆盖的结构问题决定。
