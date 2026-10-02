# Nav1.8 小分子发现 — 全流程管线

2026-10-02 更新：对接入口已加入 Windows 路径发现、编号与来源保留、受体模板分型、
构象质量检查、逐任务记录与断点续跑。**原综合分公式、归一化和权重未改动**。
当前默认运行参考配体验证；候选对接必须提供通过验证的报告。
具体命令与结果见 [参考验证与修复说明](DOCKING_VALIDATION.md)。原 `run/` 内附带文件保持原样。

一个自包含的 **Nav1.8（SCN10A）钠通道小分子发现** 计算管线，把「生成 → 二维过滤 →
双受体对接 → 亚型选择性打分 → 回灌下一轮 → 主动学习 / 评价」串成一个可重复的闭环。

> 核心科学问题：找到**对 Nav1.8 强效、且相对 Nav1.5（心肌通道，心脏安全反筛）选择性**的
> 小分子。参考分子为 FDA 已批准的 VX-548（suzetrigine）与 Site C 阻断剂 A-803467。

---

## 1. 全流程概览

```
            ┌──────────────────────────────────────────────────────────────┐
            │                      nav18_full_pipeline                     │
            │                                                              │
  ① 生成    │  navgen generate  ──►  candidates.csv（含 2D 过滤 + 奖励）      │
  ② 对接打分│  score.py ──► 双受体对接 Nav1.8/Nav1.5 ──► subtype_scores.tsv  │
  ③ 回灌    │  navgen feedback ──► 下一轮种子 seeds.smi                       │
  ④ 主动学习│  navgen select-oracle ──► 待测批次 + 域外待测队列                 │
  ⑤ 评价    │  navgen evaluate ──► 分布 / 骨架 / 相似性 / 参考新颖性            │
            │                                                              │
  ⑥ 选择性  │  navgen train-selectivity ──► 配对 ΔΔG 随机森林模型（需真实标签） │
            │  （模型奖励折进 ① 的 reward；域外分子自动拒收）                   │
            └──────────────────────────────────────────────────────────────┘
                     ▲                                                  │
                     └──────────── ③ 产出的种子回到 ① 下一轮 ────────────┘
```

一条最简闭环：`generate → score.py → feedback → generate → ...`。② 的输出直接喂给 ③
（`--scores` 读 score.py 的 TSV），③ 的种子改配置文件 `seed_file` 后重跑 ① 即完成一轮。

---

## 2. 目录结构

```
nav18_full_pipeline/
├── README.md                 # 本文档
├── pyproject.toml            # 依赖声明（rdkit/meeko/sklearn/joblib/pandas/numpy）
├── score.py                  # ② 双受体对接 + 亚型选择性打分（主脚本）
├── navgen/                   # ①③④⑤⑥ 生成器 + 选择性 + 回灌 + 主动学习 + 评价
│   ├── cli.py                #   CLI 入口（8 个子命令）
│   ├── generation.py         #   种子附近生成（A-803467 类似物枚举）
│   ├── scoring.py            #   2D 过滤 + 奖励
│   ├── pipeline.py           #   生成/过滤/入库/导出的编排器
│   ├── selectivity.py        #   配对 ΔΔG 随机森林模型（训练/预测/域外检测）
│   ├── feedback.py           #   打分结果 → 下一轮种子
│   ├── active_learning.py    #   UCB 主动选样
│   ├── evaluation.py         #   候选库评价指标
│   ├── protein_rules.py      #   protein-word 启发式评分（可选）
│   ├── site_rules.py         #   Site C 二维结构规则代理
│   ├── conformers.py         #   多构象生成（ETKDG + MMFF）
│   ├── reinvent.py           #   REINVENT4 配置生成 + 外部奖励接口
│   ├── docking.py            #   Windows 版 Vina 编排（历史，本管线用 score.py）
│   └── ...                   #   config.py / __init__.py / __main__.py / cli_docking.py
├── scgd/                     # 对接原语（score.py 依赖的最小集合）
│   ├── prep.py               #   配体/受体 PDBQT 准备（Meeko + RDKit）
│   ├── docking.py            #   Vina 打分解析（read_score）
│   └── config.py             #   常量（Vina 路径等）
├── receptors/                # 受体结构（score.py 自动抽链/加氢/制备）
│   ├── 7WE4.pdb              #   Nav1.8（链 A，Site C）
│   └── 6LQA.pdb              #   Nav1.5（链 B，quinidine 结合腔）
├── data/                     # 种子、结构规则、标签模板
│   ├── seeds.smi             #   演示种子库
│   ├── seeds_a803467.smi     #   A-803467 种子
│   ├── nav18_site_c_rules.json  # Site C 关键残基 + 结构规则
│   ├── paired_labels.template.csv  # 配对 ΔΔG 标签模板（训练选择性模型用）
│   ├── feedback.template.csv       # 回灌标签模板
│   └── demo_protein.fasta    #   演示蛋白序列（protein-word 评分用）
├── configs/                  # 生成/筛选配置（路径均相对配置文件）
│   ├── demo.json
│   ├── a803467_site_c.json
│   └── site_c_reference.json
└── tests/                    # 单元测试（9 个文件）
```

---

## 3. 依赖与安装

- Python ≥ 3.10；新增修复在本项目 Windows / Python 3.10 环境验证。
- 对接依赖包括 RDKit、Meeko、PDBFixer、OpenMM、Gemmi；原生成模块还使用 pandas、scikit-learn、joblib。

```bash
cd nav18_full_pipeline
python3 -m pip install -r requirements-docking.txt
# 若用独立环境运行完整生成模块，再安装 .[docking,dev]
# 与 nav1p8-molecule-generator 共用环境时，不要重新安装同名 navgen 包。
```

- **AutoDock Vina 1.2.7**：优先用 `--vina` 显式指定；否则检查 `VINA_EXECUTABLE`、PATH、
  同级主项目中的 `vina_1.2.7_win.exe` 和 macOS Homebrew 路径。普通 Vina 使用 CPU。

```bash
python3 -m pytest -q
```

---

## 4. 快速开始（A-803467 局部枚举 → 对接 → 回灌）

```bash
cd nav18_full_pipeline

# ① 生成候选（A-803467 类似物，2D 过滤后 ~7 个）
python3 -m navgen generate --config configs/a803467_site_c.json --output outputs/round1

# ②a 参考配体重对接；必须先确认报告 gate_passed=true
python3 score.py --stage reference --workdir outputs/reference_check

# ②b 验证通过后，双受体对接 + 原综合打分
python3 score.py --stage candidates --input outputs/round1/candidates.csv \
                 --smiles-column smiles --label-column candidate_id \
                 --validation-report outputs/reference_check/report.json \
                 --out outputs/round1/subtype_scores.tsv

# ③ 回灌：按 docking 初筛挑下一轮种子
python3 -m navgen feedback --scores outputs/round1/subtype_scores.tsv \
                 --output data/round2_seeds.smi --ranking docking_triage --top-k 20

# 下一轮：把 config 的 seed_file 改成 data/round2_seeds.smi，回到 ①
```

**默认 `--stage reference`**：验证 A-803467、奎尼丁，并运行奎尼丁 +1 质子化对照：

```bash
python3 score.py
```

---

## 5. 各模块职责与命令

| 步骤 | 命令 | 输入 → 输出 | 说明 |
|---|---|---|---|
| 生成 | `navgen generate` | 种子 → `candidates.csv` | 2D 过滤 + 奖励（QED/膜/SiteC/protein-word/选择性模型） |
| 参考验证 | `python3 score.py --stage reference` | 实验配体 → `report.json` | 同一受体坐标系下的重对接与姿势检查 |
| 对接打分 | `python3 score.py --stage candidates` | SMILES → `subtype_scores.tsv` | 验证通过后，Nav1.8 + Nav1.5 双受体对接与打分差 |
| 回灌 | `navgen feedback` | 打分结果 → `seeds.smi` | top-k + 随机探索 + 指纹去近邻 |
| 主动学习 | `navgen select-oracle` | 池 + 模型 → `oracle_batch.csv` | UCB = 均值 + β×树间标准差，域外进待测队列 |
| 评价 | `navgen evaluate` | 候选库 → `eval.json` | validity/uniqueness/分布/骨架/相似性/新颖性 |
| 训练选择性模型 | `navgen train-selectivity` | 配对标签 → `.joblib` | 随机森林，Murcko 骨架留出验证，比常数基线 |
| REINVENT 接口 | `navgen prepare-reinvent` / `score-stdin` | 配置 / SMILES → 奖励 | Mol2Mol prior + ExternalProcess 奖励回传 |

CLI 全命令：`python3 -m navgen --help`。

---

## 6. 打分方案

### 6.1 双受体对接 + 选择性（score.py）

每个分子分别对接 Nav1.8 与 Nav1.5，选择性 = 两者 Vina 打分差：

```
Δ = score_nav15 − score_nav18        （> 0 偏向 Nav1.8）
综合分 S = 0.40·结合力 + 0.30·选择性 + 0.15·QED + 0.07·SA + 0.08·膜
```

| 受体 | PDB | 链 | 位点 | 盒中心 (x,y,z) | 盒尺寸 |
|---|---|---|---|---|---|
| Nav1.8 | 7WE4 | A | Site C（中央腔） | (137.204, 136.509, 127.538) | 24 Å |
| Nav1.5 | 6LQA | B | quinidine 结合腔 | (128.692, 125.579, 137.582) | 24 Å |

各分项归一化到 [0,1]：结合力 `clip((−score−5)/5)`；选择性 `clip(Δ/3,−1,1)·0.5+0.5`；
QED 直接用；SA `clip((5−sa)/3)`；膜分项按原代码，是 MW/cLogP/TPSA 三项开区间
指示值（区间内为 1，边界及区间外为 0）的均值，并非双 sigmoid。本次保留此行为和所有权重；
测试通过原函数 AST 指纹核对公式没有变化。

### 6.2 生成器奖励（navgen/scoring.py）

2D 硬过滤（MW/cLogP/TPSA/QED/SA/必需 SMARTS）不通过为 0；通过后奖励为各分项加权和，
可叠加 protein-word、Site C 代理、REINVENT 原生指标，以及**已训练的选择性模型奖励**：

```
选择性奖励 = sigmoid((预测均值 − λ×树间标准差) / scale)
```

树间标准差是**未校准启发式**，不是置信区间。与训练集最大 Tanimoto 低于阈值
（`selectivity_ood_min`，默认 0.35）判为域外，启用选择性奖励时域外分子**无法通过候选筛选**，
但仍可进入主动学习待测队列。

### 6.3 选择性模型（navgen/selectivity.py，需真实标签）

```bash
python3 -m navgen train-selectivity --data data/paired_labels.csv \
                 --output models/nav18_ddg.joblib
```

标签格式（`data/paired_labels.template.csv`）：`smiles,ddg_kcal_mol,species,state,protocol,label_source`。
约定 **ΔΔG = DG(Nav1.5) − DG(Nav1.8)，正值偏向 Nav1.8**；`species` 必须 `Homo sapiens`；
单文件只允许一个 `(state, protocol)` 测定上下文；`label_source` 限 `experiment` / `free_energy`
（**禁用 docking 标签当选择性真值**）。程序按 Murcko 骨架留出验证，报告 MAE/RMSE 与常数基线。

在配置中启用（扁平字段，位于 `filters`）：

```json
{
  "state": "inactivated",
  "protocol": "your_protocol",
  "filters": {
    "selectivity_model": "../models/nav18_ddg.joblib",
    "selectivity_weight": 0.2,
    "selectivity_ood_min": 0.35,
    "selectivity_uncertainty_penalty": 1.0,
    "selectivity_reward_scale": 2.0
  }
}
```

上下文不匹配直接拒绝加载模型。joblib 只应加载自己训练或可信来源的文件。

---

## 7. 诚实边界（务必阅读）

1. **`Δ` 是 docking 打分差，不是实验结合自由能或亚型选择性。** 不假定统一的固定误差范围；
   跨受体分差仅是探索性指标。参考重对接先检验姿势恢复，不能证明活性或选择性预测有效。
2. 两亚型受体来自不同实验结构、盒坐标不同；`nav15` 分数只作反筛留存，**不能把双靶
   静态对接差标成选择性真值**（`feedback --ranking docking_triage` 仅按
   `−Nav1.8 docking / 重原子数` 排序）。
3. **选择性模型需要真实配对标签才能部署**；无数据时只能冒烟测试，不得捏造标签。
4. 更强的选择性命中需要接触级反筛（如 SCGD 的 KKGS 残基接触计数）或配对 ΔΔG 模型。
5. 生成器当前是「可解释局部枚举 + 2D 先验」，不是高质量 de novo 生成模型；REINVENT4
   Mol2Mol 需本机装好环境与 prior 后运行。

---

## 8. 测试

```bash
python3 -m pytest -q
```

覆盖：生成回放 / 非法 SMILES / 双向膜区间 / 配置校验 / 骨架无交叉划分 / 物种与状态
限制 / 域外检测 / 主动选样 / 回灌排序与去重 / 评价指标。
新增覆盖：原打分函数不变 / BOM 编码与原编号 / 构象收敛与立体化学 / 失败记录与续跑哈希 /
固定坐标系 RMSD / 实验参考坐标 / 验证门槛与失败分母。
