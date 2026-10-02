# Nav1.8 小分子生成模块

依据《课题汇报_Nav1.8小分子生成算法设计.md》的第 4–9 节实现。输入孔道位点种子，输出带来源、二维性质、奖励分和可选三维构象的候选库，供 Nav1.8 / Nav1.5 下游对接和下一轮生成使用。

当前可直接运行的部分是 **反应模板局部生成 → 二维过滤 → 多构象导出 → 外部评分回灌**。化学语言模型使用 **REINVENT4 Mol2Mol**，提供真实接口对应的采样/RL 配置生成器和外部奖励组件；需另备 REINVENT4 环境及 Mol2Mol 权重。本项目不含预训练权重或真实配对活性数据，不能据示例结果宣称获得有效或选择性药物。

## 1. 立即运行

本次已在项目目录建立 `.venv`，使用 Python 3.10，与系统 Python 3.14 隔离。PowerShell 无需激活环境：

```powershell
.\.venv\Scripts\python.exe -m navgen generate --config configs/demo.json --output outputs/my_first_run
.\.venv\Scripts\python.exe -m pytest -q
```

首次在其他机器安装，建议 Python 3.10–3.12：

```bash
python -m venv .venv
# Windows: .venv\Scripts\python.exe；Linux: .venv/bin/python
python -m pip install -e ".[dev]"
```

上面安装命令中的 `python` 须指向新环境，也可先激活环境。REACTION 示例无需 GPU、联网或模型权重。所有配置中的文件路径均**相对于配置文件所在目录**，CLI 的 `--output` 相对于当前工作目录。已有非空输出目录会拒绝覆盖，请使用新批次目录。

## 2. 实现范围与汇报对应

| 汇报要求 | 代码 | 行为 |
|---|---|---|
| 种子附近生成、可解释转换 | `navgen/generation.py` | 酰胺逆向拆分，保留一侧种子片段，替换另一侧，再做正向酰胺偶联；逐条记录反应 SMARTS、反应物和砌块编号 |
| SMILES 化学语言模型 | `navgen/reinvent.py` | 生成 REINVENT4 Mol2Mol 采样配置，读取其 CSV；hybrid 模式同时拓展 CLM 产物的酰胺类似物 |
| 廉价过滤 | `navgen/scoring.py` | ECFP4/Morgan-2048 Tanimoto、MW、cLogP、TPSA、QED、Ertl SA、PAINS、重原子数、可旋转键数、可选 SMARTS |
| 膜性质第一层 | `navgen/scoring.py` | MW/cLogP/TPSA 双向优选区间，超出区间两侧都降分 |
| ΔΔG 差值代理 | `navgen/selectivity.py` | 随机森林直接学习配对 ΔΔG，按 Murcko 骨架划分验证集，与常数基线比较 |
| Prior–Agent RL | `navgen/reinvent.py` | 使用 REINVENT4 原有 DAP 与多样性过滤，不另写 RL；同一 Scorer 经 ExternalProcess 回传奖励 |
| 多构象和失败处理 | `navgen/conformers.py` | ETKDGv3 + MMFF94s，必要时 UFF；保存全部收敛构象及最低分子内能量构象，失败单独标记 |
| 迭代种子与探索 | `navgen/feedback.py` | 外部真值 top-k + 随机探索 + 指纹去近邻；也可显式选择 docking 初筛模式 |
| 主动学习 | `navgen/active_learning.py` | 预测均值 + β×树间标准差选样，保留多样性；域外分子可进入待测队列 |
| 候选库与评价 | `navgen/pipeline.py`, `evaluation.py` | SQLite、CSV、SMILES、JSONL、SDF，分布、骨架数、批内相似性、可选参考集新颖性 |

已有 `进展1.ipynb` 和 `clean_PDB/` 保留原样。它们是 Nav1.7 阶段的结构与对接工作，不能直接当作本模块已验证的 Nav1.8 / Nav1.5 对接设置。

## 3. 示例与筛选参数

`data/seeds.smi` 为经 PubChem CID 16038374 核对的 A-803467，来源见 `data/README.md`。示例包含 35 个酸/胺砌块。它们用于验证结构生成，未核验可采购性、反应条件或收率。

`configs/demo.json` 使用较宽松的探索阈值：Tanimoto ≥ 0.35、QED ≥ 0.35、SA ≤ 5，MW 180–550、cLogP −0.5–5、TPSA 15–120。膜性质优选区间为 MW 250–450、cLogP 1–3.5、TPSA 30–90；这是**可调的工程先验**，不代表 Nav1.8 实验推导阈值。`configs/clm.json` 则示例采用汇报提及的 QED ≥ 0.6、SA ≤ 4。

默认原种子不进入候选库。配置可增加：

```json
{
  "filters": {
    "similarity_min": 0.5,
    "qed_min": 0.6,
    "sa_max": 4.0,
    "required_smarts": "C(=O)N"
  }
}
```

这是配置片段，合并到完整配置中使用。全部默认项见 `navgen/config.py`。unknown key、反向区间、NaN 权重、没有模型却启用选择性奖励等均会报错。

SMILES 保留手性、电荷和互变异构体身份，仅规范原子排序并去除原子映射号。多片段盐和虚原子会被过滤，程序不会静默中和、去盐或猜测 pH 质子化状态。对接前仍需按团队统一条件完成质子化、互变异构体处理及 PDBQT 转换。

二维总奖励为各分项的非负加权均值，硬过滤不通过则为 0。膜区间分数在优选区间内为 1，区间外用高斯尾部衰减。分子量/重原子硬阈值及尺寸惩罚限制增大分子刷分。缺失的选择性预测保留 `null`，默认权重为 0；正权重必须指定模型。

## 4. 输出与对接交接

每个批次目录包含：

| 文件 | 用途 |
|---|---|
| `candidates.csv` / `.smi` / `.jsonl` | 通过二维筛选的分子，按奖励降序输出；JSONL 保留完整分项 |
| `molecules.sqlite` | 唯一分子、每次尝试及重复来源、三维处理状态 |
| `rejected.csv` | 合法但被过滤的唯一分子及原因；非法 SMILES 位于数据库 attempts 表 |
| `metrics.json` | 有效率、唯一率、分布直方图/分位数、骨架数、相似度、新颖性 |
| `manifest.json` | 有效配置、输入 SHA256、版本、种子、运行状态及时间 |
| `conformers.sdf` | 可选，全部成功收敛构象，携带 candidate_id / conformer_id |
| `best_conformers.sdf` | 可选，每分子一个最低分子内能量构象 |
| `conformer_status.json` | 可选，成功、嵌入失败、优化未收敛等状态；失败能量为 null |

下游以 `candidate_id` 关联评分。该编号由保留立体信息的规范 SMILES 哈希得到。对接可使用全部收敛构象；**最低分子内能量构象不等于最佳结合构象**。默认尝试生成 5 个构象，RMSD 去重后可能少于 5。二维通过但构象失败的分子仍保留在候选库中，不会悄悄消失。

## 5. 接入 REINVENT4 化学语言模型 / RL

接口依据 REINVENT4 提交 `ee0d56f4a07472bbb622cd0858184d06f11bff5d` 的 `configs/sampling.toml`、`configs/staged_learning.toml` 和 `comp_external_process.py` 核对。

先在适配 GPU/PyTorch 的独立环境安装 [REINVENT4](https://github.com/MolecularAI/REINVENT4)，准备其可信 **Mol2Mol prior**。本模块环境只承担 RDKit/代理评分，不要求安装 PyTorch。

```powershell
.\.venv\Scripts\python.exe -m navgen prepare-reinvent --config configs/demo.json --prior D:/models/mol2mol_medium_similarity.prior --output outputs/reinvent --device cuda:0 --num-smiles 1000
```

在 REINVENT4 环境运行：

```bash
reinvent /absolute/project/outputs/reinvent/sampling.toml
```

然后回到本模块环境：

```powershell
.\.venv\Scripts\python.exe -m navgen generate --config configs/clm.json
```

`hybrid` 模式先枚举原种子，再依次读取 CLM 分子并枚举其酰胺类似物；达到预算即停止。因此预算很小时可能尚未消费 CLM 输入，须通过数据库 source 列确认实际来源。只处理 CLM 原始产物时将 backend 设为 `smiles`。未经模板转换的 CLM 产物保留空 route，**不声称已有合成路线**。

RL 配置在同目录的 `staged_learning.toml`：

```bash
reinvent /absolute/project/outputs/reinvent/staged_learning.toml
```

其 ExternalProcess 通过指定 Python 执行 `-m navgen score-stdin`，stdin 每行一个 SMILES，stdout 返回 `{"version":1,"payload":{"nav18_reward":[...]}}`，保留输入顺序和长度。`prepare-reinvent --python` 可指定独立评分环境的绝对 Python 路径，该环境须安装本项目。

默认 RL 奖励只包含二维项。启用选择性后再重新生成配置；配置目录保存了当时的评分配置快照。若更换模型，建议生成新目录而非覆盖旧实验。批次多样性约束由 REINVENT4 diversity filter 承担。

**已验证配置解析和真实评分子进程协议，尚未在本项目环境执行完整 REINVENT4 采样/训练**，因为项目未提供 prior 权重。Mol2Mol 的 `num_smiles` 是每个输入种子的采样数，不能直接当作总库规模。外部 CLM 的随机性由 REINVENT4 控制，本项目的 seed 不保证其 GPU 训练确定性。

## 6. 选择性代理训练

填充 `data/paired_labels.template.csv`，每行需要：

```text
smiles,ddg_kcal_mol,species,state,protocol,label_source
```

统一符号：**ΔΔG = DG(Nav1.5) − DG(Nav1.8)，单位 kcal/mol，正值偏向 Nav1.8**。species 必须为 `Homo sapiens`，state/protocol 明确且同一训练文件只允许一个测定上下文；label_source 限 `experiment` / `free_energy`。使用前须自行核实配对标签来自匹配通道状态、测定条件和正确结合位点。将 IC50 比值转换为自由能需要额外机制假设，不能直接视为真值。

```powershell
.\.venv\Scripts\python.exe -m navgen train-selectivity --data data/paired_labels.csv --output models/nav18_ddg.joblib
```

程序采用骨架留出验证，保存 MAE、RMSE、常数基线误差、逐分子留出预测和骨架分组；随后用全部输入重训部署模型。至少 12 个唯一结构、3 类骨架只是**程序冒烟测试下限**，远不足以证明药物设计性能。没有配对真实数据时不要为启用功能捏造标签。测试目录中的合成数字仅用于软件单元测试。

在配置中匹配 state/protocol 并启用：

```json
{
  "state": "inactivated",
  "protocol": "your_matched_protocol_id",
  "selectivity": {
    "model": "../models/nav18_ddg.joblib",
    "ood_similarity_min": 0.35,
    "uncertainty_penalty": 1.0,
    "reward_scale": 2.0
  },
  "weights": {"selectivity": 0.2}
}
```

奖励使用 `sigmoid((预测均值 − λ×树间标准差) / scale)`。树间标准差是**未校准启发式**，不能当置信区间。与训练集最大 Tanimoto 低于阈值时标为域外；启用选择性奖励时该分子无法通过候选筛选，但仍能进入主动学习待测队列。上下文不匹配直接拒绝加载模型。joblib 只应加载自己训练或可信来源的文件。

该基线输入为分子指纹，按状态/协议分别建模；尚未实现口袋特征编码、通道状态 token、动力学/侧窗代理或神经网络膜渗透模型。直接学习差值是否优于其他方法仍需真实数据验证。

## 7. 下游回灌与下一轮

填充 `data/feedback.template.csv`。status 支持 `ok`、`embedding_failed`、`docking_failed`、`missing`；后三者不参与排序或训练。保持 species/state/protocol 与配置一致，禁止不同状态混排。

有配对实验/自由能标签：

```powershell
.\.venv\Scripts\python.exe -m navgen feedback --config configs/round1.json --database outputs/round1/molecules.sqlite --feedback data/round1_feedback.csv --output data/round2_seeds.smi --top-k 20
```

只有 docking 初筛结果时，label_source 填 `docking`，显式使用：

```powershell
.\.venv\Scripts\python.exe -m navgen feedback --config configs/round1.json --database outputs/round1/molecules.sqlite --feedback data/round1_feedback.csv --output data/round2_seeds.smi --ranking docking_triage
```

此模式按 `−Nav1.8 docking / 重原子数` 初筛排序，nav15_docking 仅保留供审查，**不会把双靶静态 docking 差值标成选择性真值**。上述命令中的 round1.json 需复制 demo 配置并填入实际状态、协议和输出目录；默认 unspecified 无法用于回灌。

默认 80% top-k，20% 随机探索，再执行指纹相似性上限约束；候选不足时如实输出较少分子，不靠重复补足。探索来自成功获得标签的剩余分子，不会把失败对象充作可用标签。将下一轮配置的 seeds_file 改为新文件即可再次运行 generate；原批次和回灌审计 JSON 均保留。

## 8. 主动学习与评价

先用现有模型挑少量分子送昂贵计算/实验：

```powershell
.\.venv\Scripts\python.exe -m navgen select-oracle --config configs/with_model.json --pool data/pool.smi --output data/oracle_batch.csv --count 32 --beta 1.0
```

`with_model.json` 是按第 6 节启用模型的配置。将新真值与原数据整理去重后训练**新模型文件**，更新下一轮配置，实现主动学习循环。程序不会自动启动 docking/FEP、提交算力任务或自行把预测当标签。

评价约定：validity = 可解析尝试 / 原始尝试，uniqueness = 唯一可解析分子 / 可解析尝试，二维通过率使用唯一可解析分子作分母。反应后端在提交给流水线前已做产物 sanitize，因此其 validity 不等价于未经过滤的 CLM 语法有效率。非法产物在模板层被丢弃；相应 100% 不能拿来宣称语言模型有效率。

分布采用最多 2,000 个通过分子的均匀蓄水池样本；批内 Tanimoto 最多采样 10,000 对。设置 `reference_file` 后才报告参考集结构/骨架新颖性及最近邻相似性；未提供时为 null，绝不把“不同于种子”叫作“不同于 ChEMBL”。参考集新颖性当前为内存内精确扫描，大规模 ChEMBL 应另建检索索引。

算法对比应使用相同种子、阈值、评测协议、尝试预算和最终库规模；候选不足的批次须明确报告，不能直接与大库取最优分数比较。随机 ChEMBL 基线可经 backend=smiles 走同一流水线；本项目未附 ChEMBL，因此未伪造随机分子基线结果。

尚无真实结果的指标（逆合成验证率、侧窗通过率、埋藏度 KL、双靶误差相关系数）保持 null。SA 分数和模板匹配不构成“保证可合成”；cLogP 不是 logD，也不是跨膜自由能。本模块是可运行的研究工程基础，百万级库、RL 收敛与科学增益需要更大砌块库、真实权重和后续数据实测。

## 9. 复现和开发

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m navgen --help
```

测试覆盖反应路线回放、非法 SMILES、重复计数、双向膜区间、配置校验、相同 seed 可复现、构象与失败标记、REINVENT 子进程协议、骨架无交叉划分、物种/状态限制、域外检测、主动选样以及回灌顺序与缺失值处理。

`requirements-validated.txt` 记录本次运行环境实际依赖版本。源码使用 `pyproject.toml` 声明兼容范围；需严格复现实验时同时保存模型、输入数据与 manifest 的 SHA256，并按验证环境锁定依赖。
