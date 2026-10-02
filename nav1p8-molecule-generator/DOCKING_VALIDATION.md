# Docking 验证与复现

## 当前结论（2026-09-30）

当前双靶刚性 Vina 基线未通过原生配体重对接验证，暂不用于新增候选的可靠排序或 REINVENT 奖励。程序成功返回分数，不等于找回正确结合姿势。

首轮验证为每个靶点 5 个独立 ETKDG 初始构象 × 3 个 Vina 随机种子，`exhaustiveness=8`，共 30 次。每次只将第一名用于主验证；同时记录所有输出姿势的 RMSD，区分“搜索到但未排第一”和“没有搜索到”。

| 验证对象 | 运行成功 | 第一名通过姿势检查 | 第一名 RMSD 中位数 |
|---|---:|---:|---:|
| 7WE4 / A-803467 | 15/15 | 5/15 | 7.84 Å |
| 6LQA / 中性奎尼丁 QDN | 15/15 | 0/15 | 10.17 Å |

Nav1.8 每次输出的多个姿势中均能找到约 0.9 Å 的近原生姿势，但第一名仅 5 次正确。这说明最低 Vina 分数目前不是可靠的姿势选择依据。Nav1.5 中性参考状态本身存在质子化限制，另设叔胺质子化 +1 对照；Nav1.8 另设 `exhaustiveness=32` 复查。敏感性检查结果见下方补充。

2026-09-26 的 8 分子 × 2 靶点历史原始文件后来已由用户补入本地 `outputs/a803467_nav18_nav15_round1/docking_runs/20260926_174205/`，两个受体 PDBQT 也已提供；不再属于原始文件缺失。`outputs/` 被 Git 忽略，因此这些完整历史姿势和受体不会随代码克隆。本轮验证摘要已单独发布在 [results/docking_validation_20260930](results/docking_validation_20260930/README.md)。

已完成的敏感性检查仍未通过门槛：Nav1.8 `exhaustiveness=32` 为 3/9 第一名通过、RMSD 中位数 10.67 Å；Nav1.5 质子化 QDN +1 为 0/9 通过、中位数 10.16 Å。对应报告见上述共享目录。新增候选的打分仍处于暂缓状态。

## 这次修复了什么

- `scripts/clean_nav18_chain_a.py` / `clean_nav15_chain_b.py` 只提取链并选择一致的 altloc，保留缺原子的残基、链名、编号和插入码；不再按原子完整性删除残基或连续重编号。
- `scripts/prepare_docking_validation.py` 使用 PDBFixer 修补缺失重原子，不构建缺失环区；校验原有重原子坐标和标识符不变。保存原始结构哈希、缺原子清单、距离及处理日志。
- 7WE4 的 16 个侧链不完整残基距参考配体至少 31.26 Å；6LQA 目标链无此类缺失。修补后部分远端区域仍触发 Meeko 键连接/模板错误；当前 Nav1.8 成功制备中显式排除了 7 个残基，其所有重原子均在搜索框外 27 Å 以上。完整编号与距离见 `audit.json`。这不是把局部修补结构当作完整精修蛋白。
- Meeko 出错时仅允许显式排除**每个原子均距搜索框超过 8 Å**的残基，逐项记录；口袋附近出现同类问题立即失败，不使用全局忽略错误选项。
- 使用当前项目 Python，直接调用 Meeko API；不依赖原作者的 `~/miniconda3/envs/vina_clean`。保留 Vina 可执行文件哈希、依赖版本、输入哈希、每次命令、标准输出、错误输出、PDBQT 和 SDF 姿势。
- 每个构象独立生成并要求 MMFF 收敛。单个配体准备/运行失败会写入记录和失败分母，不会被静默跳过。

## 验证口径与边界

RMSD 使用 RDKit `CalcRMS`，对重原子进行对称性校正，**不平移或旋转配体**。参考来自沉积结构的结合态坐标；7WE4 的商业二维 SDF 不作姿势真值。6LQA 的 QDN 由 CCD 分子图赋键级，核验碳立体中心；中性叔胺 N 的立体标记按 RDKit 支持范围处理，并记录限制。

程序门槛是：全部任务执行成功，至少 80% 的第一名姿势同时满足 RMSD ≤2 Å、没有蛋白—配体重原子间距 <2 Å 的原子对、4 Å 残基接触召回率 ≥50%。这是本项目的初步质量门槛，不是活性预测有效性的证明；简单距离接触也不等同于氢键或完整相互作用分析。

模型仍是刚性、纯蛋白、单受体构象基线。7WE4 有距参考配体约 4.59 Å 的 LPE 和 5.93 Å 的胆固醇，本轮均未显式纳入，因此不能排除脂质环境对姿势排序的影响。受体采用 Meeko 模板质子化；尚未开展完整 pKa/互变异构体枚举、受体状态匹配、膜环境或结合动力学验证。双靶 docking 分差不得解释为实验选择性或真实 ΔΔG。

## 运行方式

在本子项目目录用 PowerShell 执行，Python 位于父目录 `.venv`：

```powershell
& ../.venv/Scripts/python.exe -m pip install -e '.[docking]'
& ../.venv/Scripts/python.exe scripts/prepare_docking_validation.py
& ../.venv/Scripts/python.exe scripts/run_validated_docking.py --stage reference
```

受体输出默认 `outputs/docking_validation/receptors_v2/`。若已有目录，准备脚本拒绝覆盖；`--resume` 仅复用哈希一致的完成结果。更换原始结构或参数时应使用新的 `--output`，并修改配置中的 `receptors`。

每次验证在 `outputs/docking_validation/runs/<时间戳>_reference/` 新建独立目录。`manifest.json` 为输入与环境记录；`results.json` 为逐次结果；`summary.csv` 和 `report.json` 为汇总。

敏感性检查：

```powershell
& ../.venv/Scripts/python.exe scripts/run_validated_docking.py --config configs/docking_validation_nav18_ex32.json
& ../.venv/Scripts/python.exe scripts/prepare_qdn_protonation_control.py
& ../.venv/Scripts/python.exe scripts/run_validated_docking.py --config configs/docking_validation_nav15_qdn_plus1.json
```

新增的 17 个候选来自 `outputs/round2_reaction/evaluation_final/candidates.csv`。后续使用同一配置和通过门槛的报告运行：

```powershell
& ../.venv/Scripts/python.exe scripts/run_validated_docking.py --stage candidates --validation-report '<通过验证的 report.json>'
```

程序会核对受体、参考坐标、搜索框、Vina 哈希以及关键搜索参数。未通过门槛时直接拒绝候选对接。候选默认 3 个初始构象 × 3 个随机种子，报告分数中位数、范围、标准差及碰撞检查，保留参考对照。当前没有通过门槛的双靶报告，因此这 17 个候选尚未运行本轮对接。

下一步应优先评估保留邻近脂质的受体方案、受体状态与口袋定义，以及独立的姿势重评分或有依据的相互作用约束；用额外参考配体验证，避免仅对 A-803467 调参。旧 `batch_dock_generated_candidates.py` 保留作历史流程，默认配置不是本轮经过验证的入口。
