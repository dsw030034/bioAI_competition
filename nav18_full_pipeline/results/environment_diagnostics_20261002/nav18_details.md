# Nav1.8：保留邻近 LPE 的匹配对照

**结果：加入实验坐标支持的截短脂质后，15 次 A-803467 重对接的第一名均通过原有姿势标准。原无脂质结果为 11/15。** 这支持“删去脂质后，错误姿势占用了原本不空的空间”这一解释。本轮没有修改综合打分公式、权重、Vina 排序方式或生产验证闸门，也没有对 2,944 个候选运行对接。

| 受体模型 | 完成 | 第一名通过 | 第一名 RMSD 中位数 | 第一名与已解析脂质重原子距离 <2 Å 的次数 |
|---|---:|---:|---:|---:|
| 既有纯蛋白基线 | 15/15 | 11/15 | 0.919 Å | 4 |
| 蛋白＋截短 LPE，两性离子净电荷 0 | 15/15 | 15/15 | 0.905 Å | 0 |
| 蛋白＋截短 LPE，CCD 酸形式净电荷 +1 敏感性对照 | 15/15 | 15/15 | 0.905 Å | 0 |

原有标准为：不对齐配体的对称校正重原子 RMSD ≤2 Å、蛋白—配体无重原子距离 <2 Å、实验接触残基恢复率 ≥0.5。另检查脂质—配体无 <2 Å 重原子距离。新的第一名 RMSD 范围为 0.871–1.002 Å，接触恢复率为 0.8–0.9。两种脂质头部状态的 15 组完整姿势 PDBQT 和对应分数均相同；这只表明本次对照结果对这项头部建模选择不敏感，不能据此测定脂质质子化状态。

原来 4 个失败的匹配任务现在第一名 RMSD 为 0.891–0.912 Å；它们与脂质的最近重原子距离从 0.713–0.812 Å 变为 5.066–5.116 Å。逐项结果见 `matched_comparison.csv`。没有按参考 RMSD 重排输出姿势，比较的始终是 Vina 原始第一名。

## 脂质模型如何建立

原结构 `receptors/7WE4.pdb` 中的 `LPE A:2008` 是 **1-O-octadecyl-sn-glycero-3-phosphocholine**，不是磷脂酰乙醇胺。CCD 含 34 个重原子，实验只给出 22 个。依据 CCD 原子名称和键连接保留全部 22 个重原子，唯一截断键为 C16–C17；未解析的 C17–C28 共 12 个碳不补猜坐标，C16 用氢封端形成甲基。截短模型相当于六碳醚链的磷酸胆碱，不应称为完整实验脂质。

主模型采用季铵 N+ 和去质子化磷酸 O32−，总电荷 0。CCD 本身采用 O32–H、总电荷 +1，因此额外完整重复了 15 次酸形式敏感性对照。这两个状态都是明确声明的模型；本轮没有进行 pKa 预测。标准 Vina 不以 Gasteiger 部分电荷作为显式静电项，状态检查主要体现供氢原子和原子类型的变化。

用 RDKit 按 CCD 建图，只优化新增氢的位置；Meeko 分型后将 ATOM 行加入刚性受体。所有 22 个已解析重原子坐标完全不变，原蛋白 PDBQT 行原样保留；脂质没有 ROOT/BRANCH/TORSDOF，季铵氮无氢键供体 H，也未被标为受体 NA。添加前的几何检查中，脂质—蛋白最近重原子距离为 3.210 Å，脂质—实验 A-803467 为 4.589 Å，无 <2 Å 接触。审计见各状态的 `preparation_audit.json`。

## 匹配设置与复现

直接复用 `reference_validation_20261002_iofix/ligands/nav18_c0..c4.pdbqt` 的五个输入，不重新生成构象；每个输入使用种子 20261002、20261003、20261004。与基线相同的 Vina 1.2.7、中心 (137.204, 136.509, 127.538)、24 Å 立方体、exhaustiveness=8、num_modes=9、cpu=2；两个任务并行。仅超时保护设置为 1,200 秒；实际单次主模型耗时中位数约 19.8 秒。

从工作区根目录执行（已有输出用 `--resume`，新运行用新 `--output`）：

```powershell
.\.venv\Scripts\python.exe -B nav18_full_pipeline/scripts/diagnose_nav18_lipid.py --resume
.\.venv\Scripts\python.exe -B nav18_full_pipeline/scripts/diagnose_nav18_lipid.py --output nav18_full_pipeline/outputs/nav18_lipid_control_20261002_head_sensitivity --states ccd_acid_plus1 --resume
.\.venv\Scripts\python.exe -B nav18_full_pipeline/scripts/test_diagnose_nav18_lipid.py
```

3 项化学及数据保真测试通过：状态电荷/供氢/截断端价态，坐标和受体原始行不变及原子序号唯一，异常缺失重原子被拒绝。`manifest.json` 保存输入、Vina、代码、依赖版本与校验值；每项任务的 `poses.job.json`、`poses.log.json` 保存参数、状态、标准输出和耗时。`LPE_ccd.cif` 与 `ccd_provenance.json` 保存外部 CCD 来源及校验值。

## 本轮结论的范围

这是有匹配基线的原因检验，使用了曾用于发现问题的输入和随机种子，**不能计作独立留出验证**。脂质固定且被截短，未描述完整膜环境、脂质运动或较大候选引起的脂质置换。下一步可用新的配体初始构象和随机种子检验 Nav1.8 稳定性。Nav1.5 的诊断和双靶点闸门仍独立保留；本轮结果不等于验证了亲和力、选择性或批准全量筛选。

主要文件：`report.json`、`summary.csv`、`matched_comparison.csv`、`head_state_sensitivity_comparison.json`；酸形式完整结果位于相邻目录 `../nav18_lipid_control_20261002_head_sensitivity/`。
