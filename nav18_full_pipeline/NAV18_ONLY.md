# Nav1.8 单靶点探索运行

本轮处理现有 2,944 个候选，采用固定的 7WE4 蛋白＋已观测截短 LPE 两性离子模型，在 A-803467 Site C 搜索。先重新生成参考输入并用新随机种子做 15 次采样复核，再计算母体＋4 个已有代表，最后处理全库。母体单独作为参照，不计入 2,944 个候选。

此前确已完成旧受体下母体＋7 个简单替换候选的对接，以及含脂质条件下母体的 15/15 匹配重对接。本次使用新的输入池和搜索种子；它是固定计算条件的采样复核，不是独立生物实验。

## 运行与续跑

在本目录 PowerShell 运行（本机 Windows）：

```powershell
& ..\.venv\Scripts\python.exe scripts\nav18_only_exploration.py --stage all --workers 6
```

中断后使用同一命令加 `--resume`。完整通过参考与小批检查后，也可直接续跑全库：

```powershell
& ..\.venv\Scripts\python.exe scripts\nav18_only_exploration.py --stage library --workers 6 --resume
```

同一运行目录有进程锁，不能重复启动。程序复核输入、版本、代码、参数和已完成产物哈希；发生变更时拒绝混用旧结果。失败任务续跑前保存旧记录，成功任务不重新对接。六个并行任务各用两个 Vina CPU 线程。运行期间电脑需要保持开机；退出终端不会停止以后台方式启动的进程，电脑关机后需要续跑。

默认产物：`outputs/nav18_only_20261002/`。

| 文件 | 用途 |
|---|---|
| `run_state.json` | 总任务状态；`finished_with_failures` 表示有计算失败待处理 |
| `reference/report.json` | 新采样复核是否具备 Nav1.8 探索资格 |
| `reference/input_conformer_diversity.json` | 五次输入池所选构象的实际相似程度 |
| `pilot_selection.json`、`pilot_status.csv` | 预评分选定代表及小批结果 |
| `input_audit_summary.json` | 全库输入审计：2,931 条可准备，13 条待明确立体化学 |
| `library_summary.json` | 全库进度与成功、暂缓、失败、姿势待审核数量 |
| `library_status.csv` | 全部 2,944 条及原始来源元数据；每完成 10 条更新 |
| `library_results.json` | 完整结果与逐姿势几何指标 |
| `library_raw_score_order.csv` | 全库结束后，首位姿势通过几何检查的原始 Vina 分数顺序 |
| `candidates/<ID的哈希>/poses.sdf` | 候选全部输出姿势，可用 PyMOL 等查看 |
| `frozen_inputs/`、`manifest.json` | 受体、脂质、参考和计算协议快照 |

CSV 的 `pose_path` 相对运行目录；同目录还有 Meeko 导出的 `poses.sdf`、输入构象和 Vina 日志。CSV 空值表示未知，JSON 保留 `null`。

## 预先固定的条件与适用范围

- 框中心 `(137.204, 136.509, 127.538)`，边长 24 Å；exhaustiveness 8，最多 9 个输出姿势。
- ETKDGv3 每个输入池尝试 20 个构象，用原有几何、立体和能量规则选一个有效低能输入。参考用五次新的输入池（准备种子 20261012–20261016），各配三个新 Vina 种子 20261012/13/14。五个输入池可能收敛至相似构象，不声称五个不同构象簇。
- 参考第一名须 RMSD ≤2 Å、原生蛋白接触召回 ≥0.5、蛋白和脂质均无 <2 Å 重原子对，并通过额外几何检查；15 次全部计算成功且至少 12 次满足标准。
- 候选采用同样准备方法和第一个固定 Vina 种子。几何检查包括结构及立体身份、内部几何、蛋白/脂质冲突和搜索框边缘。候选无需复现母体姿势，也无需比母体分数差。输出多个 mode 间的 RMSD 只描述同次搜索内差异，不能代表多次独立搜索的稳定性。
- 四代表在查看分数前从已审阅结构中固定：位点枚举主系列 `c5f6502e9184ecce`、Mol2Mol 主系列 `6f286ae3198cc668`、酰胺重组扩展系列 `8b59b199f37d9e78`、酰胺重组系列外 `017c979568cdc3ff`。它们覆盖三方法、三范围，用于技术检查，不验证全库预测准确性。
- 保留输入质子化/互变异构状态。1,464 个状态提示需要后续审核；13 个立体未指定分子显式暂缓，不随机指定异构体。新的构象准备失败单独记录，不静默丢失。

本轮固定脂质仅含实验可见的 22 个重原子，缺失尾链未重建，不模拟脂质置换、受体柔性或完整膜。原始 Vina 分数不是实验活性，参考姿势复现也不验证亲和力排序；扩展与系列外分子的适用性尤其需要后续审查。初筛结束后，应综合几何、来源与结构多样性确定复核小组，再做多种子/化学态敏感性检查。

原 `score.py`、`scgd/`、综合分权重及双靶点门槛保持不变。所有本轮结果带 `mode=nav18_only_exploration`、`nav15_risk=unknown`、`nav15_score=null`、`delta=null`、`combined_score=null`、`production_gate_released=false`。新 Nav1.8 探索资格只适用于本轮固定条件。
