# 参考配体重对接结果

2026-10-02：45 次对接全部执行成功；原打分框架、归一化与权重保持不变。

| 参考 | 第一名通过 / 总数 | 第一名 RMSD 中位数 |
|---|---:|---:|
| A-803467 / Nav1.8 | 11/15 | 0.919 Å |
| 奎尼丁 / Nav1.5 | 0/15 | 10.178 Å |
| 奎尼丁 +1 / Nav1.5，敏感性对照 | 0/15 | 9.948 Å |

验证门槛未通过，未开展 2,944 候选的全量对接。姿势验证不能替代活性或选择性验证。
完整修复、方法、限制与运行命令见 [验证说明](../../DOCKING_VALIDATION.md)。

- `report.json`、`summary.csv`：协议、输入哈希和最终结论。
- `results.json`：每次任务的各姿势分数、RMSD、碰撞与接触恢复。
- `input_audit_summary.json`、`stereochemistry_review.csv`：2,944 个输入审查，以及 13 个立体化学待审候选。
- `*_receptor_audit.json`、`receptor_coordinate_check.json`：修补、排除、原子分型和实验坐标核查。
- `test_verification.json`：36 项测试通过。
- `resume_verification.json`：105 个配体、姿势、日志文件的哈希与修改时间均保持不变，无重复对接。
- `candidate_gate_verification.json`：验证未通过时，候选入口在创建计算目录前拒绝运行。
- `checkpoint_recovery.json`：仅修复 Windows 汇总文件占用后的报告恢复追溯。

完整姿势与受体位于本机 `outputs/`；该目录默认不进 Git。
本摘要位于新增的 `nav18_full_pipeline/` 中，尚未合并或上传到原 GitHub 生成仓库。
