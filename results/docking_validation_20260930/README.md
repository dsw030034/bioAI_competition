# 原生配体重对接验证摘要

以下为已运行的原始报告和汇总，保存输入哈希与验证门槛。完整受体、逐次 pose 和 Vina 日志仍保留在原本地 `outputs/`，本目录只共享较小的验证摘要。

| 实验 | 第一名姿势通过 | 第一名 RMSD 中位数 | 原始报告 |
|---|---|---|---|
| Nav1.8 / A-803467，exhaustiveness=8 | 5/15 | 7.84 Å | [report](20260930_185318_604070_reference/report.json) |
| Nav1.5 / 中性 QDN，exhaustiveness=8 | 0/15 | 10.17 Å | [report](20260930_185318_604070_reference/report.json) |
| Nav1.8 / A-803467，exhaustiveness=32 | 3/9 | 10.67 Å | [report](20260930_190901_499283_reference/report.json) |
| Nav1.5 / QDN +1，exhaustiveness=8 | 0/9 | 10.16 Å | [report](20260930_191002_893902_reference/report.json) |

均未通过本项目 >=80% 的参考验证门槛，不用于新增候选的可靠排序或 RL 奖励。分数不解释为真实选择性或结合自由能。定义、限制和重新验证命令见 [DOCKING_VALIDATION.md](../../DOCKING_VALIDATION.md)。
