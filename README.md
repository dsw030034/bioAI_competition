# Nav1.8 分子生成项目

2026-10-03 更新：**2,944 个候选的 Nav1.8 单靶点对接已处理完成：2,931 个成功，13 个因立体化学未指定暂缓；1,818 个首位姿势通过几何检查，1,113 个需要框边界复核。** 没有计算失败。固定含脂质条件下，A-803467 新采样参考复核 15/15 通过，RMSD 中位数 0.916 Å。

优先阅读 [本轮正式结果报告](nav18_full_pipeline/results/nav18_only_20261002/RESULTS_REPORT.md)、[Nav1.8 运行与续跑说明](nav18_full_pipeline/NAV18_ONLY.md)。原始分数不代表实验活性；Nav1.5 风险保持未知，原双靶点打分门槛未释放。

当前生成代码目录为 **nav1p8-molecule-generator/**；新增对接与打分代码位于 **nav18_full_pipeline/**。

| 目录 | 内容 |
|---|---|
| [nav1p8-molecule-generator](nav1p8-molecule-generator/) | 当前 Git 项目、源码、配置、测试、结构数据和最新实验结果 |
| [nav18_full_pipeline](nav18_full_pipeline/) | 对接、原双靶点打分框架与验证记录；已完成单独的 Nav1.8 全库探索，Nav1.5 定量排名仍暂停。见 [正式结果报告](nav18_full_pipeline/results/nav18_only_20261002/RESULTS_REPORT.md) |
| [docs](docs/) | 课题汇报、技术选型、参赛方案和调研资料 |
| [archive](archive/) | 旧版本代码、旧示例结果及可再生成缓存，供追溯 |
| `.venv/` | 保留的 Python 环境，已重新绑定当前子项目 |

优先阅读 [第二轮运行说明](nav1p8-molecule-generator/ROUND2.md)。

```powershell
Set-Location nav1p8-molecule-generator
& ../.venv/Scripts/python.exe -m navgen --help
& ../.venv/Scripts/python.exe -m pytest -q
```

2026-09-30 整理：根目录旧 navgen/configs/data/tests/outputs 与旧安装说明移入 archive/legacy_20260918，原研究文档集中到 docs。当前子项目的结构、Vina、实验产物及 Git 历史均未移动。归档内历史绝对路径仅作记录，旧代码不再作为默认运行入口。
