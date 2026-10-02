# 完整项目共享与环境恢复

本次发布包含整个 `bioAI_competition` 项目根目录，并保留既有生成仓库的提交历史。生成模块位于 `nav1p8-molecule-generator/`，对接模块位于 `nav18_full_pipeline/`；请从对应子目录运行命令。完整项目仓库为 [dsw030034/bioAI_competition](https://github.com/dsw030034/bioAI_competition)。

上传范围包括项目代码、文档、研究资料、历史归档、受体与参考配体、生成结果、全部对接姿势、日志、审计和模型文件。原来被 `.gitignore` 忽略的研究产物在本次发布中显式纳入，不仅上传筛选后的候选。

以下内容不进入版本库：本机 `.venv/`、嵌套 `.git/`、Python/pytest 缓存、可重建的 egg-info 以及临时运行锁。它们属于本机环境或版本管理元数据，不是实验记录。发布清单记录每个包含文件的大小与 SHA256，并列出排除的类别及数量。

超过 50 MiB 的模型/依赖压缩包由 Git LFS 保存。完整下载需要安装 [Git LFS](https://git-lfs.com/)，然后运行：

```powershell
git lfs install
git clone https://github.com/dsw030034/bioAI_competition.git bioAI_competition
Set-Location bioAI_competition
git lfs pull
```

已有旧布局的队友建议重新 clone 到新文件夹，避免将本地未跟踪输出和新的完整目录混在一起。原工作目录可以保留作备份。

## 计算环境

本轮实测 Windows + Python 3.10，RDKit 2026.3.6、Meeko 0.8.0、Vina 1.2.7、gemmi 0.7.5、PDBFixer 1.12.0、OpenMM 8.6.1、NumPy 2.2.6。详细冻结版本和代码哈希见 `nav18_full_pipeline/outputs/nav18_only_20261002/manifest.json`。

可先按生成模块的安装说明创建新的 `.venv`。PDBFixer/OpenMM 的平台安装以原项目说明与冻结版本为准；`.venv` 不可直接从别人的电脑拷贝复用。实际运行前核对依赖及 Vina 版本；版本不同应建立新运行目录，不能将其混入旧哈希验证的续跑。

## 本轮结果入口

- [正式结果报告](nav18_full_pipeline/results/nav18_only_20261002/RESULTS_REPORT.md)
- [2,944个候选状态](nav18_full_pipeline/outputs/nav18_only_20261002/library_status.csv)
- [原始姿势、完整审计与运行说明](nav18_full_pipeline/NAV18_ONLY.md)
- [完成产物完整性审计](nav18_full_pipeline/results/nav18_only_20261002/completion_audit.json)

计算产物中的旧绝对路径是历史溯源信息；仓库内相对目录保持一致。分析时优先使用清单的相对路径和 SHA256，不因换电脑而修改原始实验记录。
