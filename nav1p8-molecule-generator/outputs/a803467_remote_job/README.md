# A-803467 远程 Mol2Mol 采样

这个包只负责预训练 Mol2Mol 采样，不进行训练、RL、docking 或活性打分。可在本地 CPU 或远程 GPU 的 REINVENT4 环境运行。请先确认远程操作系统、GPU/CUDA 和现有环境；本包不自动安装或覆盖环境。

## 前置条件

- 一个能够运行 REINVENT4 的独立 Python 环境，含 PyTorch 与 RDKit。
- 可信来源的原始 `mol2mol_medium_similarity.prior`。旧结果的 `stage1.chkpt` 是历史任务产物，不能直接当作原始 prior。
- 若新建环境，遵循 https://github.com/MolecularAI/REINVENT4 的对应版本安装说明，并固定版本。2026-09-30 查询到主分支要求 Python >=3.11；当前项目本地 Python 3.10 不应直接混装该主分支。ShEPhERD 的依赖要求另行处理。

## 运行

解压后进入包目录，在 REINVENT4 环境执行。将权重路径换成实际路径：

```bash
python run_series_mol2mol.py --prior /path/to/mol2mol_medium_similarity.prior --device cuda:0 --preflight
python run_series_mol2mol.py --prior /path/to/mol2mol_medium_similarity.prior --device cuda:0 --samples 100 --output mol2mol_result
```

Windows 也可运行同一 Python 命令，使用本机路径。CPU 则用 `--device cpu`。

预检检查包、权重文件和 CUDA 设备，不加载权重、不创建输出。实际采样会记录依赖版本、权重哈希、随机种子、输入种子、完整日志。保留重复和非法 SMILES；它们属于原始采样结果，不能先删去后再计算有效率。独立的后处理检查核心与位点约束，**Mol2Mol 解码本身没有强制这套核心约束**。

## 返回本地

复制完整的 `mol2mol_result/`，包含 `sampling_mol2mol.csv`、`manifest.json`、`sampling.toml`、种子和日志。不要只返回手动筛选的分子。

在项目 `nav1p8-molecule-generator` 目录执行：

```powershell
& ../.venv/Scripts/python.exe -m navgen.series_generation --config configs/a803467_generation.json --mol2mol-csv outputs/mol2mol_result/sampling_mol2mol.csv --output outputs/a803467_with_mol2mol
```

导入器逐行验证 `Input_SMILES` 必须是 A-803467；旧苯甲酰苯胺采样会被拒绝。随后和本地两种生成方法合并去重，保留所有来源与原始模型字段，分开输出严格主库和宽核心扩展库。默认每方法处理上限 200 条；若增加采样量，应同步修改配置预算，否则只处理预算内的行并标明截断。

远程真实推理尚未验证；本地已测试输入母体检查、无评分汇总与生成结果去重。若环境版本不兼容，完整日志会留在结果目录供排查。
