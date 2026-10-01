# 远程 GPU：A-803467 Mol2Mol

> 后续扩容已在同一官方 prior/独立环境完成 5×200 次采样：351 个非母体唯一结构，299 个相对旧库新增；下载后校验通过。分批脚本、增量统计和最终合并库见 [GENERATION_EXPANSION.md](GENERATION_EXPANSION.md)。下文的 100 条为首轮记录。

**已完成真实验证**：2026-10-01 独立环境依赖检查、CUDA 矩阵计算、官方权重校验和 100 条 A-803467 条件采样均通过。使用 RTX 4090，REINVENT 4.7.15、PyTorch 2.9.1/CUDA 12.8、RDKit 2026.3.6。没有启动 RL 或 docking。

本地证据位于 `outputs/remote_mol2mol_20261001/`：`environment/environment_check.json`、完整依赖锁定文件、安装日志和 `a803467_mol2mol_20261001_100/` 的采样配置、清单、原始 CSV、日志。回传 CSV 的 SHA256 与远程 manifest 一致。

100 条原始样本全部可解析，61 个唯一结构；其中严格主库 20 个、ROUND2 扩展范围 30 个、超出范围 11 个。与本地枚举/反应结果合并后是 79 个主库候选和 38 个扩展候选，其中 Mol2Mol 独有新增分别为 16 和 29 个。合并结果为 `outputs/a803467_with_mol2mol_20261001/`。这不是活性筛选结论。

本次配置请求 `randomize_smiles=true`，但官方加载器根据旧版 medium prior 的训练元数据自动设为 false；实际设置已记录在 stderr 日志中。随机种子 20260930，`unique_molecules=false`，完整保留重复输出。复现以模型元数据和实际日志为准。

2026-10-01 已实际登录截图中的实例 `ackcs-00gjhrqo`。主机 `ssh.bj8.bz1.paratera.com`，端口 2233，用户名 `root@ackcs-00gjhrqo`。此前使用的 `root` 或 `root@ackcs-00gjgs4m` 不对应当前实例。

## 连接与文件位置

本机 SSH 配置已将该实例设为 `nav18-gpu` 别名；原配置备份为 `~/.ssh/config.before_nav18_20261001`。密码不存入项目或 SSH 配置。平台网关目前仍要求密码认证，在容器内添加公钥没有使网关认证成功。

```powershell
ssh nav18-gpu
```

已验证硬件：RTX 4090，24564 MiB；驱动 580.105.08；Python 3.12.3；系统自带 PyTorch 2.7.0a0 / CUDA 12.8 可用。

项目位于挂载目录 `/root/shared-nvme/nav18/`，其挂载容量为 50 GB。持久化、释放实例后的保留与计费以平台规则为准。

| 目录 | 用途 |
|---|---|
| `env/` | 本项目独立 Python 环境，不覆盖系统 Python |
| `software/REINVENT4-v4.7/` | 官方固定提交的源代码 |
| `models/` | 原始 Mol2Mol prior 与校验来源记录 |
| `jobs/a803467_remote_job/` | 当前种子配置及采样脚本 |
| `runs/` | 每次采样的独立结果目录 |
| `logs/` | 安装日志、环境校验与依赖锁定记录 |

## 软件与模型来源

- REINVENT 官方 tag `v4.7`，源码提交 `31b90474cabbebe4a37beb3f16e5b770ec602779`（包内版本为 4.7.15）。源码下载地址：`https://codeload.github.com/MolecularAI/REINVENT4/tar.gz/31b90474cabbebe4a37beb3f16e5b770ec602779`。上传的是该归档的源码、配置及根目录文件，省略大型测试数据和示例；未修改算法源码。
- 权重来自官方 README 指向的 Zenodo 历史版本：`https://zenodo.org/records/15641297/files/mol2mol_medium_similarity.prior`，80198341 字节。
- 官方 MD5：`78b75e478bfd7dcccee8ece010bc0b07`；本地与远程均已核对。
- SHA256：`a0f40a2fb4baedaf51ae199bb38a61a8c8d26f03062977eaf06e34079e9f22ca`。
- 不是历史 `stage1.chkpt`。Zenodo 的概念 DOI 会跳转到新版本，因此复现时固定上述具体记录。

安装使用独立 `venv` 和 REINVENT 的完整必需依赖，PyTorch 2.9.1 / CUDA 12.8，不覆盖镜像的预装开发版。精确 Python 包版本保存为远程 `logs/requirements-locked.txt`。此镜像还需系统库 `libxrender1`、`libxext6`，否则 RDKit 绘图模块导入失败。

## 采样命令

在远程终端执行：

```bash
source /root/shared-nvme/nav18/env/bin/activate
cd /root/shared-nvme/nav18/jobs/a803467_remote_job
python run_series_mol2mol.py --prior /root/shared-nvme/nav18/models/mol2mol_medium_similarity.prior --device cuda:0 --preflight
python run_series_mol2mol.py --prior /root/shared-nvme/nav18/models/mol2mol_medium_similarity.prior --device cuda:0 --samples 100 --output /root/shared-nvme/nav18/runs/a803467_new_run
```

结果目录必须为新目录。只采样、不训练、不进行 docking 或活性打分；保留重复和非法原始输出。已完成首轮目录为 `runs/a803467_mol2mol_20261001_100/`，其 `manifest.json` 状态为 `completed`。新任务请使用新的结果目录。

## 返回本地

完整复制本次结果目录，不能只复制筛选后的分子。例如：

```powershell
scp -r nav18-gpu:/root/shared-nvme/nav18/runs/a803467_new_run outputs/
& ../.venv/Scripts/python.exe -m navgen.series_generation --config configs/a803467_generation.json --mol2mol-csv outputs/a803467_new_run/sampling_mol2mol.csv --output outputs/a803467_merged_new
```

合并仍按照严格核心/位点与 ROUND2 宽核心分组，没有奖励排序。Mol2Mol 的结构约束在生成后检查，不属于解码时硬约束。环境只为当前 Mol2Mol 分支配置，ShEPhERD 尚未在该环境安装。
