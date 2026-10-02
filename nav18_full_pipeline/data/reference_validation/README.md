# 参考坐标与打分公式基线

- `nav18.sdf`：7WE4 中 `95T:A:2004` 的 A-803467 实验结合态重原子坐标。
- `nav15.sdf`：6LQA 中 `QDN:B:2110` 的奎尼丁实验结合态重原子坐标；分子图和碳立体化学来自 CCD。
  中性叔胺 N 的立体标记不作为验证条件，候选分子不使用这个例外。
- `nav15_qdn_protonated.sdf`：上一参考的叔胺 +1 质子化敏感性对照，保持重原子坐标。
  不是另一个实验结构，也不是 pKa 预测。
- `QDN_ccd.json`：之前保存的 RCSB CCD 响应，来源 `https://data.rcsb.org/rest/v1/core/chemcomp/QDN`。
- `scoring_contract.json`：修改前 `score.py` 的六个打分/描述符函数 AST SHA-256；用于回归验证。

三个 SDF 从本工作区主项目 `nav1p8-molecule-generator/outputs/docking_validation/receptors_v2/`
的相应 `reference.sdf` 复制。程序每次运行会核对所有参考重原子坐标与当前原始 PDB 配体
一一对应，容差 0.002 Å，并记录文件哈希。A-803467 的二维示意图不用于姿势真值。

配体准备与几何工具适配自主项目 `navgen/ligand_preparation.py`、`navgen/docking_validation.py`；
受体修补适配自 `scripts/prepare_docking_validation.py`。新入口不依赖同名 `navgen` 包的导入顺序。
