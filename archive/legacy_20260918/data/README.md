种子来自 PubChem PUG REST：
https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/A-803467/property/CanonicalSMILES/JSON
查询日期 2026-09-18，CID 16038374。保留源返回的结构，运行时由 RDKit 标准化 SMILES。

building_blocks.csv 是用于验证程序的人工整理结构集，不是采购目录，也未核验库存、价格或具体反应条件。包含 A-803467 原有两侧片段，重建原种子会被默认过滤器排除。示例库仅覆盖酰胺替换，不能代表完整药化搜索空间。

两个 template.csv 只有列名，需填入真实数据。**没有提供虚构的活性、选择性或 docking 标签。**
ddg_kcal_mol 统一定义为 DG(Nav1.5) − DG(Nav1.8)，正值偏向 Nav1.8；来源仅限 experiment/free_energy。
若从匹配条件下的 Kd 转换，应使用 RT·ln(Kd15/Kd18) 并记录温度；IC50 比值不应未经测定机制检查直接等同于结合自由能。
