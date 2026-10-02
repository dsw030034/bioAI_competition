"""Package completed paired QDN redocking; representative poses are diagnostic."""
from pathlib import Path
import argparse
import csv
import shutil
import sys

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from rdkit import Chem
from scgd.runtime import code_hashes, read_json, sha256, write_json


def pack(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    report=read_json(source/'report.json')
    manifest=read_json(source/'manifest.json')
    if not report['complete'] or report['completed_jobs']!=90 or report['successful_jobs']!=90:
        raise ValueError('Require final 90/90 successful paired experiment')
    if code_hashes()!=manifest['production_code_hashes']:
        raise ValueError('Production code differs from completed experiment')
    rows=read_json(source/'results.json')
    boundaries=read_json(source/'boundary_grid_audit.json')['rows']
    if len(boundaries)!=sum(len(r['all_modes']) for r in rows):
        raise ValueError('Boundary audit is an incomplete snapshot')
    preservation=read_json(source/'preservation_evidence.json')
    for label in ['original_42_cached_jobs','retry_89_cached_jobs']:
        if not preservation[label]['sha256_and_mtime_unchanged']:
            raise ValueError('Cached artifact preservation failed')
    output.mkdir(parents=True,exist_ok=False)
    copies={}
    names=['report.json','results.json','summary.csv','jobs_summary.csv','manifest.json',
           'box_definition.json','coverage_audit.json','ring_coverage_audit.json',
           'boundary_grid_audit.json','input_integrity.json','preservation_evidence.json']
    for name in names:
        shutil.copy2(source/name,output/name)
        copies[name]={'source':str((source/name).relative_to(BASE)),'sha256':sha256(source/name)}
    samples={}
    for state in ['nav15','nav15_qdn_protonated']:
        src=source/'ligands'/state
        samples[state]=read_json(src/'report.json')
        for name in ['report.json','energy_audit.csv']:
            dst=output/(state+'_sampling_'+name)
            shutil.copy2(src/name,dst)
            copies[dst.name]={'source':str((src/name).relative_to(BASE)),'sha256':sha256(src/name)}
    summary=[]
    representatives=[]
    passing=[]
    pose_dir=output/'structures'
    pose_dir.mkdir()
    for state in samples:
        native=Chem.SDMolSupplier(str(BASE/'data/reference_validation'/f'{state}.sdf'))[0]
        Chem.MolToPDBFile(native,str(pose_dir/(state+'_experimental.pdb')))
        with Chem.SDWriter(str(pose_dir/(state+'_experimental.sdf'))) as writer:
            writer.write(native)
    for group in report['summary']:
        state,box=group['reference'],group['box']
        g=[r for r in rows if r['reference']==state and r['box']==box]
        best=min(g,key=lambda r:(r['affinity_kcal_mol'],r['conformer_id'],r['vina_seed']))
        near_row,near_mode=min(((r,m) for r in g for m in r['all_modes']),key=lambda pair:pair[1]['rmsd_A'])
        edge=[b for b in boundaries if b['reference']==state and b['box']==box and b['rank']==1]
        summary.append({'reference':state,'box':box,'expected_jobs':group['expected_jobs'],
                        'top1_pass_count':group['top1_pass_count'],
                        'any_output_pass_count':group['any_output_pass_count'],
                        'median_top1_rmsd_A':group['median_top1_rmsd_A'],
                        'median_best_of_modes_rmsd_A':group['median_best_of_modes_rmsd_A'],
                        'minimum_output_rmsd_A':near_mode['rmsd_A'],
                        'closest_pose_rank':near_mode['rank'],
                        'top1_actual_grid_edge_count':sum(b['near_actual_grid_edge_0p05A'] for b in edge),
                        'ensemble_score_selected_top1_pass_count':group['ensemble_score_selected_top1_pass_count']})
        for kind,row,rank in [('lowest_score_top1',best,1),('closest_experimental_diagnostic',near_row,near_mode['rank'])]:
            job=source/'jobs'/f"{row['reference']}_{row['conformer_id']}_{row['box']}_s{row['vina_seed']}"
            mol=list(Chem.SDMolSupplier(str(job/'poses.sdf')))[rank-1]
            name=f'{state}_{box}_{kind}'
            Chem.MolToPDBFile(mol,str(pose_dir/(name+'.pdb')))
            with Chem.SDWriter(str(pose_dir/(name+'.sdf'))) as writer:
                writer.write(mol)
            representatives.append({'reference':state,'box':box,'kind':kind,'source_job':str(job.relative_to(BASE)),
                                    'source_pose_sha256':sha256(job/'poses.sdf'),'rank':rank,
                                    'affinity_kcal_mol':row['all_modes'][rank-1]['affinity_kcal_mol'],
                                    'rmsd_A':row['all_modes'][rank-1]['rmsd_A'],
                                    'selected_for_candidate_scoring':False})
        receptor=BASE/'outputs/receptors_20261002/nav15/prepared.pdb'
        pml=f'''reinitialize
load "{receptor.as_posix()}", receptor
load "{(pose_dir/(state+'_experimental.pdb')).as_posix()}", experimental
load "{(pose_dir/(state+'_'+box+'_lowest_score_top1.pdb')).as_posix()}", lowest_score_top1
load "{(pose_dir/(state+'_'+box+'_closest_experimental_diagnostic.pdb')).as_posix()}", closest_experimental_diagnostic
hide everything
show cartoon, receptor
set cartoon_transparency, 0.65
color gray70, receptor
show sticks, experimental or lowest_score_top1 or closest_experimental_diagnostic
color green, experimental
color red, lowest_score_top1
color cyan, closest_experimental_diagnostic
select pocket, byres (receptor within 5 of experimental)
show sticks, pocket
zoom experimental or lowest_score_top1 or closest_experimental_diagnostic, 5
'''
        (pose_dir/(state+'_'+box+'_compare.pml')).write_text(pml,encoding='utf-8')
        for row in g:
            for mode in row['all_modes']:
                if mode['pose_passed']:
                    passing.append({'reference':state,'box':box,'conformer_id':row['conformer_id'],
                                    'vina_seed':row['vina_seed'],'rank':mode['rank'],
                                    'rmsd_A':mode['rmsd_A'],'contact_recall':mode['contact_recall'],
                                    'affinity_kcal_mol':mode['affinity_kcal_mol'],
                                    'score_gap_from_top1_kcal_mol':mode['affinity_kcal_mol']-row['affinity_kcal_mol']})
    write_json(output/'representative_poses.json',representatives)
    write_json(output/'passing_output_poses.json',passing)
    for name,data in [('comparison.csv',summary),('passing_output_poses.csv',passing)]:
        with (output/name).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]))
            writer.writeheader();writer.writerows(data)
    table=['| 状态 / 搜索框 | 第一名通过 | 任一输出通过 | 第一名 RMSD 中位数 | 最接近实验的输出 RMSD | 第一名贴实际边界 |',
           '|---|---:|---:|---:|---:|---:|']
    for s in summary:
        label='中性' if s['reference']=='nav15' else '+1 敏感性对照'
        n=s['expected_jobs']
        table.append(f"| {label} / {s['box']} | {s['top1_pass_count']}/{n} | {s['any_output_pass_count']}/{n} | {s['median_top1_rmsd_A']:.3f} Å | {s['minimum_output_rmsd_A']:.3f} Å | {s['top1_actual_grid_edge_count']}/{n} |")
    text=f'''# Nav1.5：去重多构象与孔道范围的对照重对接

2026-10-02。90 个唯一对接任务全部成功。**扩大构象覆盖后，+1 孔道组偶尔找回合格姿势，但第一名选择仍未解决。**
生产 `score.py`、`scgd/`、评分公式/权重、受体、姿势门槛保持不变，尚未启动 2,944 个候选的全量对接。

## 完整结果

{chr(10).join(table)}

姿势合格仍要求受体坐标系中的对称性校正重原子 RMSD≤2 Å、实验残基接触召回率≥50%、无蛋白–配体重原子间距<2 Å。
第一名成功率要求≥80%。四组均未达到该要求；跨构象每个种子按原 Vina 分数选出的赢家也均为 0/3 通过。
表中“任一输出通过”是诊断，不替代第一名门槛。

+1 孔道组的 3 个合格输出全部排第 9 名，RMSD 为 1.348–1.438 Å，接触召回 100%，没有硬碰撞；
比各自错误第一名高 0.512、0.890、0.992 kcal/mol。它们来自两个起始构象、三个随机种子，
不构成三个独立实验。分数越低越有利；正确姿势被找到却没有赢得原评分排序。
中性孔道组最近的姿势为 2.049 Å、排第 8 名，未放宽 2 Å 门槛。

## 构象与对照设计

中性和 +1 状态各从分子连接关系独立生成 4×128 个 ETKDGv3 构象；生成前删除所有实验坐标。
各 512 个构象均通过 MMFF94s 收敛、键长、碰撞及立体化学检查。按能量由低到高，以对齐后的对称性重原子 RMSD 0.75 Å 去重。
中性得到 10 个独有构象：1 个在 10 kcal/mol 同状态能量窗外，1 个超过本轮上限，选中 8 个；
+1 得到 7 个，全部选中。选中构象最小成对差异分别为 0.868 和 1.253 Å。
MMFF 能量只用于各状态内部选输入，不进入 Vina 或综合评分；没有按实验 RMSD 挑输入。

每个选中构象 × 3 个 Vina 种子 × 两个搜索框，形成 90 次全局对接。
`exhaustiveness=16`、`num_modes=9`、每任务 2 CPU，两个框使用相同构象和种子。
中心固定为 (128.691875,125.578875,137.582083) Å。
20 Å 孔道框依据实验中央腔及 Q371/F402/V405/T1417/T1708/S1710/S1759/F1760/V1763/Y1767 定义，
实验配体距框边至少 6.06 Å；受体全部保留。28 Å 对照框容纳旧的偏离姿势。
框在计算前确定，没有根据新结果调整。

20 Å 框仍允许错误位点的其他朝向，它不能完全排除该区域。大量错误第一名贴着实际网格边缘；
默认 0.375 Å 网格下，20/28 Å 请求对应 20.25/28.125 Å 实际范围。
表中贴边按实际范围余量≤0.05 Å统计，已保存坐标舍入为 0.001 Å。
因此继续缩框不能当作修复证据。28 Å 体积为 20 Å 的 2.744 倍，相同 exhaustiveness 下搜索密度更低。

本轮去重扩充主要是环间和外侧扭转；桥环自身在选中构象间的对齐 RMSD 仅约 0.002–0.056 Å，
仍集中于相近构象族。不能据此排除桥环采样不足，也不能把实验与力场几何略有不同称为已证实的化学错误。
旧基线使用 24 Å、exhaustiveness=8，且构象池和种子不同；本轮改善不能单独归因于去重。

## 下一步

先比较这 3 个合格姿势与对应错误第一名的原 Vina 分项及关键接触，保持权重；
再据结构证据测试有限孔道侧链柔性或另一个实验受体构象，每次只改变一个因素。
不继续简单堆数量、缩框或按参考 RMSD 重排后宣称通过。
此处 +1 是敏感性对照；姿势失败不能作为候选不结合或选择性好的证据。

## 查看结构和复现

`structures/` 中实验结构为 green、全组最低 Vina 分数的第一名为 red、最接近实验的输出为 cyan。
用 PyMOL 打开对应 `*_compare.pml`；也提供 PDB/SDF。
“最接近实验”的文件仅用于事后诊断，未参与原评分排名或候选选择。
完整逐作业输入、姿势、日志在本机 `outputs/nav15_conformer_redocking_20261002/`。

本轮中断前 42 项成功结果原样复用；随后 1 项超时失败用相同参数重试成功，其余 89 项姿势/日志/参数文件原样复用。
失败日志单独保留，哈希和修改时间证据在 `preservation_evidence.json`。
独立诊断脚本只补充续跑一致性检查，原脚本快照及运行/续跑哈希均保留；未改变评分或对接/评估算法。
回归检查为 **62 项测试通过，另有 16 项子测试通过**。

在 `nav18_full_pipeline` 中用新的输出目录复现：

```powershell
& ../.venv/Scripts/python.exe -u -B scripts/diagnose_nav15_conformer_redocking.py --output outputs/nav15_conformer_redocking_new
# 同一目录续跑，核验输入和已完成任务哈希
& ../.venv/Scripts/python.exe -u -B scripts/diagnose_nav15_conformer_redocking.py --output outputs/nav15_conformer_redocking_new --resume
& ../.venv/Scripts/python.exe -B scripts/audit_nav15_redocking_coverage.py outputs/nav15_conformer_redocking_new
```

本轮是同一参考配体的位点条件化对照，未验证 Nav1.5 功能风险或双靶选择性；生产筛选门槛仍未释放。
'''
    (output/'README.md').write_text(text,encoding='utf-8')
    write_json(output/'verification.json',{'successful_jobs':90,'production_code_unchanged':True,
                 'production_code_hashes':code_hashes(),'production_gate_released':False,
                 'tests':{'passed':62,'subtests_passed':16},'preservation_evidence':preservation,
                 'copied_artifacts':copies,'packager_sha256':sha256(__file__),
                 'generated_artifacts':{str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file()}})
    print(output/'README.md')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=BASE/'outputs/nav15_conformer_redocking_20261002')
    parser.add_argument('--output',type=Path,default=BASE/'results/nav15_conformer_redocking_20261002')
    args=parser.parse_args()
    pack(args.source,args.output)
