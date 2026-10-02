"""Map final multireference top1 poses into 6LQA for spatial comparison only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import gemmi
import numpy as np
from rdkit import Chem

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE));sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_nav15_projection_clashes import protein_atoms,rigid_transform
from scgd.runtime import code_hashes


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contacts(xyz,protein,identifiers):
    distance=np.linalg.norm(np.asarray(xyz)[:,None,:]-protein[None,:,:],axis=2)
    return set(identifiers[np.any(distance<=4,axis=0)])


def spatial_comparison(xyz,site_xyz,protein,identifiers):
    a,b=contacts(xyz,protein,identifiers),contacts(site_xyz,protein,identifiers)
    return {'centroid_distance_A':float(np.linalg.norm(np.mean(xyz,axis=0)-np.mean(site_xyz,axis=0))),
            'minimum_ligand_heavy_distance_A':float(np.linalg.norm(np.asarray(xyz)[:,None,:]-np.asarray(site_xyz)[None,:,:],axis=2).min()),
            'contact_jaccard_4A':len(a&b)/max(1,len(a|b)),
            'shared_target_contact_residues_4A':sorted(a&b)}


def distribution(values):
    values=np.asarray(values,float)
    return {'minimum':float(values.min()),'median':float(np.median(values)),'maximum':float(values.max())}


def run(args):
    output=args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refuse to overwrite {output}')
    alignment=args.alignment.resolve();root=args.references.resolve()
    alignment_report=json.loads(alignment.read_text('utf-8'))
    states=('rat_6uz0','rat_8f6p');data={};source_paths=[alignment]
    for state in states:
        folder=root/state/'redocking'
        source_report=json.loads((folder/'report.json').read_text('utf-8'))
        rows=json.loads((folder/'results.json').read_text('utf-8'))
        if not source_report.get('complete') or source_report['successful_jobs']!=source_report['expected_jobs']:
            raise ValueError(f'Refuse partial/unsuccessful redocking: {state}')
        if len(rows)!=source_report['expected_jobs'] or any(r['status']!='ok' for r in rows):
            raise ValueError(f'Report/results mismatch: {state}')
        source_paths.extend([folder/'report.json',folder/'results.json']);data[state]=rows
    target_path=BASE/'outputs/receptors_20261002/nav15/prepared.pdb'
    source_paths.append(target_path)
    atom_rows=protein_atoms(target_path)
    protein=np.asarray([r['xyz'] for r in atom_rows]);ids=np.asarray([r['residue'] for r in atom_rows])
    native=next(r for c in gemmi.read_structure(str(BASE/'receptors/6LQA.pdb'))[0] for r in c if r.name=='QDN' and r.seqid.num==2110)
    source_paths.append(BASE/'receptors/6LQA.pdb')
    sites={'QDN_native':np.asarray([list(a.pos) for a in native if a.element.name not in ('H','D')]),
           'flecainide_6UZ0_mapped':np.asarray(alignment_report['mapped_sites']['rat_6uz0_pore']['xyz']),
           'ranolazine_8F6P_mapped':np.asarray(alignment_report['mapped_sites']['rat_8f6p_pore']['xyz']),
           'A803467_7WE4_mapped':np.asarray(alignment_report['mapped_sites']['nav18_7we4_pore']['xyz'])}
    frozen=code_hashes();rows_out=[];molecules=[]
    for state,rows in data.items():
        transform=alignment_report['alignments'][state]['fits']['pore']
        for row in rows:
            pose_path=Path(row['pose']).with_suffix('.sdf')
            source_paths.append(pose_path)
            original=Chem.SDMolSupplier(str(pose_path),removeHs=True)[0]
            if original is None:
                raise ValueError(f'Invalid saved top1 SDF: {pose_path}')
            mol=Chem.Mol(original);xyz=rigid_transform(original.GetConformer().GetPositions(),transform)
            for atom_index,point in enumerate(xyz):
                mol.GetConformer().SetAtomPosition(atom_index,point)
            identifier=f"{state}_{row['conformer_id']}_s{row['vina_seed']}_top1"
            distances=np.linalg.norm(xyz[:,None,:]-protein[None,:,:],axis=2)
            comparisons={site:spatial_comparison(xyz,coords,protein,ids) for site,coords in sites.items()}
            rows_out.append({'pose_id':identifier,'reference':state,'conformer_id':row['conformer_id'],
                             'vina_seed':row['vina_seed'],'rank':1,'original_affinity_kcal_mol':row['affinity_kcal_mol'],
                             'original_same_ligand_rmsd_A':row['top1_rmsd_A'],'original_pose_passed':row['top1_passed'],
                             'original_source_pose_sdf':str(pose_path),
                             'original_actual_grid_face_margin_A':row['modes'][0]['actual_grid_face_margin_A'],
                             'centroid_xyz_6LQA_frame_A':xyz.mean(0).tolist(),
                             'target_6LQA_contact_residues_4A':sorted(contacts(xyz,protein,ids)),
                             'target_6LQA_minimum_heavy_distance_A':float(distances.min()),
                             'target_6LQA_heavy_pairs_below_2A':int(np.sum(distances<2)),
                             'known_region_comparisons':comparisons,
                             'regions_ordered_by_centroid_distance':sorted(comparisons,key=lambda s:comparisons[s]['centroid_distance_A']),
                             'region_order_is_descriptive_only':True})
            molecules.append((identifier,mol))
    summaries={}
    for state in states:
        group=[r for r in rows_out if r['reference']==state]
        centers=np.asarray([r['centroid_xyz_6LQA_frame_A'] for r in group])
        pairwise=np.linalg.norm(centers[:,None,:]-centers[None,:,:],axis=2)
        summaries[state]={'poses':len(group),'centroid_pairwise_distance_A':distribution(pairwise[np.triu_indices(len(group),1)]),
                          'original_actual_grid_face_margin_A':distribution([r['original_actual_grid_face_margin_A'] for r in group]),
                          'original_top1_grid_margin_below_0p05A_count':sum(r['original_actual_grid_face_margin_A']<.05 for r in group),
                          'original_top1_grid_margin_below_0p5A_count':sum(r['original_actual_grid_face_margin_A']<.5 for r in group),
                          'region_distance_and_contact_distributions':{
                              site:{'centroid_distance_A':distribution([r['known_region_comparisons'][site]['centroid_distance_A'] for r in group]),
                                    'contact_jaccard_4A':distribution([r['known_region_comparisons'][site]['contact_jaccard_4A'] for r in group])}
                              for site in sites},
                          'centroid_nearest_region_counts_descriptive_only':{site:sum(r['regions_ordered_by_centroid_distance'][0]==site for r in group) for site in sites}}
    hashes={str(path):sha(path) for path in source_paths}
    if code_hashes()!=frozen:
        raise ValueError('Production code changed during site comparison')
    output.mkdir(parents=True);(output/'poses').mkdir()
    for identifier,mol in molecules:
        Chem.MolToPDBFile(mol,str(output/'poses'/f'{identifier}.pdb'))
        with Chem.SDWriter(str(output/'poses'/f'{identifier}.sdf')) as writer:
            writer.write(mol)
    report={'diagnostic_only':True,'poses':rows_out,'summary':summaries,
            'pore_rigid_transforms':{state:alignment_report['alignments'][state]['fits']['pore'] for state in states},
            'known_site_coordinates':{k:v.tolist() for k,v in sites.items()},
            'site_contact_definition':'All comparisons use 6LQA prepared protein heavy atoms and <=4A residue contacts after the already-recorded pore CA transform. These are not the original reference receptor contacts.',
            'selection_policy':'All final top1 poses from each self-redocking; no native-based selection or hard region threshold.',
            'chemical_and_receptor_states_separate':True,'production_code_unchanged':True,
            'limitations':['Different ligands have no atomwise RMSD comparison.',
                           'Nearest centroid rank is descriptive; it does not classify a binding site or establish an unknown-region boundary.',
                           'Rigidly transferred poses may clash in 6LQA. Target-frame contact similarity is a location comparison, not a validated bound pose.',
                           'A boundary-adjacent top1 is not evidence of stable binding in an alternate region. Original actual-grid face margins are reported.',
                           'A803467 is a homologous Nav1.8 mapping hypothesis; native Nav1.5 binding is not established.',
                           'No optimization, score changes, new Vina, gate release or selectivity claim.']}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    csvrows=[]
    for row in rows_out:
        for region,comparison in row['known_region_comparisons'].items():
            csvrows.append({'pose_id':row['pose_id'],'reference':row['reference'],'region':region,
                            **{k:v for k,v in comparison.items() if not isinstance(v,list)}})
    with (output/'comparisons.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    md=['# 多参考自对接第一名在同一受体坐标系中的位置','',
        '将 6UZ0 flecainide 与 8F6P ranolazine 各 15 份自对接第一名，按先前固定的孔道 CA 矩阵映射到 6LQA。保留全部 30 份姿势，比较与 QDN、flecainide、ranolazine 实验位置，以及 A-803467 同源映射位置的距离和接触；没有按实验姿势挑选结果。','',
        '| 自对接来源 | 对照区域 | 质心距离 min / median / max | 接触 Jaccard min / median / max |',
        '|---|---|---:|---:|']
    for state,summary in summaries.items():
        for region,values in summary['region_distance_and_contact_distributions'].items():
            d,j=values['centroid_distance_A'],values['contact_jaccard_4A']
            md.append(f"| {state} | {region} | {d['minimum']:.3f} / {d['median']:.3f} / {d['maximum']:.3f} Å | {j['minimum']:.3f} / {j['median']:.3f} / {j['maximum']:.3f} |")
    md += ['','| 来源 | 原实际网格余量 min / median / max | 距实际边缘 <0.05 Å | <0.5 Å |',
           '|---|---:|---:|---:|']
    for state,summary in summaries.items():
        b=summary['original_actual_grid_face_margin_A']
        md.append(f"| {state} | {b['minimum']:.3f} / {b['median']:.3f} / {b['maximum']:.3f} Å | {summary['original_top1_grid_margin_below_0p05A_count']}/{summary['poses']} | {summary['original_top1_grid_margin_below_0p5A_count']}/{summary['poses']} |")
    md += ['','接触统一在 6LQA prepared 蛋白上计算，不是各参考受体的原实验接触。原自对接同分子的 RMSD / 成功状态单独保留，不与跨分子位置距离混用。','',
           '接近搜索框实际网格边缘的第一名，不能当作某个替代位点的稳定结合证据。余量采用原参考受体自对接的实际网格记录；映射到 6LQA 后没有重新评分。','',
           'JSON 给出每份姿势到四区域的连续距离、接触相似度、质心排序和两两质心距离分布。排序只描述“更近哪个已知区域”；未设置硬阈值，不据此判定结合位点、未知区域或功能风险。','',
           '全部变换后的姿势保存于 `poses/`（PDB + SDF）。这是刚体映射的几何对照，未在 6LQA 中重新优化，可能与该受体侧链重叠。A-803467 映射仍是同源区域假设。','',
           '原 Vina 分数在 JSON 中保留来源值，不将两种分子、不同受体的分数合并成亲和力排名。原生产框架与门槛保持不变。']
    (output/'README.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    manifest={'script_sha256':sha(__file__),'source_sha256':hashes,'production_code_hashes':frozen,
              'production_code_unchanged':code_hashes()==frozen,
              'artifact_sha256':{str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(output/'README.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment',type=Path,default=BASE/'outputs/nav15_site_alignment_v2_20261002/report.json')
    parser.add_argument('--references',type=Path,default=BASE/'outputs/nav15_multireference_20261002')
    parser.add_argument('--output',type=Path,default=BASE/'results/nav15_multireference_20261002/redocking_sites')
    run(parser.parse_args())
