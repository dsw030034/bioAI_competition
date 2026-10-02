"""Audit unrelaxed A-803467 projection in two experimentally resolved receptors.

All existing inputs are read-only. 9V3S protein is rigidly transformed with the
already recorded pore CA fit. No optimization, docking, scoring or graph edit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import gemmi
import numpy as np

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from scgd.runtime import code_hashes

AMINO=set('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL MSE'.split())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protein_atoms(path):
    """Author atom identifiers, first model, standard amino-acid heavy atoms."""
    st=gemmi.read_structure(str(path)); items=[]
    for chain in st[0]:
        for residue in chain:
            if residue.name not in AMINO:
                continue
            for atom in residue:
                if atom.element.name in ('H','D'):
                    continue
                identifier=f'{chain.name}:{residue.seqid}:{residue.name}:{atom.name}'
                items.append({'identifier':identifier,'chain':chain.name,'residue_number':residue.seqid.num,
                              'residue':f'{chain.name}:{residue.seqid}:{residue.name}',
                              'residue_name':residue.name,'atom_name':atom.name,'xyz':list(atom.pos)})
    ids=[item['identifier'] for item in items]
    if not items or len(ids)!=len(set(ids)):
        raise ValueError('Empty or duplicate protein atom identifiers')
    return items


def original_coordinate_audit(prepared,observed):
    """Check original observed atom positions and explicitly list additions."""
    by_id={row['identifier']:row for row in prepared}
    absent=[]; displacement=[]; renamed=[]; used=set()
    for row in observed:
        match=by_id.get(row['identifier'])
        # Meeko can permute terminal carboxyl O/OXT labels. Accept only an
        # exact coordinate match in the same residue and explicitly audit it.
        if row['atom_name'] in ('O','OXT') and (match is None or np.linalg.norm(np.asarray(row['xyz'])-match['xyz'])>1e-8):
            alternatives=[r for r in prepared if r['residue']==row['residue'] and r['atom_name'] in ('O','OXT')
                          and np.linalg.norm(np.asarray(row['xyz'])-r['xyz'])<1e-8]
            if len(alternatives)==1:
                match=alternatives[0]
                renamed.append({'observed_atom':row['identifier'],'prepared_atom':match['identifier'],
                                'reason':'Terminal O/OXT name permutation; exact coordinate retained'})
        if match is None:
            absent.append(row['identifier'])
        else:
            if match['identifier'] in used:
                raise ValueError('Observed-to-prepared mapping is not one-to-one')
            used.add(match['identifier'])
            displacement.append(float(np.linalg.norm(np.asarray(row['xyz'])-match['xyz'])))
    return {'observed_heavy_atoms':len(observed),'prepared_heavy_atoms':len(prepared),
            'matched_observed_heavy_atoms':len(displacement),'missing_original_atoms':absent,
            'terminal_atom_name_permutations':renamed,
            'prepared_atoms_without_original_coordinate_mapping':[r['identifier'] for r in prepared if r['identifier'] not in used],
            'maximum_original_coordinate_displacement_A':max(displacement,default=None)}


def clash_audit(ligand,ligand_names,protein,protein_xyz=None):
    xyz=np.asarray([item['xyz'] for item in protein] if protein_xyz is None else protein_xyz,float)
    distance=np.linalg.norm(np.asarray(ligand,float)[:,None,:]-xyz[None,:,:],axis=2)
    pair_indices=np.argwhere(distance<2.)
    pairs=[{'ligand_heavy_index_zero_based':int(i),'ligand_atom_name':ligand_names[i],
            'protein_atom':protein[j]['identifier'],'protein_residue':protein[j]['residue'],
            'distance_A':float(distance[i,j])} for i,j in pair_indices]
    minimum_index=np.unravel_index(int(distance.argmin()),distance.shape)
    i,j=minimum_index
    counts=Counter(pair['protein_residue'] for pair in pairs)
    return {'ligand_heavy_atoms':len(ligand),'protein_heavy_atoms':len(protein),
            'minimum_heavy_distance_A':float(distance.min()),'heavy_atom_pairs_below_2A':len(pairs),
            'clashing_ligand_atoms':len({p['ligand_heavy_index_zero_based'] for p in pairs}),
            'clashing_protein_atoms':len({p['protein_atom'] for p in pairs}),
            'residue_pair_counts_below_2A':dict(sorted(counts.items())),
            'closest_pair':{'ligand_heavy_index_zero_based':int(i),'ligand_atom_name':ligand_names[i],
                            'protein_atom':protein[j]['identifier'],'distance_A':float(distance[i,j])},
            'pairs_below_2A':sorted(pairs,key=lambda p:p['distance_A'])}


def rigid_transform(xyz,fit):
    rotation=np.asarray(fit['rotation_row_vector'],float)
    translation=np.asarray(fit['translation_A'],float)
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-10) or abs(np.linalg.det(rotation)-1)>1e-10:
        raise ValueError('Recorded transformation is not a proper rigid rotation')
    return np.asarray(xyz,float)@rotation+translation


def run(args):
    output=args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refuse to overwrite {output}')
    alignment=args.alignment.resolve()
    path_6lqa=BASE/'outputs/receptors_20261002/nav15/prepared.pdb'
    path_9v3s=BASE/'outputs/nav15_multireference_20261002/human_9v3s/receptor/prepared.pdb'
    observed_9v3s=BASE/'outputs/nav15_multireference_20261002/human_9v3s/protein_observed.pdb'
    raw_7we4=BASE/'receptors/7WE4.pdb'
    sources=[alignment,path_6lqa,path_9v3s,observed_9v3s,raw_7we4]
    hashes={str(path):sha(path) for path in sources};frozen=code_hashes()
    report=json.loads(alignment.read_text('utf-8'))
    deposited=next(r for c in gemmi.read_structure(str(raw_7we4))[0] for r in c if r.name=='95T' and r.seqid.num==2004)
    heavy=[a for a in deposited if a.element.name not in ('H','D')]
    names=[a.name for a in heavy]
    projected=np.asarray(report['mapped_sites']['nav18_7we4_pore']['xyz'],float)
    recomputed=rigid_transform([list(a.pos) for a in heavy],report['alignments']['nav18_7we4']['fits']['pore'])
    ligand_difference=float(np.max(np.abs(recomputed-projected)))
    if ligand_difference>1e-10:
        raise ValueError('A803 projected coordinates do not match source rigid transform')
    protein_6lqa=protein_atoms(path_6lqa);protein_9v3s=protein_atoms(path_9v3s)
    preparation_audit=original_coordinate_audit(protein_9v3s,protein_atoms(observed_9v3s))
    if preparation_audit['maximum_original_coordinate_displacement_A']>1e-8:
        raise ValueError('9V3S preparation changed original heavy coordinates')
    fit9=report['alignments']['human_9v3s']['fits']['pore']
    mapped_9v3s=rigid_transform([p['xyz'] for p in protein_9v3s],fit9)
    audits={'6LQA_QDN_state':clash_audit(projected,names,protein_6lqa),
            '9V3S_azo_state_pore_mapped':clash_audit(projected,names,protein_9v3s,mapped_9v3s)}
    result={'diagnostic_only':True,'pose_role':'Unrelaxed homologous A-803467 coordinate projection; not experimental Nav1.5 binding',
            'ligand':'7WE4 A:2004:95T / A-803467, all 25 observed heavy atoms',
            'frame':'6LQA author chain B frame','ligand_coordinates_identical_for_both_receptors':True,
            'recomputed_ligand_vs_alignment_max_coordinate_difference_A':ligand_difference,
            '9V3S_preparation_coordinate_audit':preparation_audit,
            '9V3S_rigid_protein_transform':{'convention':'xyz_target=xyz_source @ rotation + translation',
                                        'rotation_row_vector':fit9['rotation_row_vector'],
                                        'translation_A':fit9['translation_A'],
                                        'pore_CA_pairs':fit9['pairs'],'pore_CA_rmsd_A':fit9['ca_rmsd_A']},
            'geometry_comparison':audits,'optimization_performed':False,'docking_performed':False,
            'scoring_performed':False,'source_sha256':hashes,
            'production_code_unchanged':code_hashes()==frozen,
            'limitations':['A803 is an experimentally observed Nav1.8 ligand, not a confirmed Nav1.5 ligand.',
                           'CA alignment does not guarantee side-chain or ligand compatibility; these are unrelaxed geometric projections.',
                           'A <2A pair is a diagnostic geometric overlap, not an inactivity or selectivity prediction.',
                           '9V3S is a different human Nav1.5 construct and ligand-bound state, not a controlled single-residue perturbation.',
                           '9V3S A1EQU chemical graph remains unresolved; this audit uses its protein coordinates only.',
                           'No side-chain optimization, flexibility, local docking or protein motion was applied.']}
    if not result['production_code_unchanged'] or any(sha(path)!=hashes[str(path)] for path in sources):
        raise ValueError('Source or production code changed during audit')
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    rows=[{'receptor':key,**pair} for key,audit in audits.items() for pair in audit['pairs_below_2A']]
    with (output/'clashing_pairs.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['receptor','ligand_heavy_index_zero_based','ligand_atom_name','protein_atom','protein_residue','distance_A'])
        writer.writeheader();writer.writerows(rows)
    md=['# A-803467 同源投影与受体侧链容纳审计','',
        '以同一份 7WE4 A-803467 的全部 25 个实验重原子，经已记录的孔道 CA 配准投影到 6LQA 坐标系；分别检查 6LQA 与同样映射到该坐标系的 9V3S prepared 蛋白。计算只包含重原子几何距离，未做优化、对接或评分。','',
        '| 受体 | 最近蛋白—配体重原子距离 | <2 Å 原子对 | 涉及残基（原编号） |',
        '|---|---:|---:|---|']
    for key,audit in audits.items():
        residues=', '.join(f'{res} ({count} 对)' for res,count in audit['residue_pair_counts_below_2A'].items()) or '无'
        md.append(f"| {key} | {audit['minimum_heavy_distance_A']:.4f} Å | {audit['heavy_atom_pairs_below_2A']} | {residues} |")
    md += ['',f"9V3S 孔道配准使用 {fit9['pairs']} 对 CA，RMSD {fit9['ca_rmsd_A']:.3f} Å。prepared 文件中全部可对应的原实验重原子最大位移为 {preparation_audit['maximum_original_coordinate_displacement_A']:.6f} Å；额外重原子和原子编号差异在 JSON 中列明。",'',
           '这些结果说明，孔道骨架对齐后，原 Nav1.8 配体的直接投影仍可能与 Nav1.5 侧链发生明显空间重叠。CA 对齐可以支持区域假设，不能证明目标侧链已经能容纳该姿势。','',
           '不同受体状态的重叠程度可为后续有限柔性或状态对照提供线索，但不能据此判断 A-803467 对 Nav1.5 无活性，或候选选择性较好；也不能将 9V3S 视为已经验证的 A-803467 结合结构。','',
           '9V3S 的 A1EQU 化学图未验；本审计只使用其蛋白重原子坐标，不涉及该配体化学图。完整原子对、来源 SHA256、刚体矩阵和原代码保持检查见 `report.json` / `manifest.json`。']
    (output/'README.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    manifest={'script_sha256':sha(__file__),'source_sha256':hashes,'production_code_hashes':frozen,
              'production_code_unchanged':result['production_code_unchanged'],
              'artifact_sha256':{p.name:sha(p) for p in output.iterdir() if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:{k:v for k,v in audit.items() if k!='pairs_below_2A'} for key,audit in audits.items()},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment',type=Path,default=BASE/'outputs/nav15_site_alignment_v2_20261002/report.json')
    parser.add_argument('--output',type=Path,default=BASE/'results/nav15_multireference_20261002/projection')
    run(parser.parse_args())
