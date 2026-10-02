"""Sequence-mapped global/pore-CA alignment of known sodium-channel sites.

Read-only on existing project artifacts; writes a new diagnostic directory.
Different ligands are compared by centroid, contacts, and nearest distance,
never by atomwise ligand RMSD. No docking or scoring is run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

import gemmi
import numpy as np
from rdkit import Chem

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from scgd.runtime import code_hashes

# Human SCN5A Q14524, annotated S5, intramembrane pore, S6 for repeats I-IV.
PORE_SEGMENTS = ((250,272),(358,378),(387,413),(839,860),(885,903),(913,941),
                 (1331,1355),(1401,1422),(1446,1470),(1651,1679),(1698,1714),(1746,1771))
AA = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
      'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
      'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def download(url,path):
    if not path.exists():
        with urllib.request.urlopen(url,timeout=60) as response:
            path.write_bytes(response.read())
    return path


def chain_ca(chain):
    return [(res, res.get_ca()) for res in chain
            if res.name in AA and res.get_ca() is not None]


def correspondence(source,target):
    """Return explicit observed-residue pairs in source/target order."""
    s, t = chain_ca(source), chain_ca(target)
    sq, tq = ''.join(AA[r.name] for r,a in s), ''.join(AA[r.name] for r,a in t)
    scoring = gemmi.AlignmentScoring()
    scoring.match, scoring.mismatch = 3,-2
    scoring.gapo, scoring.gape = -8,-1
    result = gemmi.align_string_sequences(list(sq),list(tq),[],scoring)
    sa, ta = result.add_gaps(sq,1),result.add_gaps(tq,2)
    pairs, si, ti = [],0,0
    for sc,tc in zip(sa,ta):
        if sc!='-' and tc!='-':
            sr, atom_s = s[si]; tr, atom_t = t[ti]
            number = tr.seqid.num
            pairs.append({'source_residue':f'{source.name}:{sr.seqid}:{sr.name}',
                          'target_residue':f'{target.name}:{tr.seqid}:{tr.name}',
                          'target_number':number, 'identity':sc==tc,
                          'pore_segment':any(lo<=number<=hi for lo,hi in PORE_SEGMENTS),
                          'source_xyz':list(atom_s.pos), 'target_xyz':list(atom_t.pos)})
        si += sc!='-'; ti += tc!='-'
    return pairs, {'score':result.score,'aligned_pairs':len(pairs),
                   'identical_pairs':sum(p['identity'] for p in pairs),
                   'identity_fraction':sum(p['identity'] for p in pairs)/len(pairs),
                   'source_sequence':sq,'target_sequence':tq,
                   'aligned_source':sa,'aligned_target':ta,'cigar':result.cigar_str()}


def fit(source,target):
    """Row-vector Kabsch: transformed = source @ rotation + translation."""
    s,t=np.asarray(source,float),np.asarray(target,float)
    if len(s)<3 or s.shape!=t.shape:
        raise ValueError('At least three equally-sized CA pairs required')
    cs,ct=s.mean(0),t.mean(0)
    u,_,vt=np.linalg.svd((s-cs).T@(t-ct))
    correction=np.eye(3); correction[2,2]=np.linalg.det(u@vt)
    rotation=u@correction@vt
    translation=ct-cs@rotation
    error=np.linalg.norm(s@rotation+translation-t,axis=1)
    return rotation,translation,{'pairs':len(s),'ca_rmsd_A':float(np.sqrt(np.mean(error**2))),
                                 'median_pair_distance_A':float(np.median(error)),
                                 'max_pair_distance_A':float(error.max())}


def heavy_positions(residue):
    return np.asarray([list(a.pos) for a in residue if a.element.name not in ('H','D')],float)


def target_heavy(chain):
    atoms=[]; identifiers=[]
    for residue in chain:
        if residue.name not in AA:
            continue
        for atom in residue:
            if atom.element.name not in ('H','D'):
                atoms.append(list(atom.pos))
                identifiers.append(f'{chain.name}:{residue.seqid}:{residue.name}')
    return np.asarray(atoms,float),np.asarray(identifiers)


def contacts(xyz,protein,ids):
    distance=np.linalg.norm(np.asarray(xyz)[:,None,:]-protein[None,:,:],axis=2)
    return sorted(set(ids[np.any(distance<=4.0,axis=0)]))


def compare(left,right,protein,ids):
    a,b=np.asarray(left),np.asarray(right)
    ac,bc=set(contacts(a,protein,ids)),set(contacts(b,protein,ids))
    return {'centroid_distance_A':float(np.linalg.norm(a.mean(0)-b.mean(0))),
            'minimum_heavy_atom_distance_A':float(np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2).min()),
            'contact_overlap_count_4A':len(ac&bc),'contact_union_count_4A':len(ac|bc),
            'contact_jaccard_4A':len(ac&bc)/max(1,len(ac|bc)),
            'shared_target_contact_residues_4A':sorted(ac&bc)}


def write_residue_pdb(path,residue,rotation=None,translation=None):
    st=gemmi.Structure(); model=gemmi.Model('1'); chain=gemmi.Chain('L')
    copy=residue.clone()
    if rotation is not None:
        for atom in copy:
            xyz=np.asarray(list(atom.pos))@rotation+translation
            atom.pos=gemmi.Position(*xyz)
    chain.add_residue(copy); model.add_chain(chain); st.add_model(model)
    st.write_pdb(str(path))


def box_coverage(xyz,center,requested):
    """Saved-coordinate geometry, not an inference of Vina boundary energy."""
    xyz,center=np.asarray(xyz,float),np.asarray(center,float)
    actual=float(np.ceil(requested/.375)*.375)
    def one(extent):
        margins=extent/2-np.abs(xyz-center)
        return {'minimum_heavy_atom_face_margin_A':float(margins.min()),
                'heavy_atoms_outside':int(np.any(margins<0,axis=1).sum()),
                'whole_heavy_ligand_inside':bool(np.all(margins>=0))}
    return {'requested_extent_A':requested,'actual_grid_extent_A':actual,
            'nominal':one(requested),'actual_grid':one(actual)}


def supplement_box_coverage(folder):
    """Add fresh supplement files, preserving all existing alignment artifacts."""
    folder=Path(folder).resolve()
    paths=[folder/name for name in ('box_coverage.json','site_box_hypotheses.json','box_coverage.md')]
    if any(path.exists() for path in paths):
        raise FileExistsError('Refuse to overwrite an existing box-coverage supplement')
    report=json.loads((folder/'report.json').read_text('utf-8'))
    center=np.array([128.691875,125.578875,137.582083])
    coverage={'original_qdn_center_xyz_A':center.tolist(),
              'method':'Heavy-coordinate face margins. Actual Vina default extent is ceil(size/0.375)*0.375 with force_even_voxels=False; these are not scores or boundary penalties.',
              'mapped_sites':{}}
    hypotheses={'diagnostic_only':True,'automatically_applied':False,
                'method':'Center on mapped observed-ligand centroid. Suggested cubic extent is max(20A, maximum ligand extent+12A), for at least 6A nominal padding; receptor state/site validation still required.',
                'site_hypotheses':{}}
    lines=['# 原 QDN 搜索框对其他实验配体位置的覆盖','',
           '按已映射的全部配体重原子检查原 QDN 中心的 20 / 24 / 28 Å 立方框。名义尺寸与 Vina 默认网格实际取整尺寸分开；几何越界不能直接反推出评分惩罚。','',
           '| 配体映射（孔道配准） | 框尺寸 | 名义最小余量 / 框外重原子 | 实际网格最小余量 / 框外重原子 |',
           '|---|---:|---:|---:|']
    for name,site in report['mapped_sites'].items():
        xyz=np.asarray(site['xyz'])
        coverage['mapped_sites'][name]={str(size):box_coverage(xyz,center,size) for size in (20.,24.,28.)}
        if site['fit_scope']=='pore':
            for size,item in coverage['mapped_sites'][name].items():
                n,a=item['nominal'],item['actual_grid']
                lines.append(f"| {name} | {size} Å | {n['minimum_heavy_atom_face_margin_A']:.3f} Å / {n['heavy_atoms_outside']} | {a['minimum_heavy_atom_face_margin_A']:.3f} Å / {a['heavy_atoms_outside']} |")
            centroid=xyz.mean(0)
            # A centroid can differ from a bounding-box center. Use exact
            # maximum radial axis distance to guarantee the stated padding.
            extent=max(20.,float(2*np.abs(xyz-centroid).max()+12))
            hypotheses['site_hypotheses'][name]={
                'source':site['source'],'ligand_centroid_xyz_A':centroid.tolist(),
                'observed_heavy_bbox_min_xyz_A':xyz.min(0).tolist(),
                'observed_heavy_bbox_max_xyz_A':xyz.max(0).tolist(),
                'suggested_cubic_extent_A':extent,'minimum_padding_A':6.,
                'source_fit':'pore','source_CA_rmsd_A':report['alignments'][site['source']]['fits']['pore']['ca_rmsd_A'],
                'interpretation':'Transferred experimental coordinates define a site hypothesis; not a validated Nav1.5 pose or a production docking box.'}
    lines += ['','20 Å 搜索范围是以 QDN 为中心的位点条件；若它截断另一种已知配体的位置，不能将候选在该框中的不利表现解释成 Nav1.5 风险低。',
              '','`site_box_hypotheses.json` 保存后续可审核的位点中心坐标和边界余量建议。它没有被应用于任何对接或生产配置，也没有释放筛选门槛。9V3S 仅使用未修改的实验原子坐标，化学图仍未验证。']
    write_json(paths[0],coverage);write_json(paths[1],hypotheses)
    paths[2].write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(folder/'box_coverage_manifest.json',{'script_sha256':digest(__file__),
                'source_report_sha256':digest(folder/'report.json'),
                'output_sha256':{path.name:digest(path) for path in paths},
                'production_code_hashes':code_hashes(),'no_docking_performed':True})
    print(paths[0],flush=True)


def package_output(folder,destination):
    """Copy a self-contained review package with portable relative PML paths."""
    folder,destination=Path(folder).resolve(),Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'Refuse to overwrite {destination}')
    report=json.loads((folder/'report.json').read_text('utf-8'))
    destination.mkdir(parents=True)
    for path in folder.iterdir():
        if path.is_file() and path.suffix in ('.json','.csv','.md'):
            shutil.copy2(path,destination/path.name)
    shutil.copytree(folder/'structures',destination/'structures')
    shutil.copy2(BASE/'receptors/6LQA.pdb',destination/'structures/nav15_target.pdb')
    text=(folder/'compare_sites.pml').read_text('utf-8')
    text=text.replace((BASE/'receptors/6LQA.pdb').as_posix(),'structures/nav15_target.pdb')
    text=text.replace((folder/'structures').as_posix()+'/','structures/')
    (destination/'compare_sites.pml').write_text(text,encoding='utf-8')
    export_audits={}
    for name,site in report['mapped_sites'].items():
        st=gemmi.read_structure(str(destination/'structures'/f'{name}.pdb'))
        xyz=np.asarray([list(a.pos) for m in st for c in m for r in c for a in r if a.element.name not in ('H','D')])
        if len(xyz)!=len(site['xyz']):
            raise ValueError(f'PDB export heavy count changed: {name}')
        error=float(np.max(np.abs(xyz-np.asarray(site['xyz']))))
        if error>.000501:
            raise ValueError(f'PDB export geometry changed: {name}')
        export_audits[name]={'heavy_count':len(xyz),'maximum_coordinate_formatting_difference_A':error}
    original=(destination/'README.md').read_text('utf-8')
    summary=['','## 对位点问题的解释','',
             '宽框中性 / +1 第一名分别距 6UZ0 flecainide 的孔道配准位点质心 2.202 / 2.173 Å；全局配准为 2.248 / 2.217 Å，结论一致。映射配体在 6LQA 中的接触 Jaccard 均为 0.769（10/13）；使用源 6UZ0 实验接触映射到 6LQA 后为 0.643。',
             '',
             '因此，“偏离 QDN 实验姿势”不能等同于“落入没有意义的地方”。这一区域与另一种已知 Nav1.5 阻断剂 flecainide 的实验区域明显相邻。该结果支持多位点核查；不能证明 QDN 本身真实结合在那里，更不能用作候选选择性良好的证据。',
             '',
             '中性 20 Å 组按原分数选出的第一名位于另一位置，距 flecainide 映射质心 8.882 Å；+1 的 20 Å 第一名为 1.889 Å。三份通过原 QDN 姿势条件的 rank9 都是回顾性诊断样例；生产排序仍由原 Vina 分数决定。',
             '',
             '8F6P ranolazine 的孔道配准 CA RMSD 为 0.831 Å，位点质心距 QDN 为 2.563 Å；宽框第一名距它约 7.55–7.59 Å。7WE4 A-803467 同源映射质心距 QDN 为 4.585 Å；它是跨亚型的位置假设，不能直接视为 Nav1.5 已知配体。',
             '',
             '6UZ0 关键对应包括 human F934 ↔ rat F937、human F1459 ↔ rat F1461。全部双向对应、源原始接触和目标受体上的接触保存在 JSON 中。',
             '',
             '以 QDN 中心设置的名义 20 Å 框会将映射 A-803467 的 1 个和 flecainide 的 3 个重原子截在框外；实际 20.25 Å 网格仍分别有 1 个、1 个在外。原 24 Å 框可以覆盖，但最小余量只有 1.648 / 0.718 Å。详见 `box_coverage.md`。',
             '',
             '## 查看结构','',
             '本分享包包含受体、全部映射 PDB 和 QDN 代表 SDF。先在 PyMOL 将当前目录切到本 `alignment` 文件夹，再执行 `@compare_sites.pml`。PML 使用相对路径，可随整个文件夹移动。',
             '',
             '`human_9v3s_*` PDB 只表示原实验配体的 38 个重原子位置；A1EQU 的 CCD 化学图未验证，不能把这些文件当作已验证的化学 SDF 或重对接输入。没有为 A1EQU 输出 SDF。',
             '',
             '`package_manifest.json` 记录本包全部文件 SHA256 和 PDB 坐标格式化误差。独立数学、序列对应和网格取整检查共 5 项通过。']
    (destination/'README.md').write_text(original+'\n'.join(summary)+'\n',encoding='utf-8')
    verification={'alignment_and_grid_tests_passed':5,'mapped_pdb_export_audits':export_audits,
                  'production_code_unchanged':report['production_code_unchanged'],
                  'source_report_sha256':digest(folder/'report.json'),
                  'no_additional_docking_performed':True}
    write_json(destination/'package_verification.json',verification)
    write_json(destination/'package_manifest.json',{
        'source_directory':str(folder),'packaging_script_sha256':digest(__file__),
        'artifact_sha256':{str(p.relative_to(destination)):digest(p) for p in destination.rglob('*') if p.is_file()},
        'original_source_hashes':report['source_sha256'],
        'production_code_hashes':code_hashes(),'production_gate_released':False})
    print(destination/'README.md',flush=True)


def main(args):
    output=args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refuse to overwrite {output}')
    output.mkdir(parents=True); (output/'source').mkdir(); (output/'structures').mkdir()
    frozen=code_hashes()
    target_path=BASE/'receptors/6LQA.pdb'
    target=gemmi.read_structure(str(target_path))[0]['B']
    protein,protein_ids=target_heavy(target)
    native=next(r for r in target if r.name=='QDN' and r.seqid.num==2110)
    native_xyz=heavy_positions(native)
    write_residue_pdb(output/'structures/qdn_native.pdb',native)
    # Save exact topology evidence; ensure fixed segment definition agrees.
    topology_path=download('https://rest.uniprot.org/uniprotkb/Q14524.json',output/'source/Q14524.json')
    topology=json.loads(topology_path.read_text('utf-8'))
    annotated=[(f['location']['start']['value'],f['location']['end']['value'])
               for f in topology['features'] if (f['type']=='Intramembrane' and f.get('description')=='Pore-forming')
               or (f['type']=='Transmembrane' and any(f'Name={s} of repeat' in f.get('description','') for s in ('S5','S6')))]
    if sorted(annotated)!=sorted(PORE_SEGMENTS):
        raise ValueError('UniProt topology differs from fixed local-fit definition')
    sources={'nav18_7we4':(BASE/'receptors/7WE4.pdb','A','95T'),
             'rat_6uz0':(download('https://files.rcsb.org/download/6UZ0.cif',output/'source/6UZ0.cif'),None,'K4D'),
             'human_9v3s':(download('https://files.rcsb.org/download/9V3S.cif',output/'source/9V3S.cif'),None,'A1EQU')}
    if args.include_8f6p:
        sources['rat_8f6p']=(download('https://files.rcsb.org/download/8F6P.cif',output/'source/8F6P.cif'),None,'XHO')
    sites={}; alignments={}; source_hashes={str(target_path):digest(target_path)}
    for name,(path,chain_name,ligand_name) in sources.items():
        st=gemmi.read_structure(str(path)); model=st[0]
        chain=model[chain_name] if chain_name else max(model,key=lambda c:len(chain_ca(c)))
        ligands=[r for c in model for r in c if r.name==ligand_name]
        if not ligands:
            raise ValueError(f'No {ligand_name} in {path}; inspect source ligand identifier')
        ligand=ligands[0]
        pairs,sequence=correspondence(chain,target)
        source_protein,source_ids=target_heavy(chain)
        source_contacts=contacts(heavy_positions(ligand),source_protein,source_ids)
        source_to_target={p['source_residue']:p['target_residue'] for p in pairs}
        target_to_source={p['target_residue']:p['source_residue'] for p in pairs}
        mapped_source_contacts=sorted({source_to_target[c] for c in source_contacts if c in source_to_target})
        alignments[name]={'source_chain':chain.name,'ligand':f'{ligand.name}:{ligand.seqid}',
                          'sequence_alignment':sequence,'fits':{},
                          'raw_ligand_heavy_atoms':len(heavy_positions(ligand)),
                          'raw_source_protein_contact_residues_4A':source_contacts,
                          'raw_contacts_mapped_to_target_ids':mapped_source_contacts,
                          'raw_contacts_without_ca_correspondence':[c for c in source_contacts if c not in source_to_target],
                          'ligand_chemical_graph_validated':False if name=='human_9v3s' else None,
                          'ligand_geometry_only':True,
                          'chemical_caveat':'9V3S A1EQU CCD graph is disputed (single N-N vs experimental cis-azo). Raw atom coordinates only; no validated ligand SDF or graph is inferred.' if name=='human_9v3s' else None}
        write_json(output/f'{name}_residue_correspondence.json',pairs)
        source_hashes[str(path)]=digest(path)
        for scope in ('global','pore'):
            used=[p for p in pairs if scope=='global' or p['pore_segment']]
            rotation,translation,audit=fit([p['source_xyz'] for p in used],[p['target_xyz'] for p in used])
            xyz=heavy_positions(ligand)@rotation+translation
            label=f'{name}_{scope}'
            sites[label]={'xyz':xyz.tolist(),'contact_residues_4A':contacts(xyz,protein,protein_ids),
                          'fit_scope':scope,'source':name,'diagnostic_only':True,
                          'source_to_target_residue_correspondence':source_to_target,
                          'target_to_source_residue_correspondence':target_to_source,
                          'raw_contacts_mapped_to_target_ids':mapped_source_contacts}
            alignments[name]['fits'][scope]={**audit,'rotation_row_vector':rotation.tolist(),
                                            'translation_A':translation.tolist(),
                                            'identical_pairs':sum(p['identity'] for p in used),
                                            'qdn_comparison':compare(xyz,native_xyz,protein,protein_ids)}
            write_residue_pdb(output/'structures'/f'{label}.pdb',ligand,rotation,translation)
        global_xyz=np.asarray(sites[f'{name}_global']['xyz']); local_xyz=np.asarray(sites[f'{name}_pore']['xyz'])
        alignments[name]['global_to_pore_ligand_centroid_shift_A']=float(np.linalg.norm(global_xyz.mean(0)-local_xyz.mean(0)))
    dockdir=args.docking.resolve(); report=json.loads((dockdir/'report.json').read_text('utf-8'))
    if not report.get('complete') or report['successful_jobs']!=report['expected_jobs']:
        raise ValueError('Docking comparison requires final successful report')
    rows=json.loads((dockdir/'results.json').read_text('utf-8')); selected=[]
    for state in ('nav15','nav15_qdn_protonated'):
        for box in ('pore20','wide28'):
            group=[r for r in rows if r['reference']==state and r['box']==box and r['status']=='ok']
            winner=min(group,key=lambda r:r['affinity_kcal_mol'])
            selected.append((f'{state}_{box}_score_top1',winner,1,False))
    for row in rows:
        for mode in row.get('all_modes',[]):
            if mode['pose_passed']:
                selected.append((f"passing_{row['reference']}_{row['conformer_id']}_s{row['vina_seed']}_r{mode['rank']}",row,mode['rank'],True))
    docksites={}
    for label,row,rank,diagnostic in selected:
        job=dockdir/'jobs'/f"{row['reference']}_{row['conformer_id']}_{row['box']}_s{row['vina_seed']}"
        mol=list(Chem.SDMolSupplier(str(job/'poses.sdf'),removeHs=True))[rank-1]
        if mol is None:
            raise ValueError('Invalid saved pose SDF')
        xyz=mol.GetConformer().GetPositions(); mode=row['all_modes'][rank-1]
        Chem.MolToPDBFile(mol,str(output/'structures'/f'{label}.pdb'))
        with Chem.SDWriter(str(output/'structures'/f'{label}.sdf')) as writer:
            writer.write(mol)
        docksites[label]={'reference':row['reference'],'box':row['box'],'conformer_id':row['conformer_id'],
                          'vina_seed':row['vina_seed'],'rank':rank,'affinity_kcal_mol':mode['affinity_kcal_mol'],
                          'rmsd_to_qdn_A':mode['rmsd_A'],'passed':mode['pose_passed'],
                          'diagnostic_selection_by_native_geometry':diagnostic,
                          'qdn_comparison':compare(xyz,native_xyz,protein,protein_ids),
                          'known_sites':{key:compare(xyz,site['xyz'],protein,protein_ids) for key,site in sites.items()}}
        dock_contacts=set(contacts(xyz,protein,protein_ids))
        for key,comparison in docksites[label]['known_sites'].items():
            site=sites[key]
            comparison['shared_target_contacts_with_source_numbering']=[
                {'target_residue':contact,'source_residue':site['target_to_source_residue_correspondence'].get(contact)}
                for contact in comparison['shared_target_contact_residues_4A']]
            experimental=set(site['raw_contacts_mapped_to_target_ids'])
            common=sorted(dock_contacts & experimental)
            comparison['source_experimental_contact_overlap_count_4A']=len(common)
            comparison['source_experimental_contact_jaccard_4A']=len(common)/max(1,len(dock_contacts|experimental))
            comparison['source_experimental_shared_contacts']=[
                {'target_residue':contact,'source_residue':site['target_to_source_residue_correspondence'][contact]}
                for contact in common]
    final={'diagnostic_only':True,'target':'6LQA chain B','local_fit_segments_Q14524':PORE_SEGMENTS,
           'fit_policy':'Sequence-mapped observed CA pairs; all pairs retained, no spatial cherry-picking or outlier trimming. Local fit uses target S5/pore/S6 sequence segments.',
           'alignments':alignments,'mapped_sites':sites,'docking_comparisons':docksites,
           'limitations':['Homologous mapping is a spatial hypothesis, not proof of binding in the target.',
                          'Rat 6UZ0/8F6P vs human 6LQA and 7WE4 Nav1.8 vs Nav1.5 differ in species, sequence, construct and receptor conformation.',
                          'A1EQU in 9V3S is a different ligand; QDN-like rank9 poses are retrospective diagnostic examples.',
                          '9V3S ligand mapping uses raw 38-heavy-atom coordinates only. Its CCD chemical graph is unresolved; no validated A1EQU SDF or redocking claim is made.',
                          'Different molecules have no atomwise ligand RMSD; centroid/nearest-atom/contact overlap do not establish affinity or selectivity.',
                          'Observed-chain sequence alignment includes missing loops; explicit correspondences and global/local fit sensitivity are reported.'],
           'production_code_unchanged':code_hashes()==frozen,
           'source_sha256':{**source_hashes,str(topology_path):digest(topology_path),str(dockdir/'results.json'):digest(dockdir/'results.json')}}
    write_json(output/'report.json',final)
    write_json(output/'manifest.json',{'script_sha256':digest(__file__),'gemmi':gemmi.__version__,
                                     'production_code_hashes':frozen,'source_sha256':final['source_sha256']})
    csvrows=[]
    for label,item in docksites.items():
        for site,comparison in item['known_sites'].items():
            csvrows.append({'docking_pose':label,'known_site':site,**{k:v for k,v in comparison.items() if not isinstance(v,list)}})
    with (output/'comparisons.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    lines=['reinitialize',f'load "{target_path.as_posix()}", nav15','hide everything','show cartoon, nav15','color gray70, nav15',
           f'load "{(output/"structures/qdn_native.pdb").as_posix()}", qdn_native','show sticks, qdn_native','color green, qdn_native']
    for label in list(sites)+list(docksites):
        lines += [f'load "{(output/"structures"/(label+".pdb")).as_posix()}", {label}',f'show sticks, {label}',
                  f'color {"cyan" if label.startswith("passing") else "red" if label.endswith("score_top1") else "yellow"}, {label}']
    lines += ['zoom qdn_native, 15','set stick_radius, 0.15','bg_color white']
    (output/'compare_sites.pml').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    md=['# Nav1.5 已知结合位点的结构叠合','',
        '全链序列对应后，以 CA 分别做整体与孔道局部配准；孔道区段依据 human SCN5A Q14524 的 S5、膜内 P 区、S6 注释。两种配准保留所有对应点，未按配体结果删点。原评分、对接代码与受体文件保持不变。','',
        '| 来源 | 全局 CA 对数 / RMSD | 孔道 CA 对数 / RMSD | 孔道配准后配体距 QDN 质心 | global→pore 质心位移 |',
        '|---|---:|---:|---:|---:|']
    for name,item in alignments.items():
        g,p=item['fits']['global'],item['fits']['pore']
        md.append(f"| {name} | {g['pairs']} / {g['ca_rmsd_A']:.3f} Å | {p['pairs']} / {p['ca_rmsd_A']:.3f} Å | {p['qdn_comparison']['centroid_distance_A']:.3f} Å | {item['global_to_pore_ligand_centroid_shift_A']:.3f} Å |")
    md += ['','| Vina 分数选出的第一名 | 距 flecainide 位点质心：全局 / 孔道配准 | 孔道配准后受体接触 Jaccard | 源实验接触映射后的 Jaccard |',
           '|---|---:|---:|---:|']
    for name,item in docksites.items():
        if not name.endswith('score_top1'):
            continue
        g=item['known_sites']['rat_6uz0_global'];p=item['known_sites']['rat_6uz0_pore']
        md.append(f"| {name} | {g['centroid_distance_A']:.3f} / {p['centroid_distance_A']:.3f} Å | {p['contact_jaccard_4A']:.3f} | {p['source_experimental_contact_jaccard_4A']:.3f} |")
    md += ['','`comparisons.csv` 列出各重对接代表姿势与已知位点的质心距离、最近重原子距离，以及统一以 6LQA 受体计算的 4 Å 接触交集。不同分子不计算原子 RMSD。',
           '','JSON 同时提供源实验受体原始接触残基及其到 6LQA 的序列映射；每个接触交集列出目标和源的双向编号。源受体实验接触与“映射配体在目标受体中的接触”分开记录，避免把不同侧链环境混同。',
           '','绿色为 QDN 实验配体；红色为各状态/框按原 Vina 分数选出的第一名；青色为全输出中通过姿势条件的诊断例；黄色为已知位点映射。运行 `compare_sites.pml` 可在 PyMOL 查看。',
           '','9V3S 的 A1EQU 仅映射实验原子的坐标；CCD 化学图存在 N–N 单键与实验 cis-azo 的冲突。本目录没有将该配体输出为已验证的化学 SDF，不能用于已验证的重对接。',
           '','同源映射只能说明位置关系。6UZ0 为大鼠 Nav1.5，7WE4 为 Nav1.8，9V3S 配体与构建体不同；重叠不能证明相同结合方式、亲和力或功能风险。整体/局部配准差异和残基对应已完整保存在 JSON，供审查。',
           '','本次未跑新的 Vina，未释放候选筛选门槛。']
    (output/'README.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    if not final['production_code_unchanged']:
        raise RuntimeError('Production code changed during alignment')
    print(output/'report.json',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE/'outputs/nav15_site_alignment_20261002')
    parser.add_argument('--docking',type=Path,default=BASE/'outputs/nav15_conformer_redocking_20261002')
    parser.add_argument('--include-8f6p',action='store_true',help='Add rat 8F6P observed ranolazine coordinates')
    parser.add_argument('--coverage-only',type=Path,help='Add new box-coverage supplement files to an existing alignment output; existing files preserved')
    parser.add_argument('--package',type=Path,help='Package an existing output without modifying it')
    parser.add_argument('--package-output',type=Path,default=BASE/'results/nav15_multireference_20261002/alignment')
    arguments=parser.parse_args()
    if arguments.package:
        package_output(arguments.package,arguments.package_output)
    elif arguments.coverage_only:
        supplement_box_coverage(arguments.coverage_only)
    else:
        main(arguments)
