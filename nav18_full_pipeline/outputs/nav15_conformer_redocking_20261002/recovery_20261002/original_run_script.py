"""Graph-only deduplicated QDN inputs, matched pore/wide-box global redocking.

This standalone experiment uses the original Vina weights and the original
pose criteria. It does not produce a production candidate-release report.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

import score
from nav15_conformer_sampling import sample_state
from scgd.geometry import pose_contacts
from scgd.runtime import code_hashes, dock_job, read_json, resolve_vina, sha256, versions, write_json
from scgd.validation import assess_pose, reference_molecule

STATES = ('nav15', 'nav15_qdn_protonated')
BOXES = {'pore20': 20.0, 'wide28': 28.0}
PDB = BASE / 'outputs/receptors_20261002/nav15/prepared.pdb'
RECEPTOR = PDB.with_name('receptor.pdbqt')
VINA_SEEDS = (20261011, 20261012, 20261013)
EMBED_SEEDS = (202610021, 202610022, 202610023, 202610024)
PORE_RESIDUES = (371, 402, 405, 1417, 1708, 1710, 1759, 1760, 1763, 1767)


def csv_write(path, rows):
    if rows:
        with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def boundary_audit(mol, size):
    xyz = Chem.RemoveHs(mol).GetConformer().GetPositions()
    delta = np.abs(xyz - np.array(score.RECEPTORS['nav15']['center']))
    axis_margin = size / 2 - delta.max(axis=0)
    return {'nominal_minimum_face_margin_A': float(axis_margin.min()),
            'nominal_axis_face_margins_A': list(map(float, axis_margin)),
            'near_nominal_boundary_within_0p05A': bool(axis_margin.min() <= .05),
            'heavy_centroid_xyz': list(map(float, xyz.mean(axis=0)))}


def box_definition(ref):
    center = np.array(score.RECEPTORS['nav15']['center'])
    residues = []
    xyz = ref.GetConformer().GetPositions()
    for resid in PORE_RESIDUES:
        lines = [line for line in PDB.read_text().splitlines()
                 if line.startswith('ATOM  ') and line[21] == 'B' and
                 line[22:26].strip() == str(resid) and line[76:78].strip() not in ('H', 'D')]
        if not lines:
            raise ValueError(f'Missing pore audit residue {resid}')
        coords = np.array([[float(l[i:i+8]) for i in (30, 38, 46)] for l in lines])
        d = np.linalg.norm(coords[:, None] - xyz[None], axis=2).min(axis=1)
        idx = int(d.argmin())
        ca = next(l for l in lines if l[12:16].strip() == 'CA')
        ca_xyz = np.array([float(ca[i:i+8]) for i in (30, 38, 46)])
        residues.append({'residue': f'B:{resid}:{lines[0][17:20]}',
                         'CA_inside_pore20': bool(np.all(np.abs(ca_xyz-center) < 10)),
                         'nearest_atom_to_native': lines[idx][12:16].strip(),
                         'nearest_atom_native_distance_A': float(d[idx]),
                         'nearest_atom_pore20_face_margin_A': float(10-np.abs(coords[idx]-center).max())})
    assert all(r['CA_inside_pore20'] and r['nearest_atom_pore20_face_margin_A'] > 0 for r in residues)
    return {'center_xyz_A': list(map(float, center)), 'boxes': BOXES,
            'pore_definition': '6LQA QDN central cavity; documented native 4A-contact residues plus Y1767. Fixed before this run. Same center as original pipeline.',
            'residue_audit': residues,
            'experimental_pose_boundaries': {name: boundary_audit(ref, size) for name, size in BOXES.items()},
            'ligand_search_volume_ratio_wide_to_pore': (28/20)**3,
            'receptor_cropped': False,
            'limitations': '20A is a site-conditioned ligand search; it does not geometrically eliminate every alternate orientation. Same exhaustiveness has less search density in the larger box. Reported margins use nominal requested sizes, not rounded actual grid extent.'}


def summarize(rows, states, expected_jobs):
    groups = []
    for state in STATES:
        n = len(states[state]['selected_entries'])
        expected = n * len(VINA_SEEDS)
        adequate = n >= 5 and len(VINA_SEEDS) >= 3
        for box in BOXES:
            group = [r for r in rows if r['reference'] == state and r['box'] == box]
            ok = [r for r in group if r['status'] == 'ok']
            passed = sum(r['top1_passed'] for r in ok)
            found = sum(r['any_output_pose_passed'] for r in ok)
            ensemble = []
            for seed in VINA_SEEDS:
                selected = [r for r in ok if r['vina_seed'] == seed]
                if len(selected) != n:
                    continue
                # Rank only by original top1 Vina affinity, never by native RMSD.
                best = min(selected, key=lambda r: (r['affinity_kcal_mol'], r['conformer_id']))
                ensemble.append({'vina_seed': seed, 'selected_conformer_id': best['conformer_id'],
                                 'affinity_kcal_mol': best['affinity_kcal_mol'],
                                 'top1_rmsd_A': best['top1_rmsd_A'], 'top1_passed': best['top1_passed'],
                                 'any_output_pose_passed': any(r['any_output_pose_passed'] for r in selected)})
            groups.append({'reference': state, 'box': box, 'size_A': BOXES[box],
                           'distinct_input_conformers': n, 'expected_jobs': expected,
                           'completed_jobs': len(group), 'successful_jobs': len(ok),
                           'top1_pass_count': passed, 'top1_pass_fraction': passed/max(1,expected),
                           'any_output_pass_count': found, 'any_output_pass_fraction': found/max(1,expected),
                           'median_top1_rmsd_A': float(np.median([r['top1_rmsd_A'] for r in ok])) if ok else None,
                           'median_best_of_modes_rmsd_A': float(np.median([r['best_of_modes_rmsd_A'] for r in ok])) if ok else None,
                           'median_affinity_kcal_mol': float(np.median([r['affinity_kcal_mol'] for r in ok])) if ok else None,
                           'top1_near_boundary_count': sum(r['top1_boundary']['near_nominal_boundary_within_0p05A'] for r in ok),
                           'adequate_distinct_conformer_sampling': adequate,
                           'pose_criteria_met_for_this_condition': adequate and len(ok)==expected and passed/max(1,expected)>=.8,
                           'ensemble_score_selected_top1_pass_count': sum(r['top1_passed'] for r in ensemble),
                           'ensemble_complete_seed_count': len(ensemble), 'ensemble_seed_results': ensemble})
    return {'diagnostic_only': True, 'independent_internal_conformer_generation': True,
            'site_information_supplied': 'Experimental cavity center and pore residue definition; no experimental ligand coordinates supplied as input conformers.',
            'expected_jobs': expected_jobs, 'completed_jobs': len(rows),
            'successful_jobs': sum(r['status']=='ok' for r in rows), 'complete': len(rows)==expected_jobs,
            'summary': groups,
            'criteria': 'Top1 fixed-frame symmetry heavy RMSD<=2A, contact recall>=0.5, no protein/ligand heavy pairs<2A; >=80% of expected jobs; >=5 distinct input conformers and >=3 Vina seeds. Failed jobs stay in denominator.',
            'production_gate_released': False,
            'limitations': ['Same molecule, chemical states and receptor; input conformers/seeds are correlated, not independent biological samples.',
                            '+1 is a sensitivity control. Neutral and +1 scores are not directly compared as binding-energy truth.',
                            'Ensemble winners use original Vina score only. This experiment does not change production single-conformer scoring.',
                            'No Nav1.8 run in this experiment; no whole-library screening or selectivity claim.']}


def run(output, args):
    output = Path(output).resolve()
    frozen = code_hashes()
    vina = resolve_vina()
    if args.resume:
        manifest = read_json(output/'manifest.json')
        assert manifest['production_code_hashes'] == frozen
        assert manifest['script_sha256'] == sha256(__file__)
        assert manifest['sampling_script_sha256'] == sha256(Path(__file__).with_name('nav15_conformer_sampling.py'))
        assert manifest['receptor_sha256'] == sha256(RECEPTOR) and manifest['vina_sha256']==sha256(vina)
        assert manifest['exhaustiveness']==args.exhaustiveness and manifest['cpu_per_job']==args.cpu
        states = {s: read_json(output/'ligands'/s/'report.json') for s in STATES}
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest = {'diagnostic_only': True, 'production_code_hashes': frozen,
                    'script_sha256': sha256(__file__),
                    'sampling_script_sha256': sha256(Path(__file__).with_name('nav15_conformer_sampling.py')),
                    'receptor_sha256': sha256(RECEPTOR), 'prepared_pdb_sha256': sha256(PDB),
                    'reference_sha256': {s:sha256(BASE/'data/reference_validation'/f'{s}.sdf') for s in STATES},
                    'vina_sha256': sha256(vina), 'versions': versions(),
                    'center_xyz_A': list(score.RECEPTORS['nav15']['center']), 'boxes': BOXES,
                    'embedding_seeds': list(EMBED_SEEDS), 'conformers_per_seed':128,
                    'rmsd_dedup_threshold_A':.75, 'same_state_energy_window_kcal_mol':10.,
                    'max_selected_per_state':8, 'vina_seeds':list(VINA_SEEDS),
                    'exhaustiveness':args.exhaustiveness, 'num_modes':9,
                    'cpu_per_job':args.cpu, 'workers':args.workers, 'timeout_seconds':900,
                    'selection_policy': 'Independent graph-only ETKDG, converged valid MMFF94s pool, energy-ordered 0.75A greedy dedup within 10 kcal same-state window; no native geometry selection.',
                    'scoring': 'Original Vina defaults; original combined scorer not used or changed.'}
        write_json(output/'manifest.json', manifest)
        states = {}
        for state in STATES:
            print(f'Sampling {state}: 4 x 128 graph-only embeddings', flush=True)
            states[state] = sample_state(state, output/'ligands'/state, seeds=EMBED_SEEDS,
                                         conformers_per_seed=128, rmsd_threshold=.75,
                                         energy_window=10., max_selected=8)
            print(f"{state}: {len(states[state]['selected_entries'])} distinct conformers selected", flush=True)
    refs = {s: reference_molecule(s, score) for s in STATES}
    if not (output/'box_definition.json').exists():
        write_json(output/'box_definition.json', box_definition(refs['nav15']))
    coverage = []
    for state in STATES:
        for entry in states[state]['selected_entries']:
            mol = Chem.SDMolSupplier(entry['sdf'])[0]
            if mol is None:
                raise ValueError('Invalid sampled SDF')
            aligned = float(rdMolAlign.GetBestRMS(Chem.RemoveHs(Chem.Mol(mol)), refs[state]))
            coverage.append({'reference':state, 'conformer_id':entry['conformer_id'],
                             'aligned_heavy_rmsd_to_experimental_A':aligned,
                             'energy_kcal_mol':entry['energy_kcal_mol'],
                             'use': 'Post-selection coverage evaluation only; does not determine sampling or docking rank.'})
    write_json(output/'coverage_audit.json', coverage)
    jobs = [(state, entry, box, seed) for state in STATES for entry in states[state]['selected_entries']
            for seed in VINA_SEEDS for box in BOXES]
    expected = len(jobs)
    rows = []
    def one(job):
        state, entry, box, seed = job
        conf = entry['conformer_id']
        folder = output/'jobs'/f'{state}_{conf}_{box}_s{seed}'
        folder.mkdir(parents=True, exist_ok=True)
        row = {'reference':state, 'conformer_id':conf, 'box':box, 'vina_seed':seed,
               'ligand_sha256':sha256(entry['ligand']), 'status':'running'}
        write_json(folder/'result.json', row)
        try:
            pose = dock_job(entry['ligand'], RECEPTOR, folder/'poses.pdbqt',
                            score.RECEPTORS['nav15']['center'], BOXES[box], vina=vina,
                            exhaustiveness=args.exhaustiveness, num_modes=9, seed=seed,
                            cpu=args.cpu, timeout=900, resume=args.resume)
            assessment = assess_pose(pose, refs[state], PDB)
            mols = list(Chem.SDMolSupplier(str(pose.with_suffix('.sdf'))))
            native_contacts = set(pose_contacts(refs[state], PDB)['contact_residues_4A'])
            mode_rows = []
            for rank, (mol, rmsd, affinity) in enumerate(zip(mols, assessment['all_pose_rmsd_A'], assessment['all_pose_affinity_kcal_mol']),1):
                contacts = pose_contacts(mol, PDB)
                recall = len(native_contacts & set(contacts['contact_residues_4A']))/len(native_contacts)
                mode_rows.append({'rank':rank, 'affinity_kcal_mol':affinity, 'rmsd_A':rmsd,
                                  'contact_recall':recall, **contacts, 'boundary':boundary_audit(mol,BOXES[box]),
                                  'pose_passed':rmsd<=2 and recall>=.5 and contacts['heavy_pairs_below_2A']==0})
            qualifying = [m for m in mode_rows if m['pose_passed']]
            first = qualifying[0] if qualifying else None
            row.update(status='ok', **assessment, all_modes=mode_rows,
                       any_output_pose_passed=bool(qualifying),
                       first_passing_pose_rank=first['rank'] if first else None,
                       passing_pose_score_gap_kcal_mol=first['affinity_kcal_mol']-assessment['affinity_kcal_mol'] if first else None,
                       top1_boundary=mode_rows[0]['boundary'],
                       pose_sha256=sha256(pose))
        except Exception as exc:
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(folder/'result.json', row)
        return row
    def checkpoint():
        ordered = sorted(rows,key=lambda r:(r['reference'],r['box'],r['conformer_id'],r['vina_seed']))
        write_json(output/'results.json', ordered)
        report = summarize(ordered,states,expected)
        report['production_code_unchanged'] = code_hashes()==frozen
        write_json(output/'report.json',report)
        columns = ('reference','conformer_id','box','vina_seed','status','affinity_kcal_mol','top1_rmsd_A',
                   'best_of_modes_rmsd_A','contact_recall','heavy_pairs_below_2A','top1_passed',
                   'any_output_pose_passed','first_passing_pose_rank','passing_pose_score_gap_kcal_mol')
        csv_write(output/'jobs_summary.csv',[{k:r.get(k) for k in columns} for r in ordered])
        csv_write(output/'summary.csv',[{k:v for k,v in r.items() if k!='ensemble_seed_results'} for r in report['summary']])
        return report
    checkpoint()
    print(f'Global redocking: {expected} jobs, boxes20/28, {args.workers} workers x {args.cpu} CPU',flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one,job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            checkpoint()
            print(f"{len(rows)}/{expected} {row['reference']} {row['conformer_id']} {row['box']} s{row['vina_seed']} {row['status']} RMSD={row.get('top1_rmsd_A')} pass={row.get('top1_passed')}",flush=True)
    report = checkpoint()
    if not report['production_code_unchanged'] or report['successful_jobs']!=expected:
        raise RuntimeError('Incomplete experiment or production code changed; inspect recorded results.')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE/'outputs/nav15_conformer_redocking_20261002')
    parser.add_argument('--exhaustiveness',type=int,default=16)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--cpu',type=int,default=2)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if min(args.exhaustiveness,args.workers,args.cpu)<1:
        parser.error('Resource settings must be positive')
    run(args.output,args)
