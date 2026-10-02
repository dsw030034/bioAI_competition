"""Experimental-frame reference redocking; no changes to scoring functions."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from pathlib import Path
import re

from meeko import PDBQTMolecule, RDKitMolCreate
import numpy as np
from rdkit import Chem

from .geometry import fixed_frame_rmsd, pose_contacts
from .prep import prep_ligand
from .runtime import BASE, read_json, replace_file, sha256, write_json
from .workflow import protocol

REFERENCES = {
    'nav18': {'receptor': 'nav18', 'resname': '95T', 'resid': '2004', 'role': 'native_reference'},
    'nav15': {'receptor': 'nav15', 'resname': 'QDN', 'resid': '2110', 'role': 'native_reference'},
    'nav15_qdn_protonated': {'receptor': 'nav15', 'resname': 'QDN', 'resid': '2110',
                            'role': 'protonation_sensitivity_control'},
}


def reference_molecule(name, score):
    path = BASE / 'data/reference_validation' / (name + '.sdf')
    mol = Chem.SDMolSupplier(str(path), removeHs=True)[0]
    if mol is None:
        raise ValueError(f'Invalid reference fixture: {name}')
    info = REFERENCES[name]
    cfg = score.RECEPTORS[info['receptor']]
    deposited = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
        for l in cfg['pdb'].read_text().splitlines()
        if l.startswith('HETATM') and l[17:20] == info['resname'] and l[21] == cfg['chain']
        and l[22:26].strip() == info['resid'] and l[76:78].strip() not in ('H', 'D')])
    xyz = mol.GetConformer().GetPositions()
    if deposited.shape != xyz.shape:
        raise ValueError('Reference heavy atom counts differ from deposited ligand')
    distances = np.linalg.norm(deposited[:, None, :] - xyz[None, :, :], axis=2)
    if distances.min(axis=0).max() > 0.002 or len(set(distances.argmin(axis=0))) != len(xyz):
        raise ValueError('Reference coordinates do not match the deposited binding pose')
    return mol


def assess_pose(pose, reference, receptor_pdb):
    text = Path(pose).read_text()
    affinities = [float(s) for s in re.findall(r'REMARK VINA RESULT:\s+(-?[\d.]+)', text)]
    sdf, failures = RDKitMolCreate.write_sd_string(PDBQTMolecule(text, skip_typing=True))
    if failures or not sdf:
        raise ValueError(f'Pose export failed: {failures}')
    path = Path(pose).with_suffix('.sdf')
    path.write_text(sdf, encoding='utf-8')
    poses = list(Chem.SDMolSupplier(str(path), removeHs=True))
    if not poses or any(m is None for m in poses) or len(poses) != len(affinities):
        raise ValueError('Pose/score count mismatch')
    rmsds = [fixed_frame_rmsd(p, reference) for p in poses]
    contacts = pose_contacts(poses[0], receptor_pdb)
    native_contacts = set(pose_contacts(reference, receptor_pdb)['contact_residues_4A'])
    recall = len(native_contacts & set(contacts['contact_residues_4A'])) / max(1, len(native_contacts))
    passed = rmsds[0] <= 2 and contacts['heavy_pairs_below_2A'] == 0 and recall >= 0.5
    return {'affinity_kcal_mol': affinities[0], 'all_pose_affinity_kcal_mol': affinities,
            'top1_rmsd_A': rmsds[0], 'best_of_modes_rmsd_A': min(rmsds), 'all_pose_rmsd_A': rmsds,
            'contact_recall': recall, 'top1_passed': passed, **contacts}


def summarize(rows, expected_per_reference, adequate_sampling):
    summary = []
    for name, info in REFERENCES.items():
        group = [r for r in rows if r['reference'] == name]
        ok = [r for r in group if r['status'] == 'ok']
        passes = sum(r['top1_passed'] for r in ok)
        summary.append({'reference': name, 'role': info['role'], 'expected': expected_per_reference,
                        'completed': len(group), 'successful': len(ok), 'top1_pass_count': passes,
                        'top1_pass_fraction': passes / expected_per_reference,
                        'median_top1_rmsd_A': float(np.median([r['top1_rmsd_A'] for r in ok])) if ok else None,
                        'median_best_of_modes_rmsd_A': float(np.median([r['best_of_modes_rmsd_A'] for r in ok])) if ok else None,
                        'median_affinity': float(np.median([r['affinity_kcal_mol'] for r in ok])) if ok else None,
                        'gate_passed': adequate_sampling and len(ok) == expected_per_reference and passes / expected_per_reference >= 0.8})
    return summary


def run_reference(score, receptors, workdir, args):
    current_protocol = protocol(score, receptors, args)
    # Reference runs do not compute a cross-receptor delta for two different ligands.
    refs = {name: reference_molecule(name, score) for name in REFERENCES}
    jobs, rows = [], []
    for name, ref in refs.items():
        for repeat in range(args.reference_repeats):
            ligand = workdir / 'ligands' / f'{name}_c{repeat}.pdbqt'
            try:
                prep_ligand(Chem.MolToSmiles(ref), ligand, seed=args.preparation_seed + repeat,
                            candidate_id=name, num_confs=args.num_confs, resume=args.resume,
                            neutral_qdn_reference=name == 'nav15')
            except Exception as exc:
                for seed in args.vina_seeds:
                    rows.append({'reference': name, 'repeat': repeat, 'vina_seed': seed,
                                 'status': 'preparation_failed', 'error': f'{type(exc).__name__}: {exc}'})
                continue
            jobs += [(name, repeat, seed, ligand) for seed in args.vina_seeds]

    def one(job):
        name, repeat, seed, ligand = job
        target = REFERENCES[name]['receptor']
        folder = workdir / 'jobs' / f'{name}_c{repeat}_s{seed}'
        folder.mkdir(parents=True, exist_ok=True)
        row = {'reference': name, 'receptor': target, 'repeat': repeat, 'vina_seed': seed, 'status': 'running'}
        write_json(folder / 'result.json', row)
        try:
            pose = score.dock_box(ligand, receptors[target], folder / 'poses.pdbqt',
                                 score.RECEPTORS[target]['center'], score.RECEPTORS[target]['size'],
                                 vina=args.vina, exhaustiveness=args.exhaustiveness,
                                 num_modes=args.num_modes, seed=seed, cpu=args.cpu,
                                 timeout=args.timeout, resume=args.resume)
            assessment = assess_pose(pose, refs[name], receptors[target].parent / 'prepared.pdb')
            row.update(status='ok', **assessment)
        except Exception as exc:
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(folder / 'result.json', row)
        return row

    expected = args.reference_repeats * len(args.vina_seeds)
    adequate_sampling = args.reference_repeats >= 5 and len(args.vina_seeds) >= 3 and args.num_modes >= 9

    def checkpoint():
        ordered = sorted(rows, key=lambda r: (r['reference'], r['repeat'], r['vina_seed']))
        summary = summarize(ordered, expected, adequate_sampling)
        write_json(workdir / 'results.json', ordered)
        report = {'stage': 'reference', 'protocol': current_protocol, 'vina_seeds': args.vina_seeds,
                  'reference_repeats': args.reference_repeats, 'adequate_sampling': adequate_sampling,
                  'summary': summary, 'complete': len(rows) == expected * len(REFERENCES),
                  'gate_passed': all(r['gate_passed'] for r in summary if r['role'] == 'native_reference'),
                  'sensitivity_control_passed': next(r['gate_passed'] for r in summary if r['role'] == 'protonation_sensitivity_control'),
                  'reference_sha256': {n: sha256(BASE / 'data/reference_validation' / (n + '.sdf')) for n in REFERENCES},
                  'criteria': 'All jobs successful; >=80% top1 RMSD<=2 A, no heavy pairs<2 A, contact recall>=0.5; >=5 input pools x >=3 seeds.',
                  'rmsd_method': 'Symmetry-corrected heavy atom RMSD in receptor frame; no ligand alignment',
                  'limitations': 'Rigid protein-only; input chemical states; QDN +1 is a sensitivity control, not independent experimental evidence. Pose validation does not validate activity/selectivity.'}
        if not report['complete']:
            report['gate_passed'] = False
        write_json(workdir / 'report.json', report)
        temporary = workdir / 'summary.csv.tmp'
        with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        replace_file(temporary, workdir / 'summary.csv')
        return report

    checkpoint()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            checkpoint()
            print(f"{len(rows)}/{expected * len(REFERENCES)} {row['reference']} c{row['repeat']} s{row['vina_seed']} "
                  f"{row['status']} top1_RMSD={row.get('top1_rmsd_A')} pass={row.get('top1_passed')}", flush=True)
    report = checkpoint()
    print(f"Reference report: {workdir / 'report.json'}; gate_passed={report['gate_passed']}", flush=True)
    return 0 if report['gate_passed'] else 2
