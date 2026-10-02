"""Matched protein/lipid and protonation diagnostics, with frozen independent inputs.

This runner never changes production scoring, gates, or historical artifacts.
Chemical states are reported separately; no best-over-states candidate score.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from rdkit import Chem
from scgd.geometry import pose_contacts
from scgd.runtime import (code_hashes, dock_job, read_json, resolve_vina, sha256,
                          versions, work_manifest, write_json)
from scgd.validation import assess_pose

SEEDS = (20261021, 20261022, 20261023)


def frozen_paths(paths):
    return {str(Path(p).resolve()): sha256(p) for p in paths}


def verify_frozen(paths):
    for path, digest in paths.items():
        if not Path(path).is_file() or sha256(path) != digest:
            raise ValueError(f'Frozen input changed: {path}')


def lipid_distances(mol, lipid):
    if lipid is None:
        return {'min_lipid_heavy_distance_A': None, 'lipid_heavy_pairs_below_2A': None}
    xyz = Chem.RemoveHs(mol).GetConformer().GetPositions()
    lipid_xyz = Chem.RemoveHs(lipid).GetConformer().GetPositions()
    d = np.linalg.norm(xyz[:, None] - lipid_xyz[None], axis=2)
    return {'min_lipid_heavy_distance_A': float(d.min()),
            'lipid_heavy_pairs_below_2A': int((d < 2).sum())}


def summarize(rows, nconformers, has_lipid):
    expected = nconformers * len(SEEDS)
    ok = [r for r in rows if r['status'] == 'ok']
    protein_pass = sum(r['top1_passed'] for r in ok)
    passed = sum(r['top1_environment_passed'] for r in ok)
    ensemble = []
    for seed in SEEDS:
        group = [r for r in ok if r['vina_seed'] == seed]
        if len(group) != nconformers or not group:
            continue
        best = min(group, key=lambda r: (r['affinity_kcal_mol'], r['conformer_id']))
        ensemble.append({k: best[k] for k in ('conformer_id', 'vina_seed', 'affinity_kcal_mol',
                                              'top1_rmsd_A', 'top1_environment_passed')})
    return {'expected_jobs': expected, 'completed_jobs': len(rows), 'successful_jobs': len(ok),
            'distinct_input_conformers': nconformers, 'top1_protein_pass_count': protein_pass,
            'top1_pass_count': passed, 'any_mode_pass_count': sum(r['any_mode_passed'] for r in ok),
            'top1_pass_fraction': passed/max(1, expected),
            'median_top1_rmsd_A': float(np.median([r['top1_rmsd_A'] for r in ok])) if ok else None,
            'median_best_of_modes_rmsd_A': float(np.median([r['best_of_modes_rmsd_A'] for r in ok])) if ok else None,
            'top1_near_actual_grid_edge_0p05A': sum(r['modes'][0]['actual_grid_face_margin_A'] < .05 for r in ok),
            'top1_near_actual_grid_edge_0p5A': sum(r['modes'][0]['actual_grid_face_margin_A'] < .5 for r in ok),
            'includes_lipid': has_lipid,
            'reference_diagnostic_passed': bool(nconformers >= 5 and len(ok) == expected and
                                                 passed/max(1, expected) >= .8),
            'original_score_ensemble_winners': ensemble,
            'production_gate_released': False, 'selectivity_validated': False}


def read_inputs(config):
    prep = read_json(config['preparation_manifest'])
    approvals = [prep[k] for k in ('chemistry_usable', 'eligible_for_reference_redocking') if k in prep]
    if any(x is False for x in approvals) or not any(x is True for x in approvals):
        raise ValueError('Unapproved chemical identity')
    reference = Path(prep['reference_sdf'])
    protein = Path(prep['prepared_pdb'])
    receptor = Path(config.get('receptor_pdbqt', prep['receptor_pdbqt']))
    if 'lipid_sdf' in config and 'environment_manifest' not in config:
        raise ValueError('Lipid receptor requires a preparation audit')
    if 'lipid_sdf' not in config and receptor.resolve() != Path(prep['receptor_pdbqt']).resolve():
        raise ValueError('Unexplained receptor override')
    sampled = read_json(config['sampling_report'])
    if sampled.get('reference_coordinates_used') is not False or sampled.get('selection_uses_reference_rmsd') is not False:
        raise ValueError('Input sampling must explicitly exclude experimental geometry selection')
    ref = Chem.SDMolSupplier(str(reference), removeHs=True)[0]
    if ref is None or Chem.MolToSmiles(ref) != sampled['smiles']:
        raise ValueError('Sampling and reference graphs/stereo differ')
    entries = sampled['selected_entries']
    if not entries or len({e['conformer_id'] for e in entries}) != len(entries):
        raise ValueError('No conformers or duplicate identities')
    # Distinct conformers must actually satisfy the declared heavy-RMSD threshold.
    rmsd = np.asarray(sampled['selected_pairwise_rmsd']['matrix_A'])
    if rmsd.shape != (len(entries), len(entries)) or (len(entries) > 1 and
            float(rmsd[np.triu_indices(len(entries), 1)].min()) < .75 - 1e-8):
        raise ValueError('Inputs fail conformer deduplication')
    files = [Path(config['preparation_manifest']), reference, protein, receptor, Path(config['sampling_report'])]
    for entry in entries:
        path = Path(entry['ligand'])
        if sha256(path) != entry['typing_audit']['pdbqt_sha256']:
            raise ValueError('Typed ligand changed')
        for p in (path, path.with_suffix('.sdf'), path.with_suffix('.roundtrip.sdf')):
            files.append(p)
        mol = Chem.SDMolSupplier(str(path.with_suffix('.sdf')), removeHs=True)[0]
        if mol is None or Chem.MolToSmiles(mol) != Chem.MolToSmiles(ref):
            raise ValueError('Docking input molecular identity differs')
    lipid = None
    if 'lipid_sdf' in config:
        files += [Path(config['lipid_sdf']), Path(config['environment_manifest'])]
        lipid = Chem.SDMolSupplier(str(config['lipid_sdf']), removeHs=True)[0]
        if lipid is None:
            raise ValueError('Invalid lipid geometry')
        original = Path(prep['receptor_pdbqt']).read_bytes()
        if not receptor.read_bytes().startswith(original):
            raise ValueError('Matched lipid receptor must preserve all original protein bytes')
        files.append(Path(prep['receptor_pdbqt']))
    return ref, protein, receptor, entries, lipid, files


def run(args):
    config = read_json(args.config)
    ref, protein, receptor, entries, lipid, files = read_inputs(config)
    center = ref.GetConformer().GetPositions().mean(axis=0)
    size = 24.0
    if size/2 - np.abs(ref.GetConformer().GetPositions()-center).max() < 4:
        raise ValueError('Insufficient native padding')
    vina = resolve_vina()
    files += [args.config, Path(__file__), vina,
              Path(__file__).with_name('validate_nav15_multireference.py'),
              Path(__file__).with_name('nav15_conformer_sampling.py'),
              Path(__file__).with_name('diagnose_nav15_matched_geometry.py')]
    signature = {'diagnostic_only': True, 'condition': config['condition'],
                 'configuration': config, 'input_hashes': frozen_paths(files),
                 'production_code_hashes': code_hashes(), 'versions': versions(),
                 'center': center.tolist(), 'box_size_A': size, 'vina_seeds': list(SEEDS),
                 'exhaustiveness': 16, 'num_modes': 9, 'cpu_per_job': 2, 'timeout': 1200,
                 'scoring': 'Original default Vina; no changed weights or candidate formula',
                 'criteria': 'Top1 RMSD <=2 A; protein contacts recall >=0.5; protein heavy clashes <2 A absent; additionally no lipid heavy clashes when lipid present; >=80% of >=5 distinct inputs x3 seeds.',
                 'production_gate_released': False,
                 'limitations': 'Fixed experimental heavy pose with hypothesized charge is a chemical-state sensitivity control; it is not independent experimental validation of that charge. Rigid protein and fixed single lipid, no membrane dynamics or explicit electrostatics.'}
    output = args.output.resolve()
    work_manifest(output, signature, args.resume)
    native_contacts = set(pose_contacts(ref, protein)['contact_residues_4A'])
    write_json(output/'native_geometry.json', {**pose_contacts(ref, protein), **lipid_distances(ref, lipid)})
    rows = []

    def one(entry, seed):
        folder = output/'jobs'/f'{entry["conformer_id"]}_s{seed}'
        row = {'condition': config['condition'], 'conformer_id': entry['conformer_id'], 'vina_seed': seed}
        try:
            pose = dock_job(entry['ligand'], receptor, folder/'poses.pdbqt', center, size,
                            vina=vina, exhaustiveness=16, num_modes=9, seed=seed, cpu=2,
                            timeout=1200, resume=args.resume)
            assessed = assess_pose(pose, ref, protein)
            poses = list(Chem.SDMolSupplier(str(pose.with_suffix('.sdf')), removeHs=True))
            details = []
            actual_size = np.ceil(size/.375/2)*2*.375
            for rank, (mol, rmsd, affinity) in enumerate(zip(poses, assessed['all_pose_rmsd_A'],
                                                           assessed['all_pose_affinity_kcal_mol']), 1):
                contacts = pose_contacts(mol, protein)
                recall = len(native_contacts & set(contacts['contact_residues_4A'])) / max(1, len(native_contacts))
                environment = lipid_distances(mol, lipid)
                protein_passed = rmsd <= 2 and recall >= .5 and contacts['heavy_pairs_below_2A'] == 0
                details.append({'rank': rank, 'rmsd_A': rmsd, 'affinity_kcal_mol': affinity,
                                'contact_recall': recall, **contacts, **environment,
                                'actual_grid_face_margin_A': float(actual_size/2-np.abs(mol.GetConformer().GetPositions()-center).max()),
                                'protein_passed': protein_passed,
                                'passed': protein_passed and (lipid is None or environment['lipid_heavy_pairs_below_2A'] == 0)})
            row.update(status='ok', **assessed, modes=details, pose=str(pose),
                       top1_environment_passed=details[0]['passed'],
                       any_mode_passed=any(m['passed'] for m in details))
        except Exception as exc:
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(folder/'result.json', row)
        return row

    def checkpoint():
        ordered = sorted(rows, key=lambda r: (r['conformer_id'], r['vina_seed']))
        write_json(output/'results.json', ordered)
        report = summarize(rows, len(entries), lipid is not None)
        report.update(condition=config['condition'], complete=len(rows) == report['expected_jobs'],
                      production_code_unchanged=code_hashes() == signature['production_code_hashes'])
        write_json(output/'report.json', report)
        return report

    checkpoint()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, entry, seed) for entry in entries for seed in SEEDS]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            report = checkpoint()
            print(f'{config["condition"]} {len(rows)}/{report["expected_jobs"]} {row["status"]} '
                  f'RMSD={row.get("top1_rmsd_A")} pass={row.get("top1_environment_passed")}', flush=True)
    final = checkpoint()
    verify_frozen(signature['input_hashes'])
    if not final['production_code_unchanged']:
        raise ValueError('Production code changed during run')
    return 0 if final['successful_jobs'] == final['expected_jobs'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    sys.exit(run(args))
