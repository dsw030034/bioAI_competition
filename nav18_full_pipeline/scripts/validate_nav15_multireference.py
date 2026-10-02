"""Exploratory self-redocking of audited references; production scoring unchanged.

Each receptor/ligand pair has its own immutable run directory. Experimental
coordinates are used only for box definition and assessment, never sampling.
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
from rdkit.Chem import AllChem

from diagnose_nav15_matched_geometry import export_typed
from nav15_conformer_sampling import (graph_only, embed_seed_pool, deduplicate,
                                     single_conformer, rmsd_matrix)
from scgd.geometry import pose_contacts
from scgd.ligand_quality import geometry_issues
from scgd.runtime import (code_hashes, dock_job, read_json, resolve_vina, sha256,
                          versions, work_manifest, write_json)
from scgd.validation import assess_pose

EMBED_SEEDS = (202610031, 202610032)
VINA_SEEDS = (20261021, 20261022, 20261023)


def sample(reference, output):
    if output.exists():
        raise ValueError('Incomplete sampling directory exists without a frozen input manifest; '
                         'preserve it for audit and use a new output directory')
    output.mkdir(parents=True, exist_ok=False)
    source = Chem.SDMolSupplier(str(reference), removeHs=True)[0]
    if source is None:
        raise ValueError('Invalid reference graph')
    graph = graph_only(source)
    expected = Chem.MolToSmiles(graph, isomericSmiles=True)
    del source
    pool = Chem.AddHs(graph)
    if not AllChem.MMFFHasAllMoleculeParams(pool):
        raise ValueError('Incomplete MMFF94s parameters')
    records = []
    for batch, seed in enumerate(EMBED_SEEDS):
        embedded, ids = embed_seed_pool(graph, seed, 64)
        props = AllChem.MMFFGetMoleculeProperties(embedded, mmffVariant='MMFF94s')
        for within, idx in enumerate(ids):
            row = {'conformer_id': f'b{batch}_e{within:03d}', 'seed': seed,
                   'rdkit_conf_id': None, 'status': 'pending', 'energy_kcal_mol': None,
                   'minimization_codes': [], 'geometry_issues': []}
            try:
                ff = AllChem.MMFFGetMoleculeForceField(embedded, props, confId=idx)
                if ff is None:
                    raise ValueError('Missing force field')
                for iterations in (1000, 3000):
                    code = int(ff.Minimize(maxIts=iterations))
                    row['minimization_codes'].append(code)
                    if code == 0:
                        break
                energy = float(ff.CalcEnergy())
                row['energy_kcal_mol'] = energy if np.isfinite(energy) else None
                row['geometry_issues'] = geometry_issues(embedded, idx, expected)
                row['status'] = ('geometry_passed' if code == 0 and np.isfinite(energy)
                                 and not row['geometry_issues'] else 'geometry_failed')
                row['rdkit_conf_id'] = pool.AddConformer(
                    Chem.Conformer(embedded.GetConformer(idx)), assignId=True)
            except (RuntimeError, ValueError) as exc:
                row.update(status='optimization_error', error=str(exc))
            records.append(row)
        print(f'{reference.parent.name}: batch {batch}, embedded {len(ids)}/64', flush=True)
        write_json(output/'progress.json', records)
    valid = [r for r in records if r['status'] == 'geometry_passed']
    with Chem.SDWriter(str(output/'all_valid_optimized.sdf')) as writer:
        for row in valid:
            m = single_conformer(pool, row['rdkit_conf_id'])
            m.SetProp('conformer_id', row['conformer_id'])
            writer.write(m)
    unique, selected = deduplicate(pool, records, .75, 10.0, 5)
    entries = []
    for i, row in enumerate(selected):
        ligand = output/'selected'/f'c{i:02d}.pdbqt'
        audit = export_typed(single_conformer(pool, row['rdkit_conf_id']), ligand)
        entries.append({'conformer_id': row['conformer_id'], 'ligand': str(ligand),
                        'energy_kcal_mol': row['energy_kcal_mol'], 'typing_audit': audit})
    report = {'reference_coordinates_used': False, 'selection_uses_receptor': False,
              'selection_uses_reference_rmsd': False, 'smiles': expected,
              'counts': {'attempted': len(EMBED_SEEDS) * 64, 'embedded': len(records),
                         'embedding_failed': len(EMBED_SEEDS) * 64 - len(records), 'valid': len(valid),
                         'unique': len(unique), 'selected': len(entries)},
              'selected_pairwise_rmsd': rmsd_matrix(pool, selected), 'selected_entries': entries}
    write_json(output/'records.json', records)
    write_json(output/'report.json', report)
    if not entries:
        raise ValueError('No usable independent conformers')
    return report


def freeze_artifacts(directory):
    return {str(p.relative_to(directory)): sha256(p)
            for p in sorted(directory.rglob('*')) if p.is_file()}


def verify_artifacts(directory, expected):
    for relative, digest in expected.items():
        path = directory/relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'Frozen input/output changed: {path}')


def summarize(rows, nconformers):
    expected = nconformers * len(VINA_SEEDS)
    ok = [r for r in rows if r['status'] == 'ok']
    count = sum(r['top1_passed'] for r in ok)
    ensembles = []
    for seed in VINA_SEEDS:
        group = [r for r in ok if r['vina_seed'] == seed]
        if len(group) != nconformers or not group:
            continue
        best = min(group, key=lambda r: (r['affinity_kcal_mol'], r['conformer_id']))
        ensembles.append({k: best[k] for k in ('vina_seed', 'conformer_id', 'affinity_kcal_mol',
                                               'top1_rmsd_A', 'top1_passed')})
    return {'expected_jobs': expected, 'completed_jobs': len(rows), 'successful_jobs': len(ok),
            'distinct_input_conformers': nconformers, 'top1_pass_count': count,
            'any_mode_pass_count': sum(r['any_mode_passed'] for r in ok),
            'top1_pass_fraction': count/max(1, expected),
            'median_top1_rmsd_A': float(np.median([r['top1_rmsd_A'] for r in ok])) if ok else None,
            'median_best_of_modes_rmsd_A': float(np.median([r['best_of_modes_rmsd_A'] for r in ok])) if ok else None,
            'reference_diagnostic_passed': (nconformers >= 5 and len(ok) == expected
                                             and count/max(1, expected) >= .8),
            'original_score_ensemble_winners': ensembles,
            'production_gate_released': False}


def run(args):
    prep = args.reference_dir.resolve()
    output = args.output.resolve()
    reference = prep/'reference.sdf'
    receptor = prep/'receptor/receptor.pdbqt'
    receptor_pdb = prep/'receptor/prepared.pdb'
    manifest = prep/'prepare_manifest.json'
    audit = read_json(manifest)
    approvals = [audit[k] for k in ('chemistry_usable', 'eligible_for_reference_redocking') if k in audit]
    if any(flag is False for flag in approvals) or not any(flag is True for flag in approvals):
        raise ValueError('Preparation must explicitly approve chemical identity for redocking')
    # Explicit audited repair variants may live beside the original failed prep.
    reference = Path(audit.get('reference_sdf', reference)).resolve()
    receptor = Path(audit.get('receptor_pdbqt', receptor)).resolve()
    receptor_pdb = Path(audit.get('prepared_pdb', receptor_pdb)).resolve()
    ref = Chem.SDMolSupplier(str(reference), removeHs=True)[0]
    if ref is None:
        raise ValueError('Invalid experimental reference')
    xyz = ref.GetConformer().GetPositions()
    center = xyz.mean(axis=0).tolist()
    native_margin = float(args.size/2-np.abs(xyz-np.asarray(center)).max())
    if native_margin < 4:
        raise ValueError('Experimental reference has <4 A box padding; predeclare a larger box')
    vina = resolve_vina()
    signature = {'diagnostic_only': True, 'production_gate_released': False,
                 'reference': prep.name, 'reference_sha256': sha256(reference),
                 'receptor_sha256': sha256(receptor), 'prepared_pdb_sha256': sha256(receptor_pdb),
                 'preparation_manifest_sha256': sha256(manifest), 'vina_sha256': sha256(vina),
                 'script_sha256': sha256(__file__), 'production_code_hashes': code_hashes(),
                 'helper_hashes': {n: sha256(Path(__file__).with_name(n)) for n in
                                  ('nav15_conformer_sampling.py', 'diagnose_nav15_matched_geometry.py')},
                 'versions': versions(), 'center': center, 'box_size_A': args.size,
                 'native_box_margin_A': native_margin, 'exhaustiveness': 16, 'num_modes': 9,
                 'cpu_per_job': 2, 'timeout_seconds': 1200, 'vina_seeds': list(VINA_SEEDS),
                 'sampling': {'embedding_seeds': list(EMBED_SEEDS), 'conformers_per_seed': 64,
                              'MMFF94s_max_iterations': [1000, 3000], 'dedup_rmsd_A': .75,
                              'same_state_energy_window_kcal_mol': 10, 'max_selected': 5},
                 'criteria': 'Top1 fixed-frame heavy RMSD <=2 A; no heavy pairs <2 A; native contact recall >=0.5. Diagnostic pass requires >=80%, 5 distinct inputs x3 seeds.',
                 'limitations': 'Own-receptor self-redocking, rigid protein only, deposited input charge. '
                                'No cross-species affinity comparison; no candidate pose constraint.'}
    work_manifest(output, signature, args.resume)
    if (output/'input_integrity.json').exists():
        verify_artifacts(output/'sampling', read_json(output/'input_integrity.json'))
        sampled = read_json(output/'sampling/report.json')
    else:
        sampled = sample(reference, output/'sampling')
        write_json(output/'input_integrity.json', freeze_artifacts(output/'sampling'))
    native_contacts = set(pose_contacts(ref, receptor_pdb)['contact_residues_4A'])
    write_json(output/'native_contacts.json', pose_contacts(ref, receptor_pdb))
    rows = []

    def one(entry, seed):
        folder = output/'jobs'/f'{entry["conformer_id"]}_s{seed}'
        row = {'reference': prep.name, 'conformer_id': entry['conformer_id'], 'vina_seed': seed}
        try:
            pose = dock_job(entry['ligand'], receptor, folder/'poses.pdbqt', center, args.size,
                            vina=vina, exhaustiveness=16, num_modes=9, seed=seed, cpu=2,
                            timeout=1200, resume=args.resume)
            assessment = assess_pose(pose, ref, receptor_pdb)
            poses = list(Chem.SDMolSupplier(str(pose.with_suffix('.sdf')), removeHs=True))
            details = []
            # Vina expands nominal box to an even number of 0.375 A voxels.
            actual_size = np.ceil(args.size/.375/2)*2*.375
            for rank, (mol, rmsd, affinity) in enumerate(zip(
                    poses, assessment['all_pose_rmsd_A'], assessment['all_pose_affinity_kcal_mol']), 1):
                contacts = pose_contacts(mol, receptor_pdb)
                recall = len(native_contacts & set(contacts['contact_residues_4A']))/max(1, len(native_contacts))
                margin = float(actual_size/2-np.abs(mol.GetConformer().GetPositions()-center).max())
                details.append({'rank': rank, 'rmsd_A': rmsd, 'affinity_kcal_mol': affinity,
                                'contact_recall': recall, **contacts,
                                'actual_grid_face_margin_A': margin,
                                'passed': rmsd <= 2 and recall >= .5 and contacts['heavy_pairs_below_2A'] == 0})
            row.update(status='ok', **assessment, modes=details,
                       any_mode_passed=any(r['passed'] for r in details), pose=str(pose))
        except Exception as exc:
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(folder/'result.json', row)
        return row

    def checkpoint():
        ordered = sorted(rows, key=lambda r: (r['conformer_id'], r['vina_seed']))
        write_json(output/'results.json', ordered)
        report = summarize(ordered, len(sampled['selected_entries']))
        report.update(reference=prep.name, complete=len(rows) == report['expected_jobs'],
                      production_code_unchanged=code_hashes() == signature['production_code_hashes'])
        write_json(output/'report.json', report)
        return report

    checkpoint()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, entry, seed) for entry in sampled['selected_entries'] for seed in VINA_SEEDS]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            report = checkpoint()
            print(f'{len(rows)}/{report["expected_jobs"]} {prep.name} {row["status"]} '
                  f'top1_RMSD={row.get("top1_rmsd_A")} pass={row.get("top1_passed")}', flush=True)
    report = checkpoint()
    verify_artifacts(output/'sampling', read_json(output/'input_integrity.json'))
    if not report['production_code_unchanged']:
        raise ValueError('Production code changed during diagnostic')
    return 0 if report['successful_jobs'] == report['expected_jobs'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=float, default=24)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.size <= 0:
        parser.error('workers and size must be positive')
    sys.exit(run(args))
