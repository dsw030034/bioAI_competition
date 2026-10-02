"""Score and locally relax a deposited pose; this is not independent redocking.

Default Vina scoring/weights and original production code remain unchanged.
Each approved reference uses its own protein, charge state, centroid, and 24 A
box. One score-only and one local-only calculation are recorded per reference.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from meeko import PDBQTMolecule, RDKitMolCreate
from rdkit import Chem

from diagnose_nav15_native import export_fixed_pose, geometry, parse_affinity
from scgd.runtime import (code_hashes, logged_run, read_json, resolve_vina, sha256,
                          versions, work_manifest, write_json)


def native_diagnostic(reference_dir, resume=False):
    prep = Path(reference_dir).resolve()
    output = prep / 'native_diagnostic'
    prep_manifest = prep / 'prepare_manifest.json'
    audit = read_json(prep_manifest)
    approvals = [audit[k] for k in ('chemistry_usable', 'eligible_for_reference_redocking') if k in audit]
    if any(flag is False for flag in approvals) or not any(flag is True for flag in approvals):
        raise ValueError('Reference chemistry is not explicitly approved')
    reference = prep / 'reference.sdf'
    receptor = Path(audit.get('receptor_pdbqt', prep / 'receptor/receptor.pdbqt'))
    prepared = Path(audit.get('prepared_pdb', prep / 'receptor/prepared.pdb'))
    ref = Chem.SDMolSupplier(str(reference), removeHs=True)[0]
    if ref is None:
        raise ValueError('Invalid reference graph')
    coords = ref.GetConformer().GetPositions()
    center = coords.mean(axis=0)
    vina = resolve_vina()
    frozen = code_hashes()
    signature = {'reference': prep.name, 'diagnostic_only': True,
                 'independent_redocking': False, 'production_gate_released': False,
                 'reference_sha256': sha256(reference), 'receptor_sha256': sha256(receptor),
                 'prepared_pdb_sha256': sha256(prepared), 'preparation_manifest_sha256': sha256(prep_manifest),
                 'vina_sha256': sha256(vina), 'script_sha256': sha256(__file__),
                 'helper_sha256': sha256(Path(__file__).with_name('diagnose_nav15_native.py')),
                 'production_code_hashes': frozen, 'versions': versions(),
                 'scoring': 'vina default weights; original combined score not computed or modified',
                 'center': center.tolist(), 'size_A': 24., 'cpu_per_job': 2, 'seed': 20261021,
                 'native_box_face_margin_A': float(12. - np.abs(coords - center).max()),
                 'preparation': 'deposited heavy coordinates + AddHs + Meeko; no ligand embedding/minimization',
                 'limitations': 'Native-informed score/local diagnostic; neither pose recovery nor affinity validation. '
                                'Rigid protein-only and deposited ligand charge. No cross-reference affinity comparison.'}
    work_manifest(output, signature, resume)
    native_path = output / 'input/native.pdbqt'
    input_integrity = output / 'input_integrity.json'
    if input_integrity.exists():
        hashes = read_json(input_integrity)
        for name, digest in hashes.items():
            if sha256(output / 'input' / name) != digest:
                raise ValueError('Native input artifact changed')
    else:
        export_fixed_pose(ref, native_path)
        write_json(input_integrity, {p.name: sha256(p) for p in (output / 'input').iterdir() if p.is_file()})
    rows = []
    for mode in ('score_only', 'local_only'):
        job = output / mode
        result_path = job / 'result.json'
        if result_path.exists() and resume:
            previous = read_json(result_path)
            if previous['status'] == 'ok':
                for name, digest in previous['artifacts'].items():
                    if sha256(job / name) != digest:
                        raise ValueError(f'Completed diagnostic artifact changed: {job / name}')
                rows.append(previous)
                continue
        job.mkdir(parents=True, exist_ok=True)
        result = {'reference': prep.name, 'mode': mode, 'independent_redocking': False,
                  'production_gate_released': False, 'status': 'running',
                  'input_geometry': geometry(ref, ref, prepared)}
        write_json(result_path, result)
        command = [vina, '--receptor', receptor, '--ligand', native_path,
                   '--scoring', 'vina', '--' + mode, '--cpu', 2, '--seed', 20261021]
        for axis, value in zip('xyz', center):
            command += [f'--center_{axis}', float(value), f'--size_{axis}', 24.]
        pose = job / 'pose.pdbqt'
        if mode == 'local_only':
            if pose.exists():
                pose.unlink()  # Only this diagnostic's interrupted local pose.
            command += ['--out', pose]
        try:
            completed = logged_run(command, job / 'vina_log.json', timeout=300)
            result['affinity_kcal_mol'] = parse_affinity(completed.stdout)
            if mode == 'local_only':
                sdf, failures = RDKitMolCreate.write_sd_string(PDBQTMolecule(pose.read_text(), skip_typing=True))
                if failures or not sdf:
                    raise ValueError(f'Pose roundtrip failed: {failures}')
                pose.with_suffix('.sdf').write_text(sdf, encoding='utf-8')
                mol = Chem.SDMolSupplier(str(pose.with_suffix('.sdf')), removeHs=True)[0]
                if mol is None:
                    raise ValueError('Invalid local pose SDF')
                result['output_geometry'] = geometry(mol, ref, prepared)
                actual_size = float(np.ceil(24. / .375 / 2) * 2 * .375)
                result['actual_grid_face_margin_A'] = float(actual_size / 2 - np.abs(
                    mol.GetConformer().GetPositions() - center).max())
            result['status'] = 'ok'
            result['artifacts'] = {p.name: sha256(p) for p in job.iterdir()
                                   if p.is_file() and p.name != 'result.json'}
        except Exception as exc:
            result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(result_path, result)
        rows.append(result)
        write_json(output / 'results.json', rows)
        print(f'{prep.name} {mode}: {result["status"]}, affinity={result.get("affinity_kcal_mol")}, '
              f'local_RMSD={result.get("output_geometry", {}).get("fixed_frame_rmsd_A")}', flush=True)
    report = {'reference': prep.name, 'expected_jobs': 2, 'completed_jobs': len(rows),
              'successful_jobs': sum(r['status'] == 'ok' for r in rows),
              'diagnostic_only': True, 'independent_redocking': False, 'production_gate_released': False,
              'production_code_unchanged': frozen == code_hashes(), 'results': rows}
    write_json(output / 'results.json', rows)
    write_json(output / 'report.json', report)
    if not report['production_code_unchanged']:
        raise ValueError('Production code changed during diagnostic')
    return report['successful_jobs'] == 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, action='append', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    results = [native_diagnostic(p, args.resume) for p in args.reference_dir]
    sys.exit(0 if all(results) else 1)
