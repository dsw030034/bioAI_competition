"""Repeat eight matched QDN local searches with a 28 A diagnostic box.

The corresponding 24 A results remain untouched. This examines local-search
boundary sensitivity; known site placement is supplied, so it is not redocking
validation and cannot release the screening gate.
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
from meeko import PDBQTMolecule, RDKitMolCreate

import score
from diagnose_nav15_native import geometry, parse_affinity
from scgd.runtime import code_hashes, logged_run, read_json, resolve_vina, sha256, versions, write_json
from scgd.validation import reference_molecule

MATCHED = BASE / 'outputs/nav15_matched_geometry_20261002'
RECEPTOR = BASE / 'outputs/receptors_20261002/nav15'


def box_margin(mol, center, size):
    """Signed distance to a requested box face, not the rounded grid extent.

    Positive means inside. A 28 A request may produce slightly wider affinity
    maps because Vina rounds the grid voxel count up at its 0.375 A spacing.
    """
    xyz = Chem.RemoveHs(mol).GetConformer().GetPositions()
    margins = float(size) / 2 - np.abs(xyz - np.asarray(center))
    return {'minimum_face_margin_A': float(margins.min()),
            'minimum_face_margin_by_axis_A': margins.min(axis=0).tolist(),
            'heavy_atoms_outside': int(np.any(margins < 0, axis=1).sum()),
            'heavy_atoms_within_0p05A_of_face': int(np.any(margins <= .05, axis=1).sum()),
            'coordinate_rounding_tolerance_A': .002}


def pdbqt_box_margins(path, center, size):
    atoms = [(np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]), line.split()[-1])
             for line in Path(path).read_text().splitlines() if line.startswith(('ATOM  ', 'HETATM'))]
    result = {}
    for label, selection in [('all_explicit', atoms),
                             ('heavy', [(xyz, kind) for xyz, kind in atoms if kind not in ('H', 'HD')]),
                             ('hydrogen', [(xyz, kind) for xyz, kind in atoms if kind in ('H', 'HD')])]:
        if not selection:
            result[label] = {'count': 0, 'minimum_face_margin_A': None}
            continue
        xyz = np.array([a[0] for a in selection])
        margins = float(size) / 2 - np.abs(xyz - np.asarray(center))
        result[label] = {'count': len(selection), 'minimum_face_margin_A': float(margins.min()),
                         'minimum_face_margin_by_axis_A': margins.min(axis=0).tolist(),
                         'atoms_outside': int(np.any(margins < 0, axis=1).sum()),
                         'atoms_within_0p05A_of_face': int(np.any(margins <= .05, axis=1).sum())}
    return result


def run(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen = code_hashes()
    vina = resolve_vina()
    center = list(score.RECEPTORS['nav15']['center'])
    entries = [r for r in read_json(MATCHED / 'inputs.json') if r['hydrogen'] == 'rdkit_addhs']
    originals = {r['key']: r for r in read_json(MATCHED / 'results.json')
                 if r['hydrogen'] == 'rdkit_addhs' and r['mode'] == 'local_only'}
    if len(entries) != 8 or len(originals) != 8 or any(r['status'] != 'ok' for r in originals.values()):
        raise ValueError('Eight successful matched source jobs required')
    refs = {name: reference_molecule(name, score) for name in ('nav15', 'nav15_qdn_protonated')}
    manifest = {'diagnostic_only': True, 'script_sha256': sha256(__file__),
                'native_helper_sha256': sha256(Path(__file__).with_name('diagnose_nav15_native.py')),
                'production_code_hashes': frozen, 'versions': versions(),
                'receptor_sha256': sha256(RECEPTOR / 'receptor.pdbqt'),
                'prepared_pdb_sha256': sha256(RECEPTOR / 'prepared.pdb'),
                'vina_sha256': sha256(vina), 'center': center,
                'original_box_size_A': 24, 'diagnostic_box_size_A': 28,
                'margin_convention': 'Signed margin to requested nominal box; positive inside. Vina rounds grid extent up by voxel spacing, so these are conservative margins.',
                'cpu_per_job': 2, 'workers': 2, 'seed': 20261002,
                'mode': 'local_only', 'scoring': 'vina, original default weights unchanged',
                'source_results_sha256': sha256(MATCHED / 'results.json'),
                'source_inputs_sha256': sha256(MATCHED / 'inputs.json'),
                'input_hashes': {r['key']: sha256(r['ligand']) for r in entries},
                'limitations': 'Only the search box size changes. Supplied native and wrong site placements are reused; this is not independent redocking validation.'}
    write_json(output / 'manifest.json', manifest)

    def one(entry):
        key = entry['key']
        folder = output / 'jobs' / key
        folder.mkdir(parents=True)
        old = originals[key]
        old_pose = MATCHED / 'jobs' / (key + '_local_only') / 'pose.sdf'
        old_mol = Chem.SDMolSupplier(str(old_pose))[0]
        if old_mol is None:
            raise ValueError('Unreadable source local endpoint')
        row = {k: entry[k] for k in ('key', 'reference', 'source_geometry', 'hydrogen', 'site')}
        row.update(status='running', original_24_score=old['affinity_kcal_mol'],
                   original_24_geometry=old['output_geometry'],
                   original_24_box=box_margin(old_mol, center, 24),
                   original_24_pdbqt_box=pdbqt_box_margins(old_pose.with_suffix('.pdbqt'), center, 24),
                   original_24_pose_sha256=sha256(old_pose),
                   input_pdbqt_sha256=sha256(entry['ligand']))
        write_json(folder / 'result.json', row)
        pose = folder / 'pose.pdbqt'
        command = [vina, '--receptor', RECEPTOR / 'receptor.pdbqt', '--ligand', entry['ligand'],
                   '--scoring', 'vina', '--local_only', '--cpu', 2, '--seed', 20261002, '--out', pose]
        for axis, value in zip('xyz', center):
            command += [f'--center_{axis}', value, f'--size_{axis}', 28]
        try:
            result = logged_run(command, folder / 'vina_log.json', timeout=180)
            row['affinity_kcal_mol'] = parse_affinity(result.stdout)
            sdf, failures = RDKitMolCreate.write_sd_string(PDBQTMolecule(pose.read_text(), skip_typing=True))
            if failures or not sdf:
                raise ValueError('No local output geometry')
            pose.with_suffix('.sdf').write_text(sdf, encoding='utf-8')
            mol = Chem.SDMolSupplier(str(pose.with_suffix('.sdf')))[0]
            if mol is None:
                raise ValueError('Unreadable output molecule')
            row['output_geometry'] = geometry(mol, refs[entry['reference']], RECEPTOR / 'prepared.pdb')
            row['output_28_box'] = box_margin(mol, center, 28)
            row['output_28_pdbqt_box'] = pdbqt_box_margins(pose, center, 28)
            row['output_in_original_24_box'] = box_margin(mol, center, 24)
            row['output_pdbqt_in_original_24_box'] = pdbqt_box_margins(pose, center, 24)
            row['score_28_minus_24'] = row['affinity_kcal_mol'] - row['original_24_score']
            row['pose_sha256'] = sha256(pose)
            row['status'] = 'ok'
        except Exception as exc:
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(folder / 'result.json', row)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(one, entry) for entry in entries]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_json(output / 'results.json', rows)
            print(f"{len(rows)}/8 {row['key']} {row['status']} 24={row['original_24_score']} 28={row.get('affinity_kcal_mol')}", flush=True)
    summary = []
    for row in sorted(rows, key=lambda r: r['key']):
        g, box = row.get('output_geometry', {}), row.get('output_28_box', {})
        summary.append({k: row.get(k) for k in ('reference', 'source_geometry', 'site', 'status',
                                              'original_24_score', 'affinity_kcal_mol', 'score_28_minus_24')}
                       | {k: g.get(k) for k in ('fixed_frame_rmsd_A', 'contact_recall', 'heavy_pairs_below_2A')}
                       | {'original_24_minimum_face_margin_A': row['original_24_box']['minimum_face_margin_A'],
                          'output_28_minimum_face_margin_A': box.get('minimum_face_margin_A'),
                          'output_minimum_face_margin_to_original_24_A': row.get('output_in_original_24_box', {}).get('minimum_face_margin_A')})
    with (output / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    report = {'diagnostic_only': True, 'attempted': 8, 'successful': sum(r['status'] == 'ok' for r in rows),
              'production_code_unchanged': code_hashes() == frozen, 'summary': summary,
              'limitations': 'Default scoring unchanged. Both 24 and 28 A searches are local and reference-informed; boundary sensitivity does not establish native-site preference or independent redocking success. No screening gate released.'}
    write_json(output / 'report.json', report)
    if report['successful'] != 8 or not report['production_code_unchanged']:
        raise RuntimeError('Boundary diagnostic incomplete or production code changed')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE / 'outputs/nav15_box_boundary_20261002')
    run(parser.parse_args().output)
