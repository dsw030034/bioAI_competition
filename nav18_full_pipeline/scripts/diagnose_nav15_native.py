"""Score and locally optimize experimental QDN poses without changing production scoring.

This is a diagnostic, not an independent redocking validation or candidate gate.
All heavy input coordinates are preserved during hydrogen addition and typing.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate
from rdkit import Chem

import score
from scgd.geometry import fixed_frame_rmsd, pose_contacts
from scgd.runtime import BASE, code_hashes, logged_run, read_json, resolve_vina, sha256, versions, write_json
from scgd.validation import reference_molecule


def export_fixed_pose(mol, path):
    """Type a bound conformation; never embed or minimize the experimental heavy atoms."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    heavy = Chem.RemoveHs(mol)
    original = heavy.GetConformer().GetPositions().copy()
    hydrogenated = Chem.AddHs(heavy, addCoords=True)
    if not np.array_equal(original, hydrogenated.GetConformer().GetPositions()[:len(original)]):
        raise ValueError('Hydrogen addition moved observed heavy atoms')
    setups = MoleculePreparation().prepare(hydrogenated)
    if len(setups) != 1:
        raise ValueError('Unexpected multiple Meeko setups')
    text, ok, error = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise ValueError(error)
    path.write_text(text, encoding='utf-8')
    with Chem.SDWriter(str(path.with_suffix('.sdf'))) as writer:
        writer.write(hydrogenated)
    # PDBQT rounds coordinates to 0.001 A. Confirm its mapped round-trip geometry.
    sdf, failures = RDKitMolCreate.write_sd_string(PDBQTMolecule(text, skip_typing=True))
    if failures or not sdf:
        raise ValueError(f'PDBQT round-trip failed: {failures}')
    path.with_suffix('.roundtrip.sdf').write_text(sdf, encoding='utf-8')
    restored = Chem.SDMolSupplier(str(path.with_suffix('.roundtrip.sdf')))[0]
    drift = fixed_frame_rmsd(restored, heavy)
    if drift > 0.002:
        raise ValueError(f'Fixed-pose typing moved heavy coordinates: {drift}')
    audit = {'formal_charge': Chem.GetFormalCharge(heavy), 'smiles': Chem.MolToSmiles(heavy),
             'heavy_atoms': heavy.GetNumAtoms(), 'hydrogen_addition_max_heavy_displacement_A': 0.,
             'pdbqt_roundtrip_fixed_frame_rmsd_A': drift, 'preparation': 'AddHs coordinates + Meeko; no embedding or heavy-atom minimization',
             'pdbqt_sha256': sha256(path)}
    write_json(path.with_suffix('.preparation.json'), audit)
    return audit


def parse_affinity(stdout):
    match = re.search(r'^Affinity:\s*([-+\d.eE]+)', stdout, re.MULTILINE)
    if not match:
        match = re.search(r'^Estimated Free Energy of Binding\s*:\s*([-+\d.eE]+)', stdout, re.MULTILINE)
    if not match:
        raise ValueError('No Vina affinity in diagnostic stdout')
    value = float(match.group(1))
    if not np.isfinite(value):
        raise ValueError('Nonfinite diagnostic affinity')
    return value


def geometry(mol, ref, receptor_pdb):
    contacts = pose_contacts(mol, receptor_pdb)
    reference_contacts = set(pose_contacts(ref, receptor_pdb)['contact_residues_4A'])
    recall = len(reference_contacts & set(contacts['contact_residues_4A'])) / len(reference_contacts)
    rmsd = fixed_frame_rmsd(mol, ref)
    displacement = float(np.linalg.norm(Chem.RemoveHs(mol).GetConformer().GetPositions().mean(axis=0)
                        - Chem.RemoveHs(ref).GetConformer().GetPositions().mean(axis=0)))
    return {**contacts, 'fixed_frame_rmsd_A': rmsd, 'contact_recall': recall,
            'centroid_displacement_A': displacement,
            'geometry_passed': rmsd <= 2 and contacts['heavy_pairs_below_2A'] == 0 and recall >= .5}


def run(output, baseline, receptor_dir, vina, workers=2):
    output, baseline, receptor_dir = map(lambda p: Path(p).resolve(), (output, baseline, receptor_dir))
    output.mkdir(parents=True, exist_ok=False)
    vina = resolve_vina(vina)
    receptor = receptor_dir / 'receptor.pdbqt'
    receptor_pdb = receptor_dir / 'prepared.pdb'
    frozen = code_hashes()
    manifest = {'stage': 'nav15_native_pose_diagnostic', 'production_code_hashes': frozen,
                'script_sha256': sha256(__file__), 'receptor_sha256': sha256(receptor),
                'prepared_pdb_sha256': sha256(receptor_pdb), 'vina_sha256': sha256(vina),
                'versions': versions(), 'scoring': 'vina; default weights; original combined score not used or modified',
                'box': score.RECEPTORS['nav15'], 'cpu_per_job': 2, 'workers': workers,
                'seeds': [20261002, 20261003, 20261004],
                'hydrogen_policy': 'RDKit AddHs at fixed heavy coordinates; no H orientation enumeration',
                'scope': 'Native pose and c0_s20261002 global top/closest pose, neutral and +1 QDN; local optimization is diagnostic, not a redocking success.'}
    manifest['box'] = {k: str(v) if isinstance(v, Path) else v for k, v in manifest['box'].items()}
    write_json(output / 'manifest.json', manifest)
    baseline_rows = read_json(baseline / 'results.json')
    inputs = []
    for name in ('nav15', 'nav15_qdn_protonated'):
        ref = reference_molecule(name, score)
        row = next(r for r in baseline_rows if r['reference'] == name and r['repeat'] == 0 and r['vina_seed'] == 20261002)
        source = baseline / 'jobs' / f'{name}_c0_s20261002' / 'poses.sdf'
        poses = list(Chem.SDMolSupplier(str(source)))
        nearest = int(np.argmin(row['all_pose_rmsd_A']))
        for kind, mol, rank in [('native', ref, None), ('global_top1', poses[0], 1), ('global_closest', poses[nearest], nearest + 1)]:
            ligand = output / 'inputs' / f'{name}_{kind}.pdbqt'
            audit = export_fixed_pose(mol, ligand)
            item = {'reference': name, 'kind': kind, 'input_path': str(ligand), 'baseline_rank': rank,
                    'baseline_score': row['all_pose_affinity_kcal_mol'][rank - 1] if rank else None,
                    'source_sha256': sha256(source if rank else BASE / 'data/reference_validation' / f'{name}.sdf'),
                    'preparation': audit, 'input_geometry': geometry(mol, ref, receptor_pdb)}
            inputs.append(item)
    write_json(output / 'inputs.json', inputs)
    tasks = []
    for item in inputs:
        tasks.append((item, 'score_only', 20261002))
        # Repeated seed controls test deterministic local search; not independent validation trials.
        for seed in ([20261002, 20261003, 20261004] if item['kind'] == 'native' else [20261002]):
            tasks.append((item, 'local_only', seed))

    def one(task):
        item, mode, seed = task
        folder = output / 'jobs' / f"{item['reference']}_{item['kind']}_{mode}_s{seed}"
        folder.mkdir(parents=True)
        row = {**item, 'mode': mode, 'seed': seed, 'status': 'running'}
        write_json(folder / 'result.json', row)
        command = [vina, '--receptor', receptor, '--ligand', item['input_path'], '--scoring', 'vina',
                   '--' + mode, '--cpu', 2, '--seed', seed]
        cfg = score.RECEPTORS['nav15']
        for axis, center in zip('xyz', cfg['center']):
            command += [f'--center_{axis}', center, f'--size_{axis}', cfg['size']]
        pose = folder / 'pose.pdbqt'
        if mode == 'local_only':
            command += ['--out', pose]
        try:
            result = logged_run(command, folder / 'vina_log.json', timeout=180)
            row['affinity_kcal_mol'] = parse_affinity(result.stdout)
            if mode == 'local_only':
                sdf, failures = RDKitMolCreate.write_sd_string(PDBQTMolecule(pose.read_text(), skip_typing=True))
                if failures or not sdf:
                    raise ValueError(f'Local pose export failed: {failures}')
                pose.with_suffix('.sdf').write_text(sdf, encoding='utf-8')
                mol = Chem.SDMolSupplier(str(pose.with_suffix('.sdf')))[0]
                row['output_geometry'] = geometry(mol, reference_molecule(item['reference'], score), receptor_pdb)
                row['pose_sha256'] = sha256(pose)
            else:
                row['output_geometry'] = row['input_geometry']
            row['status'] = 'ok'
        except Exception as exc:
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(folder / 'result.json', row)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_json(output / 'results.json', rows)
            print(f"{len(rows)}/{len(tasks)} {row['reference']} {row['kind']} {row['mode']} {row['status']} score={row.get('affinity_kcal_mol')} error={row.get('error')}", flush=True)
    unchanged = code_hashes() == frozen
    summary = []
    for row in rows:
        g = row.get('output_geometry', {})
        summary.append({k: row.get(k) for k in ('reference', 'kind', 'mode', 'seed', 'status', 'affinity_kcal_mol')}
                       | {k: g.get(k) for k in ('fixed_frame_rmsd_A', 'centroid_displacement_A', 'contact_recall', 'heavy_pairs_below_2A', 'geometry_passed')})
    summary.sort(key=lambda r: (r['reference'], r['kind'], r['mode'], r['seed']))
    with (output / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    report = {'stage': 'diagnostic_only', 'attempted': len(tasks), 'successful': sum(r['status'] == 'ok' for r in rows),
              'production_code_unchanged': unchanged, 'summary': summary,
              'limitations': 'Known native pose is supplied; local stability is not independent redocking validation. Same scoring function but molecular geometries and unbound/internal-energy conventions can affect affinity comparisons. No pKa or H orientation ensemble. No candidate gate released.'}
    write_json(output / 'report.json', report)
    if not unchanged or report['successful'] != len(tasks):
        raise RuntimeError('Diagnostic incomplete or production scoring changed; inspect report')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE / 'outputs/nav15_native_diagnostic_20261002')
    parser.add_argument('--baseline', type=Path, default=BASE / 'outputs/reference_validation_20261002_iofix')
    parser.add_argument('--receptor-dir', type=Path, default=BASE / 'outputs/receptors_20261002/nav15')
    parser.add_argument('--vina')
    args = parser.parse_args()
    run(args.output, args.baseline, args.receptor_dir, args.vina)
