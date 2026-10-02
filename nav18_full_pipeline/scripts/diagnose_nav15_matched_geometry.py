"""Cross-place identical QDN conformations at two sites with matched hydrogen treatment.

2 chemical states x 2 source geometries x 2 H treatments x 2 placements;
each receives original Vina score_only and local_only. Diagnostic only.
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
from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

import score
from diagnose_nav15_native import geometry, parse_affinity
from scgd.geometry import fixed_frame_rmsd
from scgd.runtime import code_hashes, logged_run, read_json, resolve_vina, sha256, versions, write_json
from scgd.validation import reference_molecule

NAMES = ('nav15', 'nav15_qdn_protonated')
BASELINE = BASE / 'outputs/reference_validation_20261002_iofix'
RECEPTOR_DIR = BASE / 'outputs/receptors_20261002/nav15'


def reorder_to_reference(mol, reference):
    mol = Chem.RemoveHs(mol)
    if Chem.MolToSmiles(mol) != Chem.MolToSmiles(reference):
        raise ValueError('Different molecular identity or stereo')
    mapping = mol.GetSubstructMatch(reference, useChirality=True)
    if len(mapping) != reference.GetNumAtoms() or len(set(mapping)) != len(mapping):
        raise ValueError('Incomplete atom correspondence')
    reordered = Chem.RenumberAtoms(mol, list(mapping))
    if any(reordered.GetAtomWithIdx(i).GetAtomicNum() != a.GetAtomicNum() for i, a in enumerate(reference.GetAtoms())):
        raise ValueError('Atom correspondence changed elements')
    return reordered


def add_hydrogens(source, treatment):
    heavy = Chem.RemoveHs(source)
    before = heavy.GetConformer().GetPositions().copy()
    mol = Chem.AddHs(heavy, addCoords=True)
    record = {'treatment': treatment, 'environment': 'isolated ligand; no receptor-driven hydrogen selection',
              'minimization_codes': [], 'energy_kcal_mol': None}
    if treatment == 'fixed_heavy_mmff':
        if not AllChem.MMFFHasAllMoleculeParams(mol):
            raise ValueError('Incomplete MMFF parameters')
        properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
        ff = AllChem.MMFFGetMoleculeForceField(mol, properties)
        for index in range(heavy.GetNumAtoms()):
            ff.AddFixedPoint(index)
        ff.Initialize()
        for iterations in (1000, 3000):
            code = ff.Minimize(maxIts=iterations)
            record['minimization_codes'].append(code)
            if code == 0:
                break
        if code != 0 or not np.isfinite(ff.CalcEnergy()):
            raise ValueError('Hydrogen-only optimization failed')
        record['energy_kcal_mol'] = float(ff.CalcEnergy())
    elif treatment != 'rdkit_addhs':
        raise ValueError('Unknown hydrogen treatment')
    after = mol.GetConformer().GetPositions()[:len(before)]
    drift = float(np.linalg.norm(before-after, axis=1).max())
    if drift > 1e-8:
        raise ValueError('Hydrogen treatment moved heavy atoms')
    record['max_heavy_displacement_A'] = drift
    return mol, record


def place_rigid(mol, target):
    placed = Chem.Mol(mol)
    nheavy = target.GetNumAtoms()
    residual = float(rdMolAlign.AlignMol(placed, target, atomMap=[(i, i) for i in range(nheavy)]))
    before = mol.GetConformer().GetPositions()
    after = placed.GetConformer().GetPositions()
    distances = lambda xyz: np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    deformation = float(np.max(np.abs(distances(before)-distances(after))))
    if deformation > 1e-6:
        raise ValueError('Placement changed internal geometry')
    return placed, {'rigid_fit_to_site_pose_rmsd_A': residual, 'max_all_atom_internal_distance_change_A': deformation}


def export_typed(mol, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    setups = MoleculePreparation().prepare(mol)
    if len(setups) != 1:
        raise ValueError('Unexpected Meeko setup count')
    text, ok, error = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise ValueError(error)
    path.write_text(text, encoding='utf-8')
    with Chem.SDWriter(str(path.with_suffix('.sdf'))) as writer:
        writer.write(mol)
    sdf, failures = RDKitMolCreate.write_sd_string(PDBQTMolecule(text, skip_typing=True))
    if failures or not sdf:
        raise ValueError('PDBQT roundtrip failed')
    path.with_suffix('.roundtrip.sdf').write_text(sdf, encoding='utf-8')
    restored = Chem.SDMolSupplier(str(path.with_suffix('.roundtrip.sdf')))[0]
    drift = fixed_frame_rmsd(restored, Chem.RemoveHs(mol))
    if drift > .002:
        raise ValueError('PDBQT geometry drift')
    atoms = [line for line in text.splitlines() if line.startswith('ATOM')]
    return {'pdbqt_sha256': sha256(path), 'formal_charge': Chem.GetFormalCharge(mol),
            'pdbqt_roundtrip_rmsd_A': drift, 'atom_types': [line.split()[-1] for line in atoms],
            'torsdof': next(line for line in text.splitlines() if line.startswith('TORSDOF'))}


def run(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen = code_hashes()
    vina = resolve_vina()
    manifest = {'diagnostic_only': True, 'script_sha256': sha256(__file__),
                'native_helper_sha256': sha256(Path(__file__).with_name('diagnose_nav15_native.py')),
                'production_code_hashes': frozen, 'receptor_sha256': sha256(RECEPTOR_DIR/'receptor.pdbqt'),
                'prepared_pdb_sha256': sha256(RECEPTOR_DIR/'prepared.pdb'), 'vina_sha256': sha256(vina),
                'versions': versions(), 'box_center': list(score.RECEPTORS['nav15']['center']), 'box_size': 24,
                'cpu_per_job': 2, 'workers': 2, 'seed': 20261002,
                'design': 'same atom ordering, heavy conformation and hydrogen coordinates rigidly placed at native/wrong sites; native and wrong source geometries are each cross-placed. score_only then independent local_only.',
                'source_files': {}}
    entries = []
    refs = {}
    for name in NAMES:
        ref = reference_molecule(name, score)
        refs[name] = ref
        path = BASELINE / 'jobs' / f'{name}_c0_s20261002' / 'poses.sdf'
        wrong = reorder_to_reference(Chem.SDMolSupplier(str(path))[0], ref)
        manifest['source_files'][str(path.relative_to(BASE))] = sha256(path)
        refpath = BASE / 'data/reference_validation' / f'{name}.sdf'
        manifest['source_files'][str(refpath.relative_to(BASE))] = sha256(refpath)
        poses = {'native_site': ref, 'wrong_site': wrong}
        for source_name, source in [('native_geometry', ref), ('wrong_geometry', wrong)]:
            for hydrogen in ('rdkit_addhs', 'fixed_heavy_mmff'):
                mol, haudit = add_hydrogens(source, hydrogen)
                for site, target in poses.items():
                    placed, paudit = place_rigid(mol, target)
                    key = f'{name}_{source_name}_{hydrogen}_{site}'
                    ligand = output / 'inputs' / f'{key}.pdbqt'
                    audit = export_typed(placed, ligand)
                    entry = {'key': key, 'reference': name, 'source_geometry': source_name, 'hydrogen': hydrogen,
                             'site': site, 'ligand': str(ligand), 'hydrogen_audit': haudit,
                             'placement_audit': paudit, 'typing_audit': audit,
                             'input_geometry': geometry(placed, ref, RECEPTOR_DIR / 'prepared.pdb')}
                    entries.append(entry)
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'inputs.json', entries)

    def one(entry, mode):
        folder = output / 'jobs' / f"{entry['key']}_{mode}"
        folder.mkdir(parents=True)
        row = {k: entry[k] for k in ('key', 'reference', 'source_geometry', 'hydrogen', 'site', 'input_geometry')}
        row.update(mode=mode, status='running')
        write_json(folder / 'result.json', row)
        command = [vina, '--receptor', RECEPTOR_DIR / 'receptor.pdbqt', '--ligand', entry['ligand'],
                   '--scoring', 'vina', '--'+mode, '--cpu', 2, '--seed', 20261002]
        for axis, center in zip('xyz', score.RECEPTORS['nav15']['center']):
            command += [f'--center_{axis}', center, f'--size_{axis}', 24]
        pose = folder / 'pose.pdbqt'
        if mode == 'local_only':
            command += ['--out', pose]
        try:
            result = logged_run(command, folder / 'vina_log.json', timeout=180)
            row['affinity_kcal_mol'] = parse_affinity(result.stdout)
            if mode == 'local_only':
                sdf, failures = RDKitMolCreate.write_sd_string(PDBQTMolecule(pose.read_text(), skip_typing=True))
                if failures or not sdf:
                    raise ValueError('No local output geometry')
                pose.with_suffix('.sdf').write_text(sdf, encoding='utf-8')
                mol = Chem.SDMolSupplier(str(pose.with_suffix('.sdf')))[0]
                row['output_geometry'] = geometry(mol, refs[entry['reference']], RECEPTOR_DIR / 'prepared.pdb')
            else:
                row['output_geometry'] = entry['input_geometry']
            row['status'] = 'ok'
        except Exception as exc:
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(folder / 'result.json', row)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(one, entry, mode) for entry in entries for mode in ('score_only', 'local_only')]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_json(output/'results.json', rows)
            print(f"{len(rows)}/32 {row['key']} {row['mode']} {row['status']} score={row.get('affinity_kcal_mol')}", flush=True)
    summary = []
    for r in rows:
        g = r.get('output_geometry', {})
        summary.append({k: r.get(k) for k in ('reference', 'source_geometry', 'hydrogen', 'site', 'mode', 'status', 'affinity_kcal_mol')}
                       | {k: g.get(k) for k in ('fixed_frame_rmsd_A', 'contact_recall', 'heavy_pairs_below_2A')})
    summary.sort(key=lambda r: (r['reference'], r['source_geometry'], r['hydrogen'], r['mode'], r['site']))
    with (output/'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    pairs = []
    for name in NAMES:
        for source in ('native_geometry', 'wrong_geometry'):
            for hydrogen in ('rdkit_addhs', 'fixed_heavy_mmff'):
                for mode in ('score_only', 'local_only'):
                    selection = {r['site']: r for r in rows if r['reference']==name and r['source_geometry']==source and r['hydrogen']==hydrogen and r['mode']==mode and r['status']=='ok'}
                    if len(selection) != 2:
                        continue
                    n, w = selection['native_site'], selection['wrong_site']
                    pairs.append({'reference': name, 'source_geometry': source, 'hydrogen': hydrogen, 'mode': mode,
                                  'native_site_score': n['affinity_kcal_mol'], 'wrong_site_score': w['affinity_kcal_mol'],
                                  'wrong_minus_native_score': w['affinity_kcal_mol']-n['affinity_kcal_mol'],
                                  'native_endpoint_rmsd_A': n['output_geometry']['fixed_frame_rmsd_A'],
                                  'wrong_endpoint_rmsd_A': w['output_geometry']['fixed_frame_rmsd_A']})
    report = {'diagnostic_only': True, 'successful': sum(r['status']=='ok' for r in rows), 'attempted':32,
              'production_code_unchanged': code_hashes()==frozen, 'matched_pairs':pairs,
              'limitations': 'Rigid cross-placement can cause clashes; paired initial internal geometry is identical, local_only can relax torsions afterward. Two hydrogen treatments are not an exhaustive H-orientation ensemble. Native truth is supplied; no independent validation. No full screening gate released.'}
    write_json(output/'report.json', report)
    if report['successful']!=32 or not report['production_code_unchanged']:
        raise RuntimeError('Incomplete matched diagnostic')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE/'outputs/nav15_matched_geometry_20261002')
    args=parser.parse_args()
    run(args.output)
