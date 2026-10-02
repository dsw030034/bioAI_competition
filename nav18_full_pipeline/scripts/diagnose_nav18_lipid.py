"""Matched Nav1.8 control with a fixed, truncated experimental LPE A:2008.

This diagnostic does not alter the production scoring code or validation gate.
The phosphocholine ether CCD graph is cut only at the absent C16--C17 bond;
implicit hydrogens cap C16. All 22 observed heavy-atom coordinates are fixed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import urllib.request

import gemmi
from meeko import MoleculePreparation, PDBQTWriterLegacy
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from scgd.runtime import code_hashes, dock_job, read_json, resolve_vina, sha256, versions, write_json
from scgd.geometry import pdb_atoms, pose_contacts
from scgd.validation import assess_pose

CCD_URL = 'https://files.rcsb.org/ligands/download/LPE.cif'
DEFAULT_OUTPUT = BASE / 'outputs/nav18_lipid_control_20261002'
BASELINE = BASE / 'outputs/reference_validation_20261002_iofix'
RECEPTOR = BASE / 'outputs/receptors_20261002/nav18/receptor.pdbqt'
STATES = ('zwitterion', 'ccd_acid_plus1')
CENTER = (137.204, 136.509, 127.538)
SEEDS = (20261002, 20261003, 20261004)


def observed_lipid():
    rows = [line for line in (BASE / 'receptors/7WE4.pdb').read_text().splitlines()
            if line.startswith('HETATM') and line[17:20] == 'LPE' and line[21] == 'A'
            and line[22:26].strip() == '2008' and line[76:78].strip() not in ('H', 'D')]
    return {line[12:16].strip(): np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            for line in rows}


def build_lipid(cif_path, state):
    block = gemmi.cif.read_file(str(cif_path)).sole_block()
    atom_rows = list(block.find('_chem_comp_atom.', ['atom_id', 'type_symbol', 'charge', 'pdbx_stereo_config']))
    bond_rows = list(block.find('_chem_comp_bond.', ['atom_id_1', 'atom_id_2', 'value_order']))
    observed = observed_lipid()
    heavy_names = [row[0] for row in atom_rows if row[1] != 'H']
    missing = [name for name in heavy_names if name not in observed]
    assert len(observed) == 22 and len(heavy_names) == 34
    assert set(missing) == {f'C{i}' for i in range(17, 29)}, missing
    boundary = [tuple(row) for row in bond_rows if (row[0] in observed) != (row[1] in observed)
                and row[0] in heavy_names and row[1] in heavy_names]
    assert boundary == [('C16', 'C17', 'SING')], boundary
    rw, indices = Chem.RWMol(), {}
    for name, symbol, charge, _ in atom_rows:
        if name not in observed:
            continue
        atom = Chem.Atom(symbol)
        atom.SetFormalCharge(-1 if name == 'O32' and state == 'zwitterion' else int(charge))
        if name == 'O32' and state == 'zwitterion':
            atom.SetNoImplicit(True)
        atom.SetProp('ccd_atom_id', name)
        info = Chem.AtomPDBResidueInfo()
        info.SetName(f'{name:>4}')
        info.SetResidueName('LPE')
        info.SetResidueNumber(2008)
        info.SetChainId('A')
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)
        indices[name] = rw.AddAtom(atom)
    for first, second, order in bond_rows:
        if first in indices and second in indices:
            rw.AddBond(indices[first], indices[second], {'SING': Chem.BondType.SINGLE,
                       'DOUB': Chem.BondType.DOUBLE}[order])
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.Set3D(True)
    for name, idx in indices.items():
        conf.SetAtomPosition(idx, observed[name])
    mol.AddConformer(conf)
    Chem.AssignAtomChiralTagsFromStructure(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    assert mol.GetAtomWithIdx(indices['C2']).GetProp('_CIPCode') == 'R'
    assert mol.GetAtomWithIdx(indices['C16']).GetTotalNumHs() == 3
    assert Chem.GetFormalCharge(mol) == (0 if state == 'zwitterion' else 1)
    mol = Chem.AddHs(mol, addCoords=True, addResidueInfo=True)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            info = atom.GetPDBResidueInfo()
            info.SetName(f'H{atom.GetIdx() - len(observed) + 1:03d}')
    # Optimize added hydrogen positions only. Preserve every deposited heavy atom.
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
        raise ValueError('No MMFF parameters for truncated lipid')
    ff = AllChem.MMFFGetMoleculeForceField(mol, props)
    for idx in indices.values():
        ff.AddFixedPoint(idx)
    ff.Initialize()
    converged = ff.Minimize(maxIts=2000) == 0
    if not converged:
        raise ValueError('Fixed-heavy hydrogen optimization did not converge')
    max_drift = max(float(np.linalg.norm(mol.GetConformer().GetAtomPosition(idx) - xyz))
                    for name, xyz in observed.items() for idx in [indices[name]])
    assert max_drift < 1e-9
    setup = MoleculePreparation().prepare(mol)[0]
    pdbqt, ok, error = PDBQTWriterLegacy.write_string(setup)
    if not ok:
        raise ValueError(error)
    # Only typed coordinates enter the rigid receptor; ligand torsion directives do not.
    atom_lines = [line for line in pdbqt.splitlines() if line.startswith(('ATOM  ', 'HETATM'))]
    typed = {line[12:16].strip(): line.split()[-1] for line in atom_lines}
    assert typed['N'] == 'N' and typed['O32'] == 'OA'
    assert mol.GetAtomWithIdx(indices['N']).GetTotalNumHs(includeNeighbors=True) == 0
    serialized_heavy = {line[12:16].strip(): np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                        for line in atom_lines if line.split()[-1] not in ('H', 'HD')}
    assert set(serialized_heavy) == set(observed)
    assert all(np.array_equal(serialized_heavy[name], xyz) for name, xyz in observed.items())
    info = {'state': state, 'component_name': block.find_value('_chem_comp.name'),
            'ccd_formal_charge': int(block.find_value('_chem_comp.pdbx_formal_charge')),
            'formal_charge': Chem.GetFormalCharge(mol), 'formula': rdMolDescriptors.CalcMolFormula(mol),
            'canonical_smiles': Chem.MolToSmiles(Chem.RemoveHs(mol)), 'observed_heavy_atoms': len(observed),
            'full_ccd_heavy_atoms': len(heavy_names), 'omitted_atom_ids': missing,
            'only_cut_heavy_bond': boundary[0], 'cap': 'C16 terminal methyl; hydrogen only, no invented heavy atoms',
            'heavy_coordinate_max_drift_A': max_drift, 'pdbqt_heavy_coordinates_exact': True,
            'hydrogen_optimization_converged': converged, 'C2_CIP': 'R', 'atom_types': typed,
            'partial_charge_sum_after_H_merge': sum(float(line[70:76]) for line in atom_lines),
            'rationale': ('Primary diagnostic phosphocholine zwitterion: N+ quaternary ammonium, O32- phosphate. '
                          'Chemically plausible state, not a pKa or bound protonation determination.' if state == 'zwitterion'
                          else 'Sensitivity model follows CCD phosphate acid O32-H, net+1. Not asserted physiological.'),
            'vina_note': 'Default Vina uses atom types and donor hydrogens; Gasteiger partial charges are not an electrostatic term.'}
    return mol, atom_lines, info


def minimum_distances(mol, lipid):
    ligand_xyz = Chem.RemoveHs(mol).GetConformer().GetPositions()
    distances = np.linalg.norm(ligand_xyz[:, None, :] - np.array(list(lipid.values()))[None, :, :], axis=2)
    return {'min_lipid_heavy_distance_A': float(distances.min()),
            'lipid_heavy_pairs_below_2A': int((distances < 2.0).sum())}


def prepare(output):
    cif = output / 'LPE_ccd.cif'
    if not cif.exists():
        with urllib.request.urlopen(CCD_URL, timeout=30) as response:
            cif.write_bytes(response.read())
    write_json(output / 'ccd_provenance.json', {'url': CCD_URL, 'sha256': sha256(cif),
               'retrieved_utc': datetime.now(timezone.utc).isoformat()})
    original = RECEPTOR.read_text()
    original_lines = original.splitlines()
    serial_offset = max(int(line[6:11]) for line in original_lines if line.startswith(('ATOM', 'HETATM')))
    for state in STATES:
        folder = output / state
        folder.mkdir(exist_ok=True)
        mol, typed_lines, audit = build_lipid(cif, state)
        with Chem.SDWriter(str(folder / 'truncated_lpe.sdf')) as writer:
            mol.SetProp('model_label', 'Fixed truncated LPE A2008 diagnostic; not a complete experimental lipid')
            writer.write(mol)
        Chem.MolToPDBFile(mol, str(folder / 'truncated_lpe.pdb'))
        lipid_lines = ['ATOM  ' + f'{serial_offset + i:5d}' + line[11:] for i, line in enumerate(typed_lines, 1)]
        combined = original.rstrip('\r\n') + '\n' + '\n'.join(lipid_lines) + '\n'
        assert combined.splitlines()[:len(original_lines)] == original_lines
        assert not any(line.startswith(('ROOT', 'BRANCH', 'TORSDOF')) for line in combined.splitlines())
        (folder / 'receptor_with_lipid.pdbqt').write_text(combined)
        (folder / 'lipid_rigid.pdbqt').write_text('\n'.join(lipid_lines) + '\n')
        lipid_xyz = Chem.RemoveHs(mol).GetConformer().GetPositions()
        protein_xyz = np.array(list(pdb_atoms(RECEPTOR.parent / 'prepared.pdb').values()))
        distances = np.linalg.norm(lipid_xyz[:, None, :] - protein_xyz[None, :, :], axis=2)
        audit.update(original_protein_pdbqt_lines_unchanged=True,
                     min_lipid_protein_heavy_distance_A=float(distances.min()),
                     lipid_protein_heavy_pairs_below_2A=int((distances < 2).sum()),
                     receptor_sha256=sha256(folder / 'receptor_with_lipid.pdbqt'))
        reference = Chem.SDMolSupplier(str(BASE / 'data/reference_validation/nav18.sdf'))[0]
        audit['reference_lipid_geometry'] = minimum_distances(reference, observed_lipid())
        write_json(folder / 'preparation_audit.json', audit)


def baseline_rows():
    rows = []
    for row in read_json(BASELINE / 'results.json'):
        if row['reference'] != 'nav18':
            continue
        row = dict(row, state='no_lipid_baseline')
        path = BASELINE / 'jobs' / f"nav18_c{row['repeat']}_s{row['vina_seed']}" / 'poses.sdf'
        mol = Chem.SDMolSupplier(str(path), removeHs=True)[0]
        row.update(minimum_distances(mol, observed_lipid()))
        rows.append(row)
    return rows


def run(output, states, workers, resume):
    reference = Chem.SDMolSupplier(str(BASE / 'data/reference_validation/nav18.sdf'))[0]
    rows = baseline_rows()
    def one(state, repeat, seed):
        folder = output / state / 'jobs' / f'nav18_c{repeat}_s{seed}'
        folder.mkdir(parents=True, exist_ok=True)
        row = dict(reference='nav18', state=state, repeat=repeat, vina_seed=seed, status='running')
        try:
            pose = dock_job(BASELINE / 'ligands' / f'nav18_c{repeat}.pdbqt',
                            output / state / 'receptor_with_lipid.pdbqt', folder / 'poses.pdbqt',
                            CENTER, 24.0, exhaustiveness=8, num_modes=9, seed=seed, cpu=2,
                            timeout=1200, resume=resume)
            row.update(assess_pose(pose, reference, RECEPTOR.parent / 'prepared.pdb'))
            mol = Chem.SDMolSupplier(str(pose.with_suffix('.sdf')), removeHs=True)[0]
            row.update(minimum_distances(mol, observed_lipid()), status='ok')
            row['top1_passed_including_lipid'] = row['top1_passed'] and row['lipid_heavy_pairs_below_2A'] == 0
        except Exception as exc:
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(folder / 'result.json', row)
        return row
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, state, repeat, seed) for state in states for repeat in range(5) for seed in SEEDS]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_json(output / 'results.json', sorted(rows, key=lambda r: (r['state'], r['repeat'], r['vina_seed'])))
            print(f"{len(rows)-15}/{len(futures)} {row['state']} c{row['repeat']} s{row['vina_seed']} "
                  f"{row['status']} RMSD={row.get('top1_rmsd_A')} pass={row.get('top1_passed_including_lipid')}", flush=True)
    summaries = []
    for state in ['no_lipid_baseline', *states]:
        selected = [r for r in rows if r['state'] == state and r['status'] == 'ok']
        summaries.append({'state': state, 'expected': 15, 'successful': len(selected),
                          'top1_pass_count': sum(r['top1_passed'] for r in selected),
                          'top1_pass_including_lipid_count': sum(r['top1_passed'] and r['lipid_heavy_pairs_below_2A'] == 0 for r in selected),
                          'median_top1_rmsd_A': float(np.median([r['top1_rmsd_A'] for r in selected])) if selected else None,
                          'median_affinity_kcal_mol': float(np.median([r['affinity_kcal_mol'] for r in selected])) if selected else None,
                          'top1_lipid_overlap_count': sum(r['lipid_heavy_pairs_below_2A'] > 0 for r in selected)})
    report = {'diagnostic_only': True, 'production_gate_unchanged': True, 'summaries': summaries,
              'criteria': 'Unchanged RMSD<=2 A, no protein heavy pairs<2 A, contact recall>=0.5; additionally no lipid heavy pairs<2 A.',
              'limitations': ['Only the 22 observed lipid heavy atoms are retained; C16 capped with H, missing12 distal carbon atoms not modeled.',
                             'Lipid is fixed; this does not describe membrane dynamics or candidate-induced displacement.',
                             'Same inputs and seeds are matched explanatory controls, not held-out prospective validation.',
                             'The lipid state is an explicit model assumption; no bound-state protonation measurement is available.',
                             'Does not validate affinity/selectivity or authorize a full candidate screen.']}
    write_json(output / 'report.json', report)
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--states', nargs='+', choices=STATES, default=['zwitterion'])
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        parser.error('This control is capped at two workers (two CPU threads each).')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'script_sha256': sha256(__file__), 'production_code_hashes': code_hashes(),
                'receptor_sha256': sha256(RECEPTOR), 'raw_pdb_sha256': sha256(BASE / 'receptors/7WE4.pdb'),
                'baseline_results_sha256': sha256(BASELINE / 'results.json'),
                'ligand_sha256': {f'c{i}': sha256(BASELINE / 'ligands' / f'nav18_c{i}.pdbqt') for i in range(5)},
                'reference_sha256': sha256(BASE / 'data/reference_validation/nav18.sdf'),
                'vina_sha256': sha256(resolve_vina()), 'versions': versions(), 'states': args.states,
                'center': CENTER, 'size': 24.0, 'exhaustiveness': 8, 'num_modes': 9, 'cpu': 2,
                'seeds': SEEDS, 'input_conformers': 5}
    manifest = json.loads(json.dumps(manifest))
    mpath = args.output / 'manifest.json'
    if mpath.exists() and not args.resume:
        raise FileExistsError('Output already exists; use --resume only with identical inputs')
    if mpath.exists() and read_json(mpath) != manifest:
        raise ValueError('Inputs/settings/code changed; use a new output directory')
    write_json(mpath, manifest)
    if not (args.output / 'zwitterion/preparation_audit.json').exists():
        prepare(args.output)
    if args.prepare_only:
        print(json.dumps(read_json(args.output / 'zwitterion/preparation_audit.json'), indent=2))
    else:
        print(json.dumps(run(args.output, args.states, args.workers, args.resume), indent=2))
    if code_hashes() != manifest['production_code_hashes']:
        raise RuntimeError('Production scoring files changed during this diagnostic')


if __name__ == '__main__':
    main()
