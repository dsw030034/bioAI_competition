"""Audit original Vina 1.2.7 receptor/ligand terms at fixed Nav1.5 poses.

This standalone diagnostic never imports or changes the production scorer. Its
five weighted pair terms reproduce the upstream XS pair potentials and the
32-per-squared-A lookup used by non_cache::eval. It checks every reconstructed
intermolecular sum against an unchanged Vina --score_only execution. Terms are
reported only when that comparison agrees within the printed log precision.
This is an attribution of the existing model, not a new binding-energy model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scgd.runtime import BASE, code_hashes, logged_run, read_json, resolve_vina, sha256, write_json


TERMS = ('gauss1', 'gauss2', 'repulsion', 'hydrophobic', 'hydrogen_bond')
WEIGHTS = np.array([-.035579, -.005156, .840245, -.035069, -.587439])
ELEMENT = {'A': 'C', 'NA': 'N', 'OA': 'O', 'SA': 'S', 'HD': 'H'}
COVALENT = {'C': .77, 'N': .75, 'O': .73, 'S': 1.02, 'P': 1.06,
            'H': .37, 'F': .71, 'Cl': .99, 'Br': 1.14, 'I': 1.33}
VDW = {'C': 1.9, 'N': 1.8, 'O': 1.7, 'S': 2., 'P': 2.1,
       'F': 1.5, 'Cl': 1.8, 'Br': 2., 'I': 2.2}


@dataclass
class Atom:
    serial: int
    name: str
    residue: str
    xyz: np.ndarray
    ad: str
    xs: str = ''

    @property
    def element(self):
        return ELEMENT.get(self.ad, self.ad)


def read_atoms(path):
    atoms = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(('ATOM  ', 'HETATM')):
            atoms.append(Atom(int(line[6:11]), line[12:16].strip(),
                              f'{line[21:22].strip()}:{line[17:20].strip()}:{line[22:27].strip()}',
                              np.array([float(line[i:i+8]) for i in (30, 38, 46)]), line[77:].strip()))
    if not atoms:
        raise ValueError(f'No atoms: {path}')
    unsupported = {a.element for a in atoms} - COVALENT.keys()
    if unsupported:
        raise ValueError(f'Unsupported atom elements, refusing implicit typing: {unsupported}')
    return atoms


def assign_xs(atoms):
    """Distance-perceive bonds using upstream covalent radii + no atom between.

    Receptor atoms are rigid. For these small-molecule diagnostic inputs, bond
    perception does not implement PDBQT mobility exclusions; exact scorer-total
    verification is required for each pose before attribution is released.
    """
    coords = np.array([a.xyz for a in atoms])
    tree = cKDTree(coords)
    bonds = [set() for _ in atoms]
    for i, j in tree.query_pairs(2 * max(COVALENT.values()) * 1.1):
        r2 = float(np.sum((coords[i] - coords[j]) ** 2))
        if r2 >= (1.1 * (COVALENT[atoms[i].element] + COVALENT[atoms[j].element])) ** 2:
            continue
        between = False
        for k in tree.query_ball_point(coords[i], r2 ** .5):
            if k not in (i, j) and np.sum((coords[i]-coords[k])**2) < r2 and np.sum((coords[j]-coords[k])**2) < r2:
                between = True
                break
        if not between:
            bonds[i].add(j)
            bonds[j].add(i)
    for i, atom in enumerate(atoms):
        e = atom.element
        donor = any(atoms[j].ad == 'HD' for j in bonds[i])
        acceptor = atom.ad in ('NA', 'OA')
        if e == 'C':
            atom.xs = 'C_P' if any(atoms[j].element not in ('H', 'C') for j in bonds[i]) else 'C_H'
        elif e in ('N', 'O'):
            atom.xs = e + '_' + ('DA' if donor and acceptor else 'D' if donor else 'A' if acceptor else 'P')
        elif e in ('F', 'Cl', 'Br', 'I'):
            atom.xs = e + '_H'
        else:
            atom.xs = e + '_P'
    return bonds


def pair_terms(a, b, r):
    """Five unweighted potentials; H excluded, atom-distance cutoff 8 A."""
    if a.element == 'H' or b.element == 'H' or r >= 8.:
        return np.zeros(5)
    d = r - VDW[a.element] - VDW[b.element]
    hydrophobic = a.xs.endswith('_H') and b.xs.endswith('_H')
    donor_a = a.xs.endswith(('_D', '_DA'))
    donor_b = b.xs.endswith(('_D', '_DA'))
    acceptor_a = a.xs.endswith(('_A', '_DA'))
    acceptor_b = b.xs.endswith(('_A', '_DA'))
    hbond = donor_a and acceptor_b or donor_b and acceptor_a
    return np.array([np.exp(-(d/.5)**2), np.exp(-((d-3)/2)**2), min(d, 0)**2,
                     float(np.clip(1.5-d, 0, 1)) if hydrophobic else 0.,
                     float(np.clip(-d/.7, 0, 1)) if hbond else 0.])


def fast_pair_terms(a, b, r2):
    """Upstream precalculate_element::fast averages endpoints in r^2 bins."""
    if r2 >= 64:
        return np.zeros(5)
    i = int(r2 * 32)
    return (pair_terms(a, b, np.sqrt(i/32)) + pair_terms(a, b, np.sqrt((i+1)/32))) * .5 * WEIGHTS


def decompose(ligand_atoms, receptor_atoms):
    rec_heavy = [a for a in receptor_atoms if a.element != 'H']
    tree = cKDTree(np.array([a.xyz for a in rec_heavy]))
    rows = []
    curled_atoms = []
    for ligand in ligand_atoms:
        if ligand.element == 'H':
            continue
        pending = []
        for j in tree.query_ball_point(ligand.xyz, 8):
            receptor = rec_heavy[j]
            r2 = float(np.sum((ligand.xyz-receptor.xyz)**2))
            terms = fast_pair_terms(ligand, receptor, r2)
            pending.append((receptor, r2, terms))
        total = sum(float(t.sum()) for _, _, t in pending)
        # authentic_v[1] = 1000 in Vina::score; non_cache curls each ligand atom.
        curl_scale = 1000/(1000+total) if total > 0 else 1.
        if curl_scale != 1:
            curled_atoms.append({'ligand_serial': ligand.serial, 'unscaled_total': total, 'scale': curl_scale})
        for receptor, r2, terms in pending:
            terms *= curl_scale
            rows.append({'ligand_serial': ligand.serial, 'ligand_name': ligand.name, 'ligand_xs': ligand.xs,
                         'receptor_serial': receptor.serial, 'receptor_name': receptor.name,
                         'residue': receptor.residue, 'receptor_xs': receptor.xs, 'distance_A': r2**.5,
                         **dict(zip(TERMS, map(float, terms))), 'sum_kcal_mol': float(terms.sum())})
    totals = {name: sum(row[name] for row in rows) for name in TERMS}
    return rows, totals, curled_atoms


def parse_energies(stdout):
    names = {'affinity': 'Estimated Free Energy of Binding', 'intermolecular': 'Final Intermolecular Energy',
             'internal': 'Final Total Internal Energy', 'torsional': 'Torsional Free Energy',
             'unbound': "Unbound System's Energy"}
    return {key: float(re.search(re.escape(label) + r'\s*:\s*([-\d.]+)', stdout).group(1)) for key, label in names.items()}


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output, source, receptor, vina=None, matched=None, resume=False):
    output, source, receptor = map(lambda x: Path(x).resolve(), (output, source, receptor))
    if (output/'report.json').exists():
        raise FileExistsError(f'Refusing to overwrite completed diagnostic: {output}')
    output.mkdir(parents=True, exist_ok=True)
    frozen = code_hashes()
    vina = resolve_vina(vina)
    if (output/'manifest.json').exists():
        previous = read_json(output/'manifest.json')
        if not resume or previous['production_code_hashes'] != frozen or previous['receptor_sha256'] != sha256(receptor) or previous['vina_sha256'] != sha256(vina):
            raise ValueError('Existing diagnostic requires --resume and unchanged production/receptor/Vina.')
    rec = read_atoms(receptor)
    assign_xs(rec)
    manifest = {'diagnostic': 'Nav1.5 fixed-pose attribution of original Vina1.2.7 terms',
                'production_code_hashes': frozen, 'script_sha256': sha256(__file__),
                'receptor_path': str(receptor), 'receptor_sha256': sha256(receptor), 'vina_sha256': sha256(vina),
                'upstream': 'https://github.com/ccsb-scripps/AutoDock-Vina/tree/v1.2.7/src/lib',
                'upstream_files': {p.name: sha256(p) for p in sorted((output/'upstream_vina_1.2.7').glob('*'))},
                'weights': dict(zip(TERMS, map(float, WEIGHTS))), 'cpu_per_job': 1,
                'scope': 'Original scorer, default weights; no docking, no optimization, no candidate scoring.',
                'numerical_verification_tolerance_kcal_mol': .001,
                'curl_attribution': 'When per-ligand-atom sum >0, distribute upstream 1000/(1000+sum) factor proportionally across pair terms.',
                'xs_typing_scope': 'Receptor-distance bonds; ligand distance bonds without mobility exclusions. Every pose requires independent Vina total agreement.'}
    write_json(output/'manifest.json', manifest)
    write_csv(output/'receptor_types.csv', [{'serial': a.serial, 'name': a.name, 'residue': a.residue, 'AD_type': a.ad, 'XS_type': a.xs} for a in rec])
    inputs = []
    for reference, kind in itertools.product(('nav15', 'nav15_qdn_protonated'), ('native', 'global_top1')):
        inputs.append((reference+'_'+kind+'_fixed', source/'inputs'/f'{reference}_{kind}.pdbqt'))
        inputs.append((reference+'_'+kind+'_local', source/'jobs'/f'{reference}_{kind}_local_only_s20261002'/'pose.pdbqt'))
    if matched:
        inputs.extend(('matched_'+p.parent.name.removesuffix('_local_only'), p)
                      for p in sorted((Path(matched)/'jobs').glob('*_rdkit_addhs_*_local_only/pose.pdbqt')))
    summaries = []
    residue_rows = []
    for name, ligand in inputs:
        folder = output/name
        folder.mkdir(exist_ok=resume)
        atoms = read_atoms(ligand)
        assign_xs(atoms)
        pairs, totals, curled = decompose(atoms, rec)
        center = np.array([128.691875,125.578875,137.582083])
        max_extent = max(float(np.max(np.abs(a.xyz-center))) for a in atoms if a.element != 'H')
        # Saved coordinates round to .001 A; an edge pose may round just outside.
        # Fixed-pose scoring only: pad, never move/optimize/redock the pose here.
        box_size = 24. if max_extent <= 12 else 2*np.ceil(max_extent/.375)*.375
        command = [vina, '--receptor', receptor, '--ligand', ligand, '--scoring', 'vina', '--score_only', '--cpu', 1,
                   '--seed', 20261002, '--center_x', 128.691875, '--center_y', 125.578875, '--center_z', 137.582083,
                   '--size_x', box_size, '--size_y', box_size, '--size_z', box_size]
        previous_log = read_json(folder/'vina_log.json') if (folder/'vina_log.json').exists() else None
        previous_row = read_json(folder/'result.json') if (folder/'result.json').exists() else None
        if resume and previous_log and previous_log['status'] == 'ok' and previous_log['command'] == list(map(str,command)) and previous_row and previous_row['ligand_sha256'] == sha256(ligand):
            stdout = previous_log['stdout']
        else:
            if previous_log:
                write_json(folder/'prior_vina_log.json', previous_log)
            stdout = logged_run(command, folder/'vina_log.json', timeout=180).stdout
        energies = parse_energies(stdout)
        reconstructed = sum(totals.values())
        error = reconstructed - energies['intermolecular']
        verified = abs(error) <= .001
        row = {'pose': name, 'ligand_path': str(ligand), 'ligand_sha256': sha256(ligand),
               'fixed_pose_score_box_size_A': box_size, 'max_distance_from_box_center_A': max_extent,
               **{f'vina_{key}_kcal_mol': value for key, value in energies.items()},
               **totals, 'reconstructed_intermolecular_kcal_mol': reconstructed,
               'difference_from_vina_log_kcal_mol': error, 'verified': verified,
               'pair_count_below_4A': sum(p['distance_A']<4 for p in pairs),
               'residues_below_4A': ','.join(sorted({p['residue'] for p in pairs if p['distance_A']<4}))}
        write_json(folder/'result.json', row | {'positive_atom_curl_scaling': curled, 'atom_type_counts': dict(Counter(a.xs for a in atoms))})
        if verified:
            write_csv(folder/'pair_terms.csv', pairs)
            by_residue = defaultdict(lambda: np.zeros(5))
            for p in pairs:
                by_residue[p['residue']] += [p[t] for t in TERMS]
            res_rows = [{'pose': name, 'residue': residue, **dict(zip(TERMS,map(float, term))), 'sum_kcal_mol': float(term.sum())}
                        for residue, term in by_residue.items()]
            res_rows.sort(key=lambda r:r['sum_kcal_mol'])
            write_csv(folder/'residue_terms.csv', res_rows)
            residue_rows.extend(res_rows)
        summaries.append(row)
        print(name, 'inter=', energies['intermolecular'], 'reconstructed=',round(reconstructed,6), 'error=',round(error,6), 'verified=',verified,flush=True)
    write_csv(output/'summary.csv', summaries)
    write_csv(output/'residue_terms.csv', residue_rows)
    report = {'all_verified': all(r['verified'] for r in summaries), 'production_code_unchanged': frozen == code_hashes(),
              'attempted_score_only': len(inputs), 'summary': summaries,
              'limitations': ['Term attribution describes Vina preference, not measured binding energy.',
                             'Gaussian terms reward packing and are not uniquely hydrophobic contacts.',
                             'No explicit electrostatic term or directional hydrogen-bond energy in this scorer.',
                             'Terms decompose intermolecular energy only. Torsional/internal/unbound entries are taken from Vina.',
                             'Native/global poses differ in internal geometry; common-geometry controls are a separate experiment.',
                             'Local results are rescores of rounded saved coordinates; expect small changes relative to original local logs.']}
    write_json(output/'report.json', report)
    if not report['all_verified'] or not report['production_code_unchanged']:
        raise RuntimeError('Reconstruction or immutability verification failed; inspect report before interpreting.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE/'outputs/nav15_terms_20261002')
    parser.add_argument('--source', type=Path, default=BASE/'outputs/nav15_native_diagnostic_20261002')
    parser.add_argument('--receptor', type=Path, default=BASE/'outputs/receptors_20261002/nav15/receptor.pdbqt')
    parser.add_argument('--vina')
    parser.add_argument('--matched', type=Path, default=BASE/'outputs/nav15_matched_geometry_20261002')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.output,args.source,args.receptor,args.vina,args.matched,args.resume)
