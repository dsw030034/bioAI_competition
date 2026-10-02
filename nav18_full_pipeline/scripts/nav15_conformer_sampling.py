"""Independent, energy-audited and geometry-deduplicated QDN input sampling.

Reference fixtures supply molecular graph and atom identity only. Their coordinates
are discarded before ETKDG. Neither receptor contacts nor experimental-pose RMSD
are used to generate, minimize or select conformers. MMFF energies compare
conformers within one chemical state and do not become docking scores.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdMolTransforms

from diagnose_nav15_matched_geometry import export_typed
from scgd.ligand_quality import geometry_issues
from scgd.runtime import sha256, versions, write_json

DEFAULT_SEEDS = (202610021, 202610022, 202610023, 202610024)
STATES = ('nav15', 'nav15_qdn_protonated')


def graph_only(source):
    """Keep ordered, stereo-preserving graph; eliminate all input coordinates."""
    graph = Chem.RemoveHs(Chem.Mol(source))
    graph.RemoveAllConformers()
    if graph.GetNumConformers() != 0:
        raise ValueError('Input conformers were not removed')
    return graph


def embed_seed_pool(source, seed, count):
    graph = graph_only(source)
    mol = Chem.AddHs(graph)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 1
    params.pruneRmsThresh = -1.0  # Deduplicate after complete MMFF optimization.
    params.enforceChirality = True
    params.clearConfs = True
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(count), params=params))
    return mol, ids


def aligned_heavy_rmsd(mol, left_id, right_id):
    """Symmetry-corrected heavy RMSD after alignment, without mutating the pool."""
    heavy = Chem.RemoveHs(mol)
    return float(rdMolAlign.GetBestRMS(Chem.Mol(heavy), heavy,
                                       prbId=left_id, refId=right_id,
                                       maxMatches=10000))


def deduplicate(mol, records, threshold=0.75, energy_window=10.0, max_selected=8):
    """Energy-ordered greedy representatives; retain all unique-geometry audit."""
    if threshold <= 0 or energy_window < 0 or max_selected < 1:
        raise ValueError('Invalid conformer-selection parameters')
    valid = sorted((r for r in records if r['status'] == 'geometry_passed'),
                   key=lambda r: (r['energy_kcal_mol'], r['conformer_id']))
    unique = []
    selected = []
    if not valid:
        return unique, selected
    minimum = valid[0]['energy_kcal_mol']
    for record in valid:
        record['relative_energy_same_state_kcal_mol'] = record['energy_kcal_mol'] - minimum
        comparisons = [(aligned_heavy_rmsd(mol, record['rdkit_conf_id'], prior['rdkit_conf_id']), prior)
                       for prior in unique]
        nearest = min(comparisons, key=lambda item: item[0]) if comparisons else None
        record['nearest_prior_representative_rmsd_A'] = nearest[0] if nearest else None
        if nearest is not None and nearest[0] < threshold:
            record.update(status='duplicate_geometry', duplicate_of=nearest[1]['conformer_id'])
            continue
        unique.append(record)
        if record['relative_energy_same_state_kcal_mol'] > energy_window:
            record['status'] = 'unique_outside_energy_window'
        elif len(selected) >= max_selected:
            record['status'] = 'unique_over_selection_cap'
        else:
            record['status'] = 'selected'
            selected.append(record)
    return unique, selected


def rmsd_matrix(mol, records):
    matrix = np.zeros((len(records), len(records)))
    for i, left in enumerate(records):
        for j in range(i):
            matrix[i, j] = matrix[j, i] = aligned_heavy_rmsd(
                mol, left['rdkit_conf_id'], records[j]['rdkit_conf_id'])
    return {'conformer_ids': [r['conformer_id'] for r in records], 'matrix_A': matrix.tolist()}


def ring_torsions(mol, conf_id):
    """Ordered ring-bond torsions; explicit atom indexes make them auditable."""
    quads = set()
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) < 4:
            continue
        for offset in range(len(ring)):
            quad = tuple(ring[(offset + i) % len(ring)] for i in range(4))
            quads.add(min(quad, tuple(reversed(quad))))
    conf = mol.GetConformer(conf_id)
    return [{'atoms': list(quad), 'torsion_deg': float(rdMolTransforms.GetDihedralDeg(conf, *quad))}
            for quad in sorted(quads)]


def single_conformer(mol, conf_id):
    copy = Chem.Mol(mol)
    copy.RemoveAllConformers()
    copy.AddConformer(Chem.Conformer(mol.GetConformer(conf_id)), assignId=True)
    return copy


def sample_state(name, output, *, seeds=DEFAULT_SEEDS, conformers_per_seed=128,
                 rmsd_threshold=0.75, energy_window=10.0, max_selected=8):
    """Return a serializable report with ``selected_entries[*]['ligand']`` paths.

    ``output`` is a fresh state-specific directory. Fixed seed pools use no
    experimental geometry. All valid optimized coordinates and all attempted
    energy/convergence/geometry records are retained, including duplicates.
    """
    if name not in STATES:
        raise ValueError(f'Unsupported QDN state: {name}')
    if conformers_per_seed < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Require positive pool size and unique seeds')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture = BASE / 'data/reference_validation' / f'{name}.sdf'
    source = Chem.SDMolSupplier(str(fixture), removeHs=True)[0]
    if source is None:
        raise ValueError(f'Invalid graph fixture: {fixture}')
    graph = graph_only(source)
    expected = Chem.MolToSmiles(graph, isomericSmiles=True)
    del source  # Reference coordinates are not used again by this module.
    pool = Chem.AddHs(graph)
    if not AllChem.MMFFHasAllMoleculeParams(pool):
        raise ValueError('Incomplete MMFF94s parameters')
    records = []
    batches = []
    for batch, seed in enumerate(seeds):
        embedded, ids = embed_seed_pool(graph, seed, conformers_per_seed)
        properties = AllChem.MMFFGetMoleculeProperties(embedded, mmffVariant='MMFF94s')
        batches.append({'seed': seed, 'attempted': conformers_per_seed, 'embedded': len(ids)})
        for within, conf_id in enumerate(ids):
            record = {'conformer_id': f'{name}_b{batch}_e{within:03d}', 'seed': seed,
                      'embedded_conf_id': conf_id, 'rdkit_conf_id': None,
                      'status': 'pending', 'energy_kcal_mol': None,
                      'relative_energy_same_state_kcal_mol': None,
                      'minimization_return_codes': [], 'geometry_issues': []}
            try:
                ff = AllChem.MMFFGetMoleculeForceField(embedded, properties, confId=conf_id)
                if ff is None:
                    raise ValueError('Cannot construct MMFF94s force field')
                for iterations in (1000, 3000):
                    code = int(ff.Minimize(maxIts=iterations))
                    record['minimization_return_codes'].append(code)
                    if code == 0:
                        break
                energy = float(ff.CalcEnergy())
                if np.isfinite(energy):
                    record['energy_kcal_mol'] = energy
                if code != 0:
                    record['status'] = 'optimization_not_converged'
                elif not np.isfinite(energy):
                    record['status'] = 'nonfinite_energy'
                else:
                    record['geometry_issues'] = geometry_issues(embedded, conf_id, expected)
                    record['status'] = 'geometry_failed' if record['geometry_issues'] else 'geometry_passed'
                # Preserve every embedded optimized geometry; valid entries go to SDF below.
                record['rdkit_conf_id'] = pool.AddConformer(
                    Chem.Conformer(embedded.GetConformer(conf_id)), assignId=True)
            except (ValueError, RuntimeError) as exc:
                record.update(status='optimization_error', error=f'{type(exc).__name__}: {exc}')
            records.append(record)
        for missing in range(conformers_per_seed - len(ids)):
            records.append({'conformer_id': f'{name}_b{batch}_not_embedded{missing:03d}',
                            'seed': seed, 'embedded_conf_id': None, 'rdkit_conf_id': None,
                            'status': 'embedding_failed', 'energy_kcal_mol': None,
                            'relative_energy_same_state_kcal_mol': None,
                            'minimization_return_codes': [], 'geometry_issues': []})
        print(f'{name}: seed {seed}, embedded {len(ids)}/{conformers_per_seed}; '
              f'{sum(r["status"] == "geometry_passed" for r in records)} valid so far', flush=True)
        write_json(output / 'progress_records.json', records)

    valid_ids = {r['conformer_id'] for r in records if r['status'] == 'geometry_passed'}
    converged = sum(bool(r['minimization_return_codes']) and r['minimization_return_codes'][-1] == 0
                    for r in records)
    unique, selected = deduplicate(pool, records, rmsd_threshold, energy_window, max_selected)
    valid_sdf = output / 'all_valid_optimized.sdf'
    with Chem.SDWriter(str(valid_sdf)) as writer:
        for record in records:
            if record['conformer_id'] not in valid_ids:
                continue
            mol = single_conformer(pool, record['rdkit_conf_id'])
            for key in ('conformer_id', 'seed', 'status', 'energy_kcal_mol', 'relative_energy_same_state_kcal_mol'):
                mol.SetProp(key, str(record[key]))
            writer.write(mol)
    entries = []
    for rank, record in enumerate(selected):
        mol = single_conformer(pool, record['rdkit_conf_id'])
        if Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True) != expected:
            raise ValueError('Selected conformer molecular identity changed')
        ligand = output / 'selected' / f'c{rank:02d}.pdbqt'
        audit = export_typed(mol, ligand)
        entries.append({'reference': name, 'repeat': rank, 'conformer_id': record['conformer_id'],
                        'seed': record['seed'], 'ligand': str(ligand),
                        'sdf': str(ligand.with_suffix('.sdf')),
                        'energy_kcal_mol': record['energy_kcal_mol'],
                        'relative_energy_same_state_kcal_mol': record['relative_energy_same_state_kcal_mol'],
                        'typing_audit': audit, 'ring_torsions': ring_torsions(pool, record['rdkit_conf_id'])})
    selected_matrix = rmsd_matrix(pool, selected)
    nembedded = sum(batch['embedded'] for batch in batches)
    report = {'reference': name, 'diagnostic_only': True, 'status': 'sampled' if entries else 'no_usable_conformer',
              'graph_isomeric_smiles': expected, 'formal_charge': Chem.GetFormalCharge(graph),
              'input_graph_sha256': sha256(fixture), 'script_sha256': sha256(__file__),
              'export_helper_sha256': sha256(Path(__file__).with_name('diagnose_nav15_matched_geometry.py')),
              'versions': versions(), 'reference_coordinates_used': False,
              'selection_uses_reference_rmsd': False, 'selection_uses_receptor': False,
              'sampling': {'algorithm': 'ETKDGv3', 'num_threads': 1, 'enforce_chirality': True,
                           'preoptimization_pruning': False, 'seeds': list(seeds),
                           'conformers_per_seed': conformers_per_seed, 'force_field': 'MMFF94s',
                           'full_geometry_minimization': True, 'max_iterations': [1000, 3000]},
              'selection': {'aligned_symmetry_heavy_rmsd_threshold_A': rmsd_threshold,
                            'energy_window_same_state_kcal_mol': energy_window, 'max_selected': max_selected,
                            'method': 'energy-order greedy geometry representatives; window and cap applied after deduplication',
                            'energy_enters_docking_score': False},
              'counts': {'attempted': len(records), 'embedded': nembedded, 'optimized': converged,
                         'invalid': nembedded-len(valid_ids), 'embedding_failed': len(records)-nembedded,
                         'valid': len(valid_ids), 'duplicates': len(valid_ids)-len(unique),
                         'unique': len(unique), 'window_excluded': sum(r['status'] == 'unique_outside_energy_window' for r in unique),
                         'selection_capped': sum(r['status'] == 'unique_over_selection_cap' for r in unique),
                         'selected': len(entries)},
              'batches': batches, 'selected_entries': entries,
              'selected_pairwise_aligned_heavy_rmsd': selected_matrix,
              'unique_pairwise_aligned_heavy_rmsd': rmsd_matrix(pool, unique),
              'all_valid_optimized_sdf': str(valid_sdf),
              'all_valid_optimized_sdf_sha256': sha256(valid_sdf),
              'limitations': 'Isolated-ligand MMFF94s energies only rank the input pool within one chemical state. '
                             'ETKDG and greedy RMSD representatives do not guarantee exhaustive ring sampling. '
                             'The graph is taken from experimental QDN, but all reference conformers are removed; '
                             'no reference-coordinate conformer is injected and selection is receptor independent.'}
    if len(entries) > 1:
        matrix = np.asarray(selected_matrix['matrix_A'])
        minimum_pair = float(matrix[np.triu_indices(len(entries), k=1)].min())
        report['selected_min_pairwise_rmsd_A'] = minimum_pair
        if minimum_pair < rmsd_threshold - 1e-8:
            raise ValueError('Selected conformers fail geometry deduplication')
    else:
        report['selected_min_pairwise_rmsd_A'] = None
    write_json(output / 'records.json', records)
    write_json(output / 'report.json', report)
    fields = ['conformer_id', 'seed', 'status', 'energy_kcal_mol', 'relative_energy_same_state_kcal_mol',
              'nearest_prior_representative_rmsd_A', 'duplicate_of']
    with (output / 'energy_audit.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    print(f'{name}: {len(valid_ids)} valid, {len(unique)} unique, {len(entries)} selected', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE / 'outputs/nav15_conformer_sampling_20261002')
    parser.add_argument('--state', choices=STATES, default='nav15')
    parser.add_argument('--num-confs', type=int, default=128)
    parser.add_argument('--max-selected', type=int, default=8)
    args = parser.parse_args()
    sample_state(args.state, args.output, conformers_per_seed=args.num_confs, max_selected=args.max_selected)
