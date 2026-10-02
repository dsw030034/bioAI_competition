"""Prepare an auditable rat 6UZ0/flecainide reference; no docking or score changes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import groupby
import json
from pathlib import Path
import sys
import urllib.request

import gemmi
import numpy as np
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scgd.receptors import prepare_receptor
from scgd.geometry import pdb_atoms
from scgd.runtime import code_hashes, logged_run, read_json, sha256, write_json


def download(url, path):
    path = Path(path)
    if not path.exists():
        request = urllib.request.Request(url, headers={'User-Agent': 'Nav15-reference-diagnostic/1.0'})
        with urllib.request.urlopen(request, timeout=120) as response:
            path.write_bytes(response.read())
    return {'url': url, 'path': str(path.resolve()), 'sha256': sha256(path)}


def ccd_reference(ccd_path, pdb_path, ligand_id='K4D'):
    """Named CCD heavy graph plus observed coordinates, not CCD ideal geometry."""
    block = gemmi.cif.read_file(str(ccd_path)).sole_block()
    table = block.find('_chem_comp_atom.', ['atom_id', 'type_symbol', 'charge', 'pdbx_stereo_config'])
    atom_rows = [list(row) for row in table if row[1] != 'H']
    raw = [line for line in Path(pdb_path).read_text().splitlines()
           if line.startswith('HETATM') and line[17:20] == ligand_id]
    deposited_hydrogens = sum(line[76:78].strip() in ('H', 'D') for line in raw)
    raw = [line for line in raw if line[76:78].strip() not in ('H', 'D')]
    residue_keys = {(line[21], line[22:27].strip()) for line in raw}
    if len(residue_keys) != 1:
        raise ValueError(f'Expected one {ligand_id} residue, found {residue_keys}')
    coords = {}
    for line in raw:
        name = line[12:16].strip()
        if name in coords:
            raise ValueError(f'Duplicate/alternate {ligand_id} atom {name}')
        coords[name] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    names = [row[0] for row in atom_rows]
    if set(coords) != set(names):
        raise ValueError(f'Incomplete ligand; missing={set(names)-set(coords)} extra={set(coords)-set(names)}')
    rw = Chem.RWMol()
    for name, element, charge, stereo in atom_rows:
        atom = Chem.Atom(element)
        atom.SetFormalCharge(int(charge))
        atom.SetProp('ccd_atom_name', name)
        atom.SetProp('ccd_stereo_config', stereo)
        rw.AddAtom(atom)
    index = {name: i for i, name in enumerate(names)}
    types = {'SING': Chem.BondType.SINGLE, 'DOUB': Chem.BondType.DOUBLE,
             'TRIP': Chem.BondType.TRIPLE, 'AROM': Chem.BondType.AROMATIC}
    for row in block.find('_chem_comp_bond.', ['atom_id_1', 'atom_id_2', 'value_order']):
        first, second, order = list(row)
        if first in index and second in index:
            rw.AddBond(index[first], index[second], types[order])
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    conf = Chem.Conformer(len(names))
    for i, name in enumerate(names):
        conf.SetAtomPosition(i, coords[name])
    mol.AddConformer(conf)
    # The deposited CCD stereochemistry must be consistent with the actual pose.
    # This does not establish enantiopure experimental dosing of marketed flecainide.
    Chem.AssignStereochemistryFrom3D(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    stereo_audit = []
    for i, (name, element, charge, expected) in enumerate(atom_rows):
        atom = mol.GetAtomWithIdx(i)
        observed = atom.GetProp('_CIPCode') if atom.HasProp('_CIPCode') else None
        if expected in ('R', 'S') and observed != expected:
            raise ValueError(f'CCD versus 3D stereochemistry mismatch: {name}: {expected} vs {observed}')
        if expected in ('R', 'S') or observed:
            stereo_audit.append({'atom_name': name, 'ccd': expected, 'experimental_3d': observed})
    mol.SetProp('_Name', f'{Path(pdb_path).stem}_{ligand_id}_experimental_pose_CCD')
    return mol, {'residue': list(next(iter(residue_keys))), 'heavy_atoms': len(names),
                 'deposited_hydrogens_excluded_from_heavy_reference': deposited_hydrogens,
                 'atom_names': names, 'complete_ccd_heavy_graph': True,
                 'stereochemistry': stereo_audit,
                 'coordinates': f'{Path(pdb_path).stem} deposited experimental PDB coordinates, exact atom-name matching; no ideal coordinates or minimization',
                 'stereo_caveat': f'This validates the deposited {ligand_id} model, not enantiopure dosing or clinical enantiomer selectivity.'}


def hetero_audit(pdb_path, ligand_xyz, ligand_id='K4D'):
    groups = defaultdict(list)
    for line in Path(pdb_path).read_text().splitlines():
        if not line.startswith('HETATM') or line[17:20] == ligand_id or line[76:78].strip() == 'H':
            continue
        groups[(line[21], line[22:27].strip(), line[17:20])].append(
            (float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return sorted([{'chain': key[0], 'residue': key[1], 'name': key[2], 'observed_heavy_atoms': len(xyz),
                    'minimum_ligand_distance_A': float(np.linalg.norm(np.asarray(xyz)[:, None] - ligand_xyz[None, :], axis=2).min()),
                    'excluded_from_protein_only_receptor': True}
                   for key, xyz in groups.items()], key=lambda x: x['minimum_ligand_distance_A'])


def type_with_reviewed_false_bonds(input_path, output, observed, center, ligand_xyz):
    """Use Meeko templates after removing seven explicitly inspected spurious links."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy, Polymer, ResidueChemTemplates
    from meeko.polymer import find_inter_mols_bonds
    reviewed_pairs = {frozenset(pair) for pair in
                      [('A:189', 'A:195'), ('A:204', 'A:222'), ('A:307', 'A:311'),
                       ('A:421', 'A:425'), ('A:1364', 'A:1439'), ('A:1512', 'A:1587'),
                       ('A:1724', 'A:1730')]}
    parsed = Polymer._pdb_to_residue_mols(input_path.read_text(), None, 'A')
    if any(data[0] is None or data[2] or data[3] for data in parsed.values()):
        raise ValueError('Unexpected unparsed residue/alternate location in explicitly reviewed receptor')
    raw = {key: (data[0], data[1]) for key, data in parsed.items()}
    bonds = find_inter_mols_bonds(raw)
    removed = []
    seen = set()
    for pair, atom_pairs in list(bonds.items()):
        if frozenset(pair) not in reviewed_pairs:
            continue
        seen.add(frozenset(pair))
        for indices in atom_pairs:
            names = [raw[resid][0].GetAtomWithIdx(index).GetPDBResidueInfo().GetName().strip()
                     for resid, index in zip(pair, indices)]
            if set(names) == {'C', 'N'} or names == ['SG', 'SG']:
                raise ValueError(f'Reviewed link unexpectedly a peptide/disulfide: {pair}, {names}')
            xyz = [raw[resid][0].GetConformer().GetPositions()[index]
                   for resid, index in zip(pair, indices)]
            removed.append({'residue_pair': list(pair), 'atom_names': names,
                            'distance_A': float(np.linalg.norm(xyz[0] - xyz[1])),
                            'atoms_observed_in_source': [
                                (resid.split(':')[0], resid.split(':')[1], raw[resid][1], name) in observed
                                for resid, name in zip(pair, names)],
                            'atom_distances_to_box_A': [float(np.linalg.norm(np.maximum(np.abs(pos-center)-12,0))) for pos in xyz],
                            'atom_minimum_ligand_distances_A': [float(np.linalg.norm(ligand_xyz-pos,axis=1).min()) for pos in xyz],
                            'reason': 'Distance-only inter-residue covalent perception caused by a steric overlap; these canonical residues have no such covalent link. No coordinates moved.'})
        del bonds[pair]
    if seen != reviewed_pairs:
        raise ValueError(f'The expected explicitly reviewed false links changed: {reviewed_pairs - seen}')
    write_json(output / 'reviewed_false_bonds.json', removed)
    polymer = Polymer(raw, bonds, ResidueChemTemplates.create_from_defaults(),
                      MoleculePreparation(), forgive_extra_bonds=False)
    ignored = polymer.get_ignored_monomers()
    if ignored:
        raise ValueError(f'Receptor still has unmatched residues: {list(ignored)}')
    name_restorations = []
    for residue_id, monomer in polymer.monomers.items():
        if 'OXT' not in monomer.atom_names:
            continue
        coords = monomer.rdkit_mol.GetConformer().GetPositions()
        template_o_index = monomer.atom_names.index('O')
        template_oxt_index = monomer.atom_names.index('OXT')
        source_key = (residue_id.split(':')[0], residue_id.split(':')[1], monomer.input_resname, 'O')
        if source_key not in observed:
            continue
        if np.linalg.norm(coords[template_o_index] - observed[source_key]) < .002:
            continue
        if np.linalg.norm(coords[template_oxt_index] - observed[source_key]) > .002:
            raise ValueError(f'Terminal oxygen name mismatch is not the symmetric O/OXT swap: {source_key}')
        monomer.atom_names = list(monomer.atom_names)
        monomer.atom_names[template_o_index], monomer.atom_names[template_oxt_index] = 'OXT', 'O'
        monomer._set_pdbinfo(residue_id)
        name_restorations.append({'residue': residue_id, 'operation': 'restore original O/OXT labels after symmetric carboxylate template mapping; coordinates and atom parameters unchanged'})
    write_json(output / 'terminal_oxygen_name_restorations.json', name_restorations)
    rigid, flexible = PDBQTWriterLegacy.write_from_polymer(polymer)
    if flexible:
        raise ValueError('Unexpected flexible receptor output')
    (output / 'receptor.pdbqt').write_text(rigid)
    (output / 'prepared.pdb').write_text(polymer.to_pdb())
    (output / 'receptor.json').write_text(polymer.to_json())
    return removed


def recover_added_remote_oxt(original, center, reference, reviewed_residues=()):
    """Repair only newly built, distant malformed OXT atoms; preserve observed atoms."""
    original = Path(original)
    output = original.with_name('receptor_oxt_repaired')
    output.mkdir(exist_ok=True)
    audit = read_json(original / 'audit.json')
    observed = pdb_atoms(original / 'selected.pdb')
    repaired = pdb_atoms(original / 'repaired.pdb')
    ligand_xyz = Chem.SDMolSupplier(str(reference))[0].GetConformer().GetPositions()
    changes = {}
    for key, xyz in repaired.items():
        if key in observed or key[-1] != 'OXT':
            continue
        prefix = key[:-1]
        carbon, oxygen, alpha = (repaired[prefix + (name,)] for name in ('C', 'O', 'CA'))
        co_length = float(np.linalg.norm(xyz - carbon))
        oo_length = float(np.linalg.norm(xyz - oxygen))
        if oo_length >= 1.8 and 1.1 <= co_length <= 1.4:
            continue
        residue_xyz = np.asarray([pos for atom_key, pos in repaired.items() if atom_key[:-1] == prefix])
        distance = float(np.linalg.norm(np.maximum(np.abs(residue_xyz - center) - 12, 0), axis=1).min())
        reviewed = f'{key[0]}:{key[1]}' in reviewed_residues
        if distance <= 8 and not reviewed:
            raise ValueError(f'Malformed added OXT near search box: {key}; manual review required')
        first, second = oxygen - carbon, alpha - carbon
        direction = -(first / np.linalg.norm(first) + second / np.linalg.norm(second))
        replacement = carbon + 1.25 * direction / np.linalg.norm(direction)
        if np.linalg.norm(replacement - oxygen) < 1.8:
            raise ValueError(f'Invalid terminal carboxyl geometry after repair: {key}')
        other_heavy = np.asarray([pos for atom_key, pos in repaired.items()
                                 if atom_key != key and atom_key != prefix + ('C',)])
        min_other_heavy = float(np.linalg.norm(other_heavy - replacement, axis=1).min())
        if min_other_heavy < 1.8:
            raise ValueError(f'New terminal oxygen clashes with a nonbonded heavy atom: {key}, {min_other_heavy}')
        changes[key] = {'key': list(key), 'old_coordinates': xyz.tolist(),
                        'new_coordinates': replacement.tolist(), 'old_C_OXT_A': co_length,
                        'old_O_OXT_A': oo_length, 'new_C_OXT_A': float(np.linalg.norm(replacement-carbon)),
                        'new_O_OXT_A': float(np.linalg.norm(replacement-oxygen)),
                        'minimum_residue_distance_to_box_A': distance,
                        'source_residue_minimum_ligand_distance_A': float(np.linalg.norm(residue_xyz[:, None] - ligand_xyz[None, :], axis=2).min()),
                        'old_OXT_minimum_ligand_distance_A': float(np.linalg.norm(ligand_xyz-xyz, axis=1).min()),
                        'new_OXT_minimum_ligand_distance_A': float(np.linalg.norm(ligand_xyz-replacement, axis=1).min()),
                        'new_OXT_minimum_other_heavy_excluding_bonded_C_A': min_other_heavy,
                        'explicitly_reviewed_exception_to_remote_box_rule': reviewed,
                        'reason': 'PDBFixer-added terminal oxygen had an unphysical local contact/bond; move only this unobserved atom to the carboxyl plane'}
    if not changes:
        raise ValueError('Failed receptor preparation cannot be resolved by audited distant added OXT repair')
    lines = []
    for line in (original / 'repaired.pdb').read_text().splitlines():
        if line.startswith('ATOM  '):
            key = (line[21], line[22:27].strip(), line[17:20].strip(), line[12:16].strip())
            if key in changes:
                x, y, z = changes[key]['new_coordinates']
                line = line[:30] + f'{x:8.3f}{y:8.3f}{z:8.3f}' + line[54:]
        lines.append(line)
    input_path = output / 'oxt_repaired.pdb'
    input_path.write_text('\n'.join(lines) + '\n')
    new_atoms = pdb_atoms(input_path)
    drift = max(float(np.linalg.norm(new_atoms[key] - pos)) for key, pos in observed.items())
    if drift > .002:
        raise ValueError('OXT recovery moved original observed coordinates')
    recovery = {'status': 'running', 'original_failed_audit': str(original / 'audit.json'),
                'repair_method': 'Only previously unobserved malformed terminal OXT; carboxyl plane, C-OXT 1.25 A. Protein observed coordinates fixed. Distance >8 A outside box except explicitly reviewed residues recorded per atom; all new OXT checked against other receptor heavy atoms.',
                'oxt_repairs': list(changes.values()), 'max_observed_atom_displacement_A': drift,
                'explicit_distant_template_exclusions': {}, 'missing_loops_built': False,
                'missing_heavy_atoms_repaired': audit.get('missing_heavy_atoms_repaired', []),
                'signature': audit['signature']}
    write_json(output / 'audit.json', recovery)
    command = [sys.executable, '-m', 'meeko.cli.mk_prepare_receptor', '--read_pdb', input_path,
               '-p', output / 'receptor.pdbqt', '--write_pdb', output / 'prepared.pdb',
               '-j', output / 'receptor.json', '--default_altloc', 'A']
    try:
        try:
            logged_run(command, output / 'preparation_log.json')
        except RuntimeError:
            if original.parent.name != 'rat_6uz0':
                raise
            recovery['reviewed_false_inter_residue_bonds'] = type_with_reviewed_false_bonds(
                input_path, output, observed, center, ligand_xyz)
        atoms = [line for line in (output / 'receptor.pdbqt').read_text().splitlines()
                 if line.startswith(('ATOM  ', 'HETATM'))]
        ntypes = Counter(line.split()[-1] for line in atoms if line[12:16].strip() == 'N')
        if not atoms or not ntypes or ntypes.get('NA', 0):
            raise ValueError(f'Invalid recovered protein typing: {ntypes}')
        final_atoms = pdb_atoms(output / 'prepared.pdb')
        if not set(observed).issubset(final_atoms):
            raise ValueError('Meeko recovered receptor lost observed atoms')
        drift = max(float(np.linalg.norm(final_atoms[key] - pos)) for key, pos in observed.items())
        if drift > .002:
            raise ValueError('Meeko recovered receptor moved observed atoms')
        recovery.update(status='ok', backbone_nitrogen_types=dict(ntypes),
                        max_observed_atom_displacement_A=drift,
                        files={name: sha256(output / name) for name in
                               ('oxt_repaired.pdb', 'receptor.pdbqt', 'prepared.pdb', 'receptor.json')})
        write_json(output / 'box.json', {'center': list(center), 'size': [24.] * 3})
    except Exception as exc:
        recovery.update(status='failed', error=str(exc))
        raise
    finally:
        write_json(output / 'audit.json', recovery)
    return output / 'receptor.pdbqt'


def run(directory, pdb_id='6UZ0', ligand_id='K4D'):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    sources = directory / 'sources'
    sources.mkdir(exist_ok=True)
    frozen = code_hashes()
    urls = {f'{pdb_id}.pdb': f'https://files.rcsb.org/download/{pdb_id}.pdb',
            f'{pdb_id}.cif': f'https://files.rcsb.org/download/{pdb_id}.cif',
            'entry.json': f'https://data.rcsb.org/rest/v1/core/entry/{pdb_id}',
            'polymer_entity_1.json': f'https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1',
            f'{ligand_id}.cif': f'https://files.rcsb.org/ligands/download/{ligand_id}.cif',
            f'{ligand_id}_metadata.json': f'https://data.rcsb.org/rest/v1/core/chemcomp/{ligand_id}'}
    records = [download(url, sources / name) for name, url in urls.items()]
    write_json(directory / 'source_manifest.json', records)
    pdb_path = sources / f'{pdb_id}.pdb'
    mol, ligand_audit = ccd_reference(sources / f'{ligand_id}.cif', pdb_path, ligand_id)
    descriptor = read_json(sources / f'{ligand_id}_metadata.json')['rcsb_chem_comp_descriptor']['SMILES_stereo']
    if Chem.MolToSmiles(Chem.MolFromSmiles(descriptor)) != Chem.MolToSmiles(mol):
        raise ValueError('Rebuilt experimental ligand graph/stereo differs from CCD descriptor')
    reference = directory / 'reference.sdf'
    with Chem.SDWriter(str(reference)) as writer:
        writer.SetProps(['_Name'])
        writer.write(mol)
    ligand_audit['smiles'] = Chem.MolToSmiles(mol)
    ligand_audit['formal_charge'] = Chem.GetFormalCharge(mol)
    write_json(directory / 'ligand_audit.json', ligand_audit)
    xyz = mol.GetConformer().GetPositions()
    center = xyz.mean(axis=0).tolist()
    nearby_hetero = hetero_audit(pdb_path, xyz, ligand_id)
    write_json(directory / 'hetero_audit.json', nearby_hetero)
    polymer = read_json(sources / 'polymer_entity_1.json')
    entry = read_json(sources / 'entry.json')
    headers = [line for line in pdb_path.read_text().splitlines()
               if line.startswith(('COMPND', 'SOURCE', 'SEQADV', 'DBREF', 'REMARK   2'))]
    deleted = sorted(int(line.split()[-2]) for line in headers
                     if line.startswith('SEQADV') and 'DELETION' in line)
    deletion_ranges = [[values[0], values[-1]]
                       for _, group in groupby(enumerate(deleted), lambda item: item[1] - item[0])
                       for values in [[value for _, value in group]]]
    construct = {'description': polymer['rcsb_polymer_entity']['pdbx_description'],
                 'entity_length': polymer['entity_poly']['rcsb_sample_sequence_length'],
                 'source_segments': [{k: x.get(k) for k in ('pdbx_beg_seq_num', 'pdbx_end_seq_num', 'pdbx_gene_src_scientific_name', 'pdbx_gene_src_gene', 'pdbx_host_org_cell_line')} for x in polymer.get('entity_src_gen', [])],
                 'entity_poly_metadata': {k: polymer['entity_poly'].get(k) for k in ('pdbx_strand_id', 'rcsb_mutation_count', 'rcsb_conflict_count', 'rcsb_deletion_count')},
                 'uniprot_alignments': polymer.get('rcsb_polymer_entity_align', []),
                 'original_construct_headers': headers,
                 'deletions_in_native_rat_numbering_from_SEQADV': deletion_ranges,
                 'deleted_residues_count_from_SEQADV': len(deleted),
                 'native_segment_dbref': [line.strip() for line in headers if line.startswith('DBREF') and 'P15389' in line],
                 'fusion_segment_dbref': [line.strip() for line in headers if line.startswith('DBREF') and 'P15389' not in line],
                 'caveat': 'Engineered rat Nav1.5 construct with GFP/tag sequence. RCSB mutation/deletion counters do not prove full-length native protein; source headers retained for review.'}
    write_json(directory / 'construct_audit.json', construct)
    print(json.dumps({'ligand': ligand_audit, 'closest_hetero': nearby_hetero[:4], 'center': center}), flush=True)
    original_audit = directory / 'receptor' / 'audit.json'
    if original_audit.exists() and read_json(original_audit).get('status') == 'failed':
        receptor = recover_added_remote_oxt(directory / 'receptor', np.asarray(center), reference,
                                           reviewed_residues=('A:945',) if pdb_id == '6UZ0' else ())
    else:
        try:
            receptor = prepare_receptor(pdb_path, directory / 'receptor', 'A', center, size=24, resume=True)
        except RuntimeError:
            receptor = recover_added_remote_oxt(directory / 'receptor', np.asarray(center), reference,
                                               reviewed_residues=('A:945',) if pdb_id == '6UZ0' else ())
    audit = read_json(receptor.parent / 'audit.json')
    manifest = {'reference_sdf': str(reference), 'receptor_pdbqt': str(receptor),
                'prepared_pdb': str(receptor.parent / 'prepared.pdb'), 'center': center,
                'smiles': Chem.MolToSmiles(mol), 'state': 'neutral_CCD', 'formalcharge': Chem.GetFormalCharge(mol),
                'canonical_smiles': Chem.MolToSmiles(mol), 'chemistry_usable': True,
                'formal_charge': Chem.GetFormalCharge(mol), 'species': 'Rattus norvegicus', 'chain': 'A',
                'construct': construct, 'metadata': {'pdb_id': pdb_id, 'ligand_id': ligand_id, 'citation': entry.get('rcsb_primary_citation'),
                'resolution_A': entry['rcsb_entry_info'].get('resolution_combined'), 'source_records': records,
                'scope': 'Rat structural/mechanistic reference; not human Nav1.5 validation or selectivity evidence',
                'receptor_environment': 'protein-only; resolved hetero atoms audited separately, not automatically restored'},
                'ligand_audit': ligand_audit, 'hetero_audit': nearby_hetero,
                'receptor_audit': audit, 'production_code_hashes_before': frozen,
                'production_code_hashes_after': code_hashes(), 'script_sha256': sha256(__file__),
                'artifacts_sha256': {'reference_sdf': sha256(reference), 'receptor_pdbqt': sha256(receptor)}}
    if frozen != code_hashes():
        raise RuntimeError('Production code changed during preparation')
    write_json(directory / 'prepare_manifest.json', manifest)
    print(f'Prepared {directory / "prepare_manifest.json"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/nav15_multireference_20261002/rat_6uz0')
    args = parser.parse_args()
    run(args.output)
