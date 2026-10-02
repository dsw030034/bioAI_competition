"""Audit 9V3S from deposited coordinates without silently correcting its ligand.

The CCD hydrazine graph conflicts with the paper's azo ligand. This script keeps
the CCD molecule and an explicit N=N diagnostic graph separate, and excludes
both from independent redocking validation until the discrepancy is resolved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import urllib.request
import xml.etree.ElementTree as ET

import gemmi
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scgd.geometry import pdb_atoms, pose_contacts
from scgd.receptors import prepare_receptor
from scgd.runtime import BASE, code_hashes, read_json, sha256, versions, write_json


URLS = {
    '9V3S.cif': 'https://files.rcsb.org/download/9V3S.cif',
    'A1EQU.cif': 'https://files.rcsb.org/ligands/download/A1EQU.cif',
    'entry.json': 'https://data.rcsb.org/rest/v1/core/entry/9V3S',
    'polymer_1.json': 'https://data.rcsb.org/rest/v1/core/polymer_entity/9V3S/1',
    'ligand_metadata.json': 'https://data.rcsb.org/rest/v1/core/chemcomp/A1EQU',
    'paper.xml': 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13103313/fullTextXML',
}


def category_rows(block, name):
    cat = block.get_mmcif_category(name)
    return [dict(zip(cat, values)) for values in zip(*cat.values())]


def build_observed_molecule(ccd, observed, azo=False):
    """Atom-name graph and measured coordinates only; no ideal-coordinate use."""
    atoms = [a for a in category_rows(ccd, '_chem_comp_atom.') if a['type_symbol'] != 'H']
    bonds = category_rows(ccd, '_chem_comp_bond.')
    names = {a['atom_id']: i for i, a in enumerate(atoms)}
    if set(names) != set(observed):
        raise ValueError({'missing_deposited_atoms': sorted(set(names) - set(observed)),
                          'extra_deposited_atoms': sorted(set(observed) - set(names))})
    builder = Chem.RWMol()
    for atom in atoms:
        a = Chem.Atom(atom['type_symbol'])
        a.SetFormalCharge(int(atom['charge']))
        a.SetProp('atom_name', atom['atom_id'])
        builder.AddAtom(a)
    orders = {'SING': Chem.BondType.SINGLE, 'DOUB': Chem.BondType.DOUBLE,
              'TRIP': Chem.BondType.TRIPLE, 'AROM': Chem.BondType.AROMATIC}
    for bond in bonds:
        a, b = bond['atom_id_1'], bond['atom_id_2']
        if a not in names or b not in names:
            continue
        order = orders[bond['value_order']]
        if azo and {a, b} == {'N22', 'N23'}:
            order = Chem.BondType.DOUBLE
        builder.AddBond(names[a], names[b], order)
    mol = builder.GetMol()
    Chem.SanitizeMol(mol)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for name, index in names.items():
        conformer.SetAtomPosition(index, observed[name])
    conformer.Set3D(True)
    mol.AddConformer(conformer)
    Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    checks = []
    for a in atoms:
        atom = mol.GetAtomWithIdx(names[a['atom_id']])
        expected = a.get('pdbx_stereo_config', 'N')
        if expected in ('R', 'S'):
            measured = atom.GetProp('_CIPCode') if atom.HasProp('_CIPCode') else None
            checks.append({'atom_name': a['atom_id'], 'element': a['type_symbol'],
                           'ccd_stereo': expected, 'deposited_3D_stereo': measured,
                           'match': expected == measured})
    bond = mol.GetBondBetweenAtoms(names['N22'], names['N23'])
    audit = {'smiles': Chem.MolToSmiles(mol, isomericSmiles=True),
             'formal_charge': Chem.GetFormalCharge(mol), 'heavy_atoms': mol.GetNumAtoms(),
             'N22_N23_order': str(bond.GetBondType()), 'N22_N23_stereo': str(bond.GetStereo()),
             'N22_N23_length_A': float(np.linalg.norm(np.array(observed['N22']) - observed['N23'])),
             'C20_N22_N23_C24_dihedral_degrees': rdMolTransforms.GetDihedralDeg(
                 conformer, names['C20'], names['N22'], names['N23'], names['C24']),
             'stereocenter_checks': checks, 'all_ccd_stereocenters_match_deposited_3D': all(x['match'] for x in checks),
             'all_carbon_stereocenters_match_deposited_3D': all(x['match'] for x in checks if x['element'] == 'C'),
             'nitrogen_stereo_note': 'CCD labels bridgehead N31 S; RDKit cleanIt removes the neutral N stereo tag. This is a stereoperception limitation, not evidence for a carbon stereochemistry mismatch.',
             'coordinates': 'all 38 deposited heavy coordinates copied exactly by CCD atom names; no embedding/minimization',
             'atom_order': list(names)}
    return mol, audit


def run(output, resume=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = output / 'source'
    source.mkdir(exist_ok=True)
    frozen = code_hashes()
    sources = []
    for name, url in URLS.items():
        path = source / name
        if not path.exists():
            path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        sources.append({'path': str(path), 'url': url, 'sha256': sha256(path)})
    supplement_attempts = [
        {'url': 'https://static-content.springer-cdn.com/esm/art%3A10.1038%2Fs41467-026-70305-6/MediaObjects/41467_2026_70305_MOESM1_ESM.pdf',
         'outcome': 'TLS SSLEOFError in urllib; PDF not retrieved'},
        {'url': 'https://pmc.ncbi.nlm.nih.gov/articles/instance/13103313/bin/41467_2026_70305_MOESM1_ESM.pdf',
         'outcome': 'HTTP 200 text/html challenge, not PDF; no supplementary chemical drawing verified'},
    ]
    block = gemmi.cif.read_file(str(source / '9V3S.cif')).sole_block()
    ccd = gemmi.cif.read_file(str(source / 'A1EQU.cif')).sole_block()
    atomrows = category_rows(block, '_atom_site.')
    ligand_rows = [x for x in atomrows if x['label_comp_id'] == 'A1EQU']
    assert len(ligand_rows) == 38 and {x['auth_asym_id'] for x in ligand_rows} == {'A'}
    assert len({x['auth_seq_id'] for x in ligand_rows}) == 1
    assert all(float(x['occupancy']) == 1. for x in ligand_rows)
    observed = {x['label_atom_id']: [float(x[f'Cartn_{a}']) for a in 'xyz'] for x in ligand_rows}
    center = np.array(list(observed.values())).mean(axis=0)
    ccd_mol, ccd_audit = build_observed_molecule(ccd, observed)
    azo_mol, azo_audit = build_observed_molecule(ccd, observed, azo=True)
    for stem, mol, audit in [('reference_ccd_unvalidated', ccd_mol, ccd_audit),
                              ('reference_azo_hypothesis_unvalidated', azo_mol, azo_audit)]:
        path = output / f'{stem}.sdf'
        mol.SetProp('_Name', stem)
        mol.SetProp('validation_status', 'excluded: CCD/paper/deposited-geometry inconsistency')
        with Chem.SDWriter(str(path)) as writer:
            writer.write(mol)
        audit['sdf_path'] = str(path)
        audit['sdf_sha256'] = sha256(path)

    paper = ET.fromstring((source / 'paper.xml').read_bytes())
    excerpts = [' '.join(x.itertext()) for x in paper.iter('p') if any(
        z in ''.join(x.itertext()).lower() for z in ['azobenzene moiety incorporating',
        'to assess azo-q2a binding', 'unambiguous em densities', 'carboxyl terminal domain (1895'])]
    write_json(output / 'chemical_audit.json', {'ccd': ccd_audit, 'azo_bond_hypothesis': azo_audit,
        'paper_excerpts': excerpts,
        'conclusion': 'CCD specifies N22-N23 SINGLE with two hydrazine hydrogens; paper identifies photoswitchable cis-azobenzene. Changing bond order alone is a hypothesis, not a validated correction. Neither molecule is approved for reference docking.'})

    # Five-character ligand IDs cannot be represented in traditional PDB. Export
    # only protein ATOM rows; all ligand records stay in their original mmCIF.
    structure = gemmi.make_structure_from_block(block)
    protein = gemmi.Structure()
    model = gemmi.Model('1')
    chain = gemmi.Chain('A')
    for residue in structure[0]['A']:
        if residue.het_flag == 'A':
            chain.add_residue(residue)
    model.add_chain(chain)
    protein.add_model(model)
    protein.setup_entities()
    pdb = output / 'protein_observed.pdb'
    protein.write_pdb(str(pdb))
    converted = pdb_atoms(pdb)
    originals = {(x['auth_asym_id'], x['auth_seq_id'], x['auth_comp_id'], x['auth_atom_id']):
                 np.array([float(x[f'Cartn_{a}']) for a in 'xyz']) for x in atomrows if x['group_PDB'] == 'ATOM'}
    # pdb_atoms keys are chain, resseq, residue, atom. All this structure's
    # observed protein altlocs and insertion codes are absent.
    if set(originals) != set(converted):
        raise ValueError('mmCIF->protein PDB changed observed atom identities')
    conversion_drift = max(float(np.linalg.norm(originals[k] - converted[k])) for k in originals)
    if conversion_drift > 0.001:
        raise ValueError('mmCIF->PDB shifted observed coordinates')
    hetero = {}
    for x in atomrows:
        if x['group_PDB'] != 'HETATM' or x['label_comp_id'] == 'A1EQU':
            continue
        key = (x['auth_asym_id'], x['auth_seq_id'], x['label_comp_id'])
        hetero.setdefault(key, []).append([float(x[f'Cartn_{a}']) for a in 'xyz'])
    environment = [{'chain': k[0], 'residue': k[1], 'component': k[2], 'observed_atoms': len(coords),
                    'minimum_distance_to_ligand_A': float(np.linalg.norm(np.array(coords)[:, None] - np.array(list(observed.values())), axis=2).min())}
                   for k, coords in hetero.items()]
    write_json(output / 'environment_audit.json', {'hetero_components': environment,
        'protein_only_preparation': True, 'resolved_water_atoms': sum(x['label_comp_id'] == 'HOH' for x in atomrows),
        'observed_protein_heavy_atoms': len(originals), 'mmcif_to_pdb_max_displacement_A': conversion_drift})
    manifest = {'reference_id': 'human_9v3s', 'pdb_id': '9V3S', 'chain': 'A', 'species': 'Homo sapiens',
                'source_path': str(source / '9V3S.cif'), 'protein_observed_pdb': str(pdb),
                'reference_sdf': None, 'ligand_smiles': None,
                'reference_ccd_unvalidated_sdf': ccd_audit['sdf_path'],
                'reference_azo_hypothesis_unvalidated_sdf': azo_audit['sdf_path'],
                'center': center.tolist(), 'box_size_A': 24., 'ligand_component': 'A1EQU',
                'build': 'Human Nav1.5S, N-terminal twin-Strep/FLAG; deletions 461-657, 1066-1187, 1895-2016 (paper author numbering); not a full-length wild-type channel.',
                'resolution_A': 3.0, 'paper_doi': '10.1038/s41467-026-70305-6',
                'status': 'ligand_chemistry_conflict_excluded_from_redocking',
                'eligible_for_reference_redocking': False,
                'ligand_chemistry_audit': str(output / 'chemical_audit.json'),
                'environment_audit': str(output / 'environment_audit.json'),
                'sources': sources, 'production_code_hashes_before': frozen, 'versions': versions()}
    manifest['supplementary_access_attempts'] = supplement_attempts
    manifest['pdb_download_note'] = 'RCSB marks pdb_format_compatible=N because of 5-character ligand ID; original mmCIF retained, protein-only PDB converted using author atom/residue identifiers.'
    write_json(output / 'prepare_manifest.json', manifest)
    print(json.dumps({'ligand_chemical_audit': {'ccd': ccd_audit, 'azo_hypothesis': azo_audit}, 'center': center.tolist()}, indent=2), flush=True)
    receptor_dir = output / 'receptor'
    try:
        receptor = prepare_receptor(pdb, receptor_dir, 'A', center, 24., resume=resume)
        prepared_atoms = pdb_atoms(receptor_dir / 'prepared.pdb')
        if not set(originals).issubset(prepared_atoms):
            raise ValueError('Meeko preparation removed experimental heavy atoms')
        # Meeko can swap terminal carboxyl O/OXT names when matching a residue
        # template. Require the exact deposited coordinate under that explicit
        # chemically equivalent name; arbitrary nearest-neighbour matching is
        # forbidden. Keep the supplied names unchanged and expose every swap.
        atom_mapping, renamings = {}, []
        for key, xyz in originals.items():
            target = key
            if np.linalg.norm(xyz - prepared_atoms[key]) > 0.002 and key[-1] == 'O':
                alternative = (*key[:3], 'OXT')
                if alternative in prepared_atoms and np.linalg.norm(xyz - prepared_atoms[alternative]) <= 0.002:
                    target = alternative
                    renamings.append({'observed_atom': list(key), 'prepared_atom': list(alternative),
                        'reason': 'Meeko terminal carboxyl O/OXT template name permutation; experimental coordinate unchanged'})
            atom_mapping[key] = target
        final_drift = max(float(np.linalg.norm(originals[k] - prepared_atoms[target])) for k, target in atom_mapping.items())
        if final_drift > 0.002:
            raise ValueError('Meeko preparation shifted experimental heavy atoms')
        mapped_observed = set(atom_mapping.values())
        added = [{'atom_key': list(k), 'distance_to_ligand_A': float(np.linalg.norm(
                    np.array(list(observed.values())) - p, axis=1).min())}
                 for k, p in prepared_atoms.items() if k not in mapped_observed]
        gap_rows = []
        residues = list(chain)
        for left, right in zip(residues, residues[1:]):
            if right.seqid.num <= left.seqid.num + 1:
                continue
            coords = np.array([[a.pos.x, a.pos.y, a.pos.z] for r in (left, right) for a in r])
            gap_rows.append({'left': left.seqid.num, 'right': right.seqid.num,
                             'endpoint_min_distance_to_ligand_A': float(np.linalg.norm(
                                 coords[:, None] - np.array(list(observed.values())), axis=2).min())})
        write_json(output / 'final_geometry_audit.json', {
            'observed_heavy_atoms_retained': len(originals), 'max_final_heavy_displacement_A': final_drift,
            'explicit_terminal_O_OXT_name_permutations': renamings,
            'added_heavy_atoms': added, 'observed_numbering_gap_endpoints': gap_rows,
            'note': 'No missing loops built. Numbering gaps include engineered deletion regions; these distances are endpoint distances, not guarantees about possible missing-loop paths.'})
        manifest.update(receptor_pdbqt=str(receptor), prepared_pdb=str(receptor_dir / 'prepared.pdb'),
                        receptor_status='ok', receptor_sha256=sha256(receptor),
                        reference_geometry=pose_contacts(ccd_mol, receptor_dir / 'prepared.pdb'),
                        final_geometry_audit=str(output / 'final_geometry_audit.json'))
    except Exception as exc:
        manifest.update(receptor_status='failed', receptor_error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest['production_code_hashes_after'] = code_hashes()
        manifest['production_code_unchanged'] = frozen == manifest['production_code_hashes_after']
        manifest['script_sha256'] = sha256(__file__)
        write_json(output / 'prepare_manifest.json', manifest)
    print(json.dumps({'manifest': str(output / 'prepare_manifest.json'), 'receptor_status': manifest['receptor_status'],
                      'reference_eligible': False}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=BASE / 'outputs/nav15_multireference_20261002/human_9v3s')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.output, args.resume)
