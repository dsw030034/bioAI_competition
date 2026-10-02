"""Prepare explicit +1 sensitivity states and independent ligand conformers.

No new experimental references, pKa predictions, microscopic population claims,
production score changes, or doubly protonated states are introduced.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from rdkit import Chem

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scgd.runtime import code_hashes, read_json, sha256, versions, write_json
from validate_nav15_multireference import sample, freeze_artifacts


STATE_SPECS = (
    ('flecainide_N19_plus1', 'rat_6uz0', 'N19', 'secondary piperidine nitrogen'),
    ('ranolazine_N13_plus1', 'rat_8f6p', 'N13', 'tertiary piperazine nitrogen carrying the acetamide-side substituent'),
    ('ranolazine_N16_plus1', 'rat_8f6p', 'N16', 'tertiary piperazine nitrogen carrying the hydroxypropoxy-side substituent'),
)


def is_amide_nitrogen(atom):
    if atom.GetAtomicNum() != 7:
        return False
    return any(neighbor.GetAtomicNum() == 6 and any(
        bond.GetBondType() == Chem.BondType.DOUBLE and
        bond.GetOtherAtom(neighbor).GetAtomicNum() in (8, 16)
        for bond in neighbor.GetBonds()) for neighbor in atom.GetNeighbors())


def protonate_one_nitrogen(source, index):
    """Preserve deposited heavy coordinates/graph/stereo, add exactly H+ to N."""
    original = Chem.RemoveHs(Chem.Mol(source))
    mol = Chem.Mol(original)
    atom = mol.GetAtomWithIdx(index)
    if atom.GetAtomicNum() != 7 or atom.GetFormalCharge() != 0 or is_amide_nitrogen(atom):
        raise ValueError('Require an uncharged non-amide amine nitrogen')
    if atom.GetIsAromatic() or any(b.GetBondType() != Chem.BondType.SINGLE for b in atom.GetBonds()):
        raise ValueError('Expected a saturated amine site')
    before_h = int(atom.GetTotalNumHs())
    atom.SetFormalCharge(1)
    atom.SetNumExplicitHs(before_h + 1)
    atom.SetNoImplicit(True)
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    if Chem.GetFormalCharge(mol) != Chem.GetFormalCharge(original) + 1:
        raise ValueError('Protonation charge difference is not +1')
    if not np.array_equal(mol.GetConformer().GetPositions(), original.GetConformer().GetPositions()):
        raise ValueError('Protonation moved heavy atoms')
    stereo = []
    for old, new in zip(original.GetAtoms(), mol.GetAtoms()):
        if old.GetAtomicNum() == 6 and old.HasProp('_CIPCode'):
            cip = new.GetProp('_CIPCode') if new.HasProp('_CIPCode') else None
            if cip != old.GetProp('_CIPCode'):
                raise ValueError('Carbon stereochemical descriptor changed')
            stereo.append({'index': old.GetIdx(), 'source_CIP': old.GetProp('_CIPCode'), 'state_CIP': cip})
        if old.GetIdx() != index and old.GetFormalCharge() != new.GetFormalCharge():
            raise ValueError('Unselected atom charge changed')
    # The piperidine ammonium has two H, and the two piperazine ring paths are
    # equivalent: none of these selected nitrogens constitutes a new stable
    # stereocenter. Do not introduce an arbitrary N @/@@ tag.
    nitrogen_stereo = [dict(index=a.GetIdx(), chiral_tag=str(a.GetChiralTag()),
                          CIP=a.GetProp('_CIPCode') if a.HasProp('_CIPCode') else None)
                      for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if any(x['chiral_tag'] != 'CHI_UNSPECIFIED' or x['CIP'] is not None for x in nitrogen_stereo):
        raise ValueError('Unexpected nitrogen stereochemistry needs manual audit')
    return mol, {'protonated_index': index, 'source_nitrogen_H_count': before_h,
                 'state_nitrogen_H_count': int(atom.GetTotalNumHs()),
                 'formal_charge': Chem.GetFormalCharge(mol), 'carbon_stereo': stereo,
                 'nitrogen_stereo': nitrogen_stereo, 'max_heavy_coordinate_displacement_A': 0.,
                 'heavy_atom_count': mol.GetNumAtoms(), 'heavy_bond_count': mol.GetNumBonds()}


def sources_audit(source_dir):
    specs = [
        ('flecainide_label.xml', 'https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/ee9595d5-08a1-4529-9215-ddc3c7f929c4.xml'),
        ('ranolazine_label.xml', 'https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/d74f3dea-7357-4e9a-be9a-2f498a44108a.xml'),
        ('PMC9318806.xml', 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9318806/fullTextXML'),
        ('PMC7845660.xml', 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7845660/fullTextXML'),
        ('PMC8410097.xml', 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8410097/fullTextXML'),
    ]
    provenance = []
    for name, url in specs:
        path = source_dir / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing reviewed literature source: {path}')
        root = ET.fromstring(path.read_bytes())
        paragraphs = [' '.join(p.itertext()) for p in root.iter('p')]
        excerpts = [p for p in paragraphs if 'pKa' in p or 'pK a ' in p]
        provenance.append({'path': str(path), 'url': url, 'sha256': sha256(path), 'pKa_excerpts': excerpts})
    return {'sources': provenance,
        'flecainide': 'PMC9318806 explicitly describes piperidine protonation and pKa 9.3; PMC7845660 also reports pKa 9.3. +1 piperidine is a justified sensitivity state.',
        'ranolazine': 'PMC8410097 reports pKa 7.2, citing Peters et al. 2013 DOI 10.1111/bph.12150. Both neutral and singly protonated forms are relevant for sensitivity; this macroscopic pKa does not assign N13 vs N16 microstate fractions.',
        'limitations': 'Downloaded DailyMed labels did not provide pKa in searched text. Original Peters paper XML request returned HTTP500; the quoted value is verified in the secondary mechanistic paper, not independently remeasured or microstate-resolved. No numerical site populations assigned, no +2 states enumerated.'}


def prepare_state(output, baseline, spec, evidence_path):
    state, parent_name, atom_name, site_role = spec
    folder = output / state
    if folder.exists():
        raise FileExistsError(f'State folder exists; preserve frozen artifacts and use a fresh output: {folder}')
    folder.mkdir(parents=True)
    parent = baseline / parent_name
    parent_manifest = read_json(parent / 'prepare_manifest.json')
    if parent_manifest.get('chemistry_usable') is not True:
        raise ValueError('Parent graph not approved')
    original_ref = Path(parent_manifest['reference_sdf'])
    source = Chem.SDMolSupplier(str(original_ref), removeHs=True)[0]
    names = read_json(parent / 'ligand_audit.json')['atom_names']
    if len(names) != source.GetNumAtoms():
        raise ValueError('CCD atom-name index length mismatch')
    index = names.index(atom_name)
    mol, chemistry = protonate_one_nitrogen(source, index)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    audit_nitrogens = [{'index': a.GetIdx(), 'ccd_atom_name': names[a.GetIdx()],
                       'amide_nitrogen': is_amide_nitrogen(a),
                       'neighbors': [names[n.GetIdx()] for n in a.GetNeighbors()],
                       'formal_charge': a.GetFormalCharge()} for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    chemistry.update(state=state, parent_reference=parent_name, protonated_ccd_atom_name=atom_name,
                     site_role=site_role, nitrogen_sites=audit_nitrogens, atom_names=names,
                     smiles=smiles, stereo_policy='retain deposited carbon R; no arbitrary nitrogen stereotag',
                     source_reference_sha256=sha256(original_ref),
                     literature_evidence=str(evidence_path), literature_evidence_sha256=sha256(evidence_path),
                     experimental_status='Protonation is modeled, not measured in the deposited experimental ligand; sensitivity control only.')
    mol.SetProp('_Name', state)
    mol.SetProp('role', 'protonation_sensitivity_control')
    mol.SetProp('protonated_CCD_atom_name', atom_name)
    mol.SetProp('experimental_state_confirmed', 'false')
    ref_path = folder / 'reference.sdf'
    with Chem.SDWriter(str(ref_path)) as writer:
        writer.write(mol)
    roundtrip = Chem.SDMolSupplier(str(ref_path), removeHs=True)[0]
    if Chem.MolToSmiles(roundtrip, isomericSmiles=True) != smiles:
        raise ValueError('Protonated SDF roundtrip graph or stereo differs')
    max_drift = float(np.abs(roundtrip.GetConformer().GetPositions() - source.GetConformer().GetPositions()).max())
    if max_drift > 1e-6:
        raise ValueError('SDF roundtrip changed source heavy coordinates')
    chemistry['sdf_roundtrip_max_coordinate_displacement_A'] = max_drift
    chemistry['reference_sha256'] = sha256(ref_path)
    write_json(folder / 'chemical_audit.json', chemistry)
    manifest = {'state_id': state, 'reference_sdf': str(ref_path),
        'receptor_pdbqt': parent_manifest['receptor_pdbqt'], 'prepared_pdb': parent_manifest['prepared_pdb'],
        'center': parent_manifest['center'], 'smiles': smiles, 'formal_charge': 1,
        'chemistry_usable': True, 'eligible_for_reference_redocking': True,
        'role': 'protonation_sensitivity_control', 'independent_experimental_reference': False,
        'parent_reference': parent_name, 'parent_reference_sdf': str(original_ref),
        'parent_prepare_manifest': str(parent / 'prepare_manifest.json'),
        'parent_prepare_manifest_sha256': sha256(parent / 'prepare_manifest.json'),
        'chemical_audit': str(folder / 'chemical_audit.json'),
        'chemical_audit_sha256': sha256(folder / 'chemical_audit.json'),
        'protonated_ccd_atom_name': atom_name, 'species': parent_manifest['species'],
        'graph_provenance': 'CCD-validated parent; exactly one added proton on specified non-amide amine; deposited heavy xyz and carbon stereo preserved',
        'sampling': {'algorithm': 'ETKDGv3/MMFF94s, graph_only removes native coordinates',
                     'seeds': [202610031, 202610032], 'attempted': 128,
                     'dedup_RMSD_A': .75, 'same_state_energy_window_kcal_mol': 10., 'max_selected': 5},
        'source_files': {str(p): sha256(p) for p in (original_ref, parent / 'prepare_manifest.json',
            parent / 'ligand_audit.json', Path(parent_manifest['receptor_pdbqt']), Path(parent_manifest['prepared_pdb']))},
        'literature_evidence': str(evidence_path), 'script_sha256': sha256(__file__),
        'sample_helper_sha256': sha256(Path(__file__).with_name('validate_nav15_multireference.py')),
        'versions': versions(), 'production_code_hashes': code_hashes(), 'production_gate_released': False,
        'limitations': 'Sensitivity input state only; unknown bound protonation and no microscopic populations. Rat truncated channel, protein-only parent receptor. Passing here does not create a new independent experimental validation.'}
    write_json(folder / 'prepare_manifest.json', manifest)
    print(f'Prepared {state}: {smiles}', flush=True)
    return folder


def run(output, baseline, do_sampling=True):
    output, baseline = Path(output).resolve(), Path(baseline).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen = code_hashes()
    evidence = sources_audit(output / 'sources')
    evidence_path = output / 'literature_evidence.json'
    write_json(evidence_path, evidence)
    folders = [prepare_state(output, baseline, spec, evidence_path) for spec in STATE_SPECS]
    summary = []
    for folder in folders:
        item = {'state': folder.name, 'manifest': str(folder / 'prepare_manifest.json'),
                'manifest_sha256': sha256(folder / 'prepare_manifest.json')}
        if do_sampling:
            report = sample(folder / 'reference.sdf', folder / 'sampling')
            write_json(folder / 'sampling_integrity.json', freeze_artifacts(folder / 'sampling'))
            item['sampling_counts'] = report['counts']
            item['sampling_report'] = str(folder / 'sampling/report.json')
        summary.append(item)
        write_json(output / 'state_summary.json', {'states': summary, 'production_code_unchanged': frozen == code_hashes()})
    if code_hashes() != frozen:
        raise ValueError('Production code changed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE / 'outputs/nav15_environment_states_20261002/states')
    parser.add_argument('--baseline', type=Path, default=BASE / 'outputs/nav15_multireference_20261002')
    parser.add_argument('--no-sampling', action='store_true')
    args = parser.parse_args()
    run(args.output, args.baseline, not args.no_sampling)
