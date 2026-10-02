"""Explicitly audited distal-residue recovery for the protein-only 8F6P panel."""
from collections import Counter
from pathlib import Path
import re
import sys
import numpy as np
from rdkit import Chem
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scgd.geometry import pdb_atoms, select_chain
from scgd.runtime import code_hashes, logged_run, read_json, sha256, write_json


def fix_new_oxt(original, output, center, ligand_xyz):
    observed=pdb_atoms(original/'selected.pdb'); atoms=pdb_atoms(original/'repaired.pdb'); fixes={}
    for key,pos in atoms.items():
        if key in observed or key[-1]!='OXT': continue
        carbon,oxygen,alpha=(atoms[key[:-1]+(name,)] for name in ('C','O','CA'))
        if np.linalg.norm(pos-oxygen)>=1.8: continue
        direction=-((oxygen-carbon)/np.linalg.norm(oxygen-carbon)+(alpha-carbon)/np.linalg.norm(alpha-carbon))
        new=carbon+1.25*direction/np.linalg.norm(direction)
        others=np.asarray([xyz for k,xyz in atoms.items() if k not in (key,key[:-1]+('C',))])
        if np.linalg.norm(others-new,axis=1).min()<1.8: raise ValueError(f'New OXT clash {key}')
        fixes[key]={'key':list(key),'old_coordinates':pos.tolist(),'new_coordinates':new.tolist(),
                    'old_O_OXT_A':float(np.linalg.norm(pos-oxygen)),'new_O_OXT_A':float(np.linalg.norm(new-oxygen)),
                    'old_minimum_ligand_distance_A':float(np.linalg.norm(ligand_xyz-pos,axis=1).min()),
                    'new_minimum_ligand_distance_A':float(np.linalg.norm(ligand_xyz-new,axis=1).min()),
                    'reason':'New PDBFixer terminal oxygen overlapped observed O; explicitly restore carboxyl plane. Original observed coordinates fixed.'}
    lines=[]
    for line in (original/'repaired.pdb').read_text().splitlines():
        if line.startswith('ATOM'):
            key=(line[21],line[22:27].strip(),line[17:20],line[12:16].strip())
            if key in fixes:
                x,y,z=fixes[key]['new_coordinates']; line=line[:30]+f'{x:8.3f}{y:8.3f}{z:8.3f}'+line[54:]
        lines.append(line)
    dest=output/'oxt_repaired.pdb'; dest.write_text('\n'.join(lines)+'\n')
    write_json(output/'oxt_repairs.json',list(fixes.values()))
    return dest


def restore_terminal_labels(output,observed):
    from meeko import Polymer,PDBQTWriterLegacy
    polymer=Polymer.from_json((output/'receptor.json').read_text())
    fixes=[]
    for residue_id,monomer in polymer.monomers.items():
        if not monomer.atom_names or 'OXT' not in monomer.atom_names: continue
        coords=monomer.rdkit_mol.GetConformer().GetPositions(); i=monomer.atom_names.index('O'); j=monomer.atom_names.index('OXT')
        key=(*residue_id.split(':'),monomer.input_resname,'O')
        if key not in observed or np.linalg.norm(coords[i]-observed[key])<.002: continue
        if np.linalg.norm(coords[j]-observed[key])>.002: raise ValueError(f'O mismatch is not symmetry swap: {key}')
        monomer.atom_names=list(monomer.atom_names); monomer.atom_names[i],monomer.atom_names[j]='OXT','O'
        monomer._set_pdbinfo(residue_id); fixes.append(residue_id)
    if fixes:
        rigid,flex=PDBQTWriterLegacy.write_from_polymer(polymer)
        if flex: raise ValueError('Unexpected flexibility')
        (output/'receptor.pdbqt').write_text(rigid); (output/'prepared.pdb').write_text(polymer.to_pdb()); (output/'receptor.json').write_text(polymer.to_json())
    write_json(output/'terminal_oxygen_name_restorations.json',{'residues':fixes,'policy':'Restore observed O/OXT names only; all coordinates, charges and atom types unchanged'})


def run():
    base = Path('outputs/nav15_multireference_20261002/rat_8f6p').resolve()
    original, output = base / 'receptor', base / 'receptor_distal_filtered'
    output.mkdir(exist_ok=True)
    frozen = code_hashes()
    mol = Chem.SDMolSupplier(str(base / 'reference.sdf'))[0]
    ligand_xyz = mol.GetConformer().GetPositions()
    center = ligand_xyz.mean(axis=0)
    atoms = pdb_atoms(original / 'repaired.pdb')
    observed = pdb_atoms(original / 'selected.pdb')
    repair_source=fix_new_oxt(original,output,center,ligand_xyz)
    signature = read_json(original / 'audit.json')['signature']
    messages = ['No template matched for residue_key=\'A:300\'\nNo template matched for residue_key=\'A:719\'\nNo template matched for residue_key=\'A:336\'']
    excluded = {}
    for attempt in range(12):
        bad = set()
        for message in messages:
            bad.update(re.findall(r"No template matched for residue_key='([^']+)'", message))
            bad.update(re.findall(r'corresponding to key ([A-Za-z0-9:]+)', message))
        new_bad = bad - set(excluded)
        if not new_bad:
            raise ValueError('No new identifiable residue error for recovery')
        for key in sorted(new_bad):
            xyz = np.asarray([pos for atom_key, pos in atoms.items() if ':'.join(atom_key[:2]) == key])
            dist = float(np.linalg.norm(np.maximum(np.abs(xyz-center)-12,0),axis=1).min())
            if dist <= 8:
                raise ValueError(f'Residue too close to search box for exclusion: {key}, {dist}')
            excluded[key] = {'minimum_distance_to_box_A': dist,
                             'minimum_ligand_distance_A': float(np.linalg.norm(xyz[:, None]-ligand_xyz[None,:],axis=2).min()),
                             'heavy_atoms': len(xyz), 'attempt': attempt,
                             'reason': 'Meeko rejects this residue after PDBFixer; every atom lies >8 A outside the search box. Entire residue explicitly excluded in this diagnostic receptor.'}
            if key=='A:336':
                excluded[key]['reason']='Newly built CYS336 SG lies 2.278 A from observed CYS281 SG, causing a spurious extra disulfide. Source SSBOND explicitly pairs 281-327 at 2.027 A. Exclude distant 336; preserve the observed 281-327 disulfide.'
        write_json(output / 'exclusions.json', excluded)
        if len(excluded) > 20:
            raise ValueError('More than 20 residues require deletion; stop for explicit review')
        filtered = output / 'filtered.pdb'
        filtered.write_text('\n'.join(line for line in repair_source.read_text().splitlines()
                              if not line.startswith('ATOM  ') or f'{line[21]}:{line[22:27].strip()}' not in excluded)+'\n')
        selected = output / 'meeko_input.pdb'
        select_chain(filtered, selected, 'A')
        command = [sys.executable, '-m', 'meeko.cli.mk_prepare_receptor', '--read_pdb', selected,
                   '-p', output / 'receptor.pdbqt', '--write_pdb', output / 'prepared.pdb',
                   '-j', output / 'receptor.json', '--default_altloc', 'A']
        logfile = output / f'preparation_log_{attempt}.json'
        try:
            logged_run(command, logfile)
            break
        except RuntimeError:
            messages = [read_json(logfile)['stderr']]
    else:
        raise ValueError('Recovery exceeded 12 explicit rounds')
    restore_terminal_labels(output,observed)
    final = pdb_atoms(output / 'prepared.pdb')
    kept_observed = {key: pos for key,pos in observed.items() if ':'.join(key[:2]) not in excluded}
    if not set(kept_observed).issubset(final):
        raise ValueError('Unplanned observed-atom removal in recovered receptor')
    drift = max(float(np.linalg.norm(final[key]-pos)) for key,pos in kept_observed.items())
    if drift > .002:
        raise ValueError(f'Recovered receptor moved observed coordinates by {drift}')
    added_near = [{'key': list(key), 'minimum_ligand_distance_A': float(np.linalg.norm(ligand_xyz-pos,axis=1).min())}
                  for key,pos in final.items() if key not in observed and np.linalg.norm(ligand_xyz-pos,axis=1).min()<=8]
    ntypes = Counter(line.split()[-1] for line in (output/'receptor.pdbqt').read_text().splitlines()
                     if line.startswith('ATOM') and line[12:16].strip()=='N')
    if not ntypes or ntypes.get('NA',0):
        raise ValueError('Invalid backbone nitrogen typing')
    audit = {'status': 'ok', 'signature': signature, 'original_failed_audit': str(original / 'audit.json'),
             'max_observed_atom_displacement_A': drift, 'explicit_distant_template_exclusions': excluded,
             'retained_observed_heavy_atoms': len(kept_observed), 'source_observed_heavy_atoms': len(observed),
             'new_heavy_atoms_within_8A_of_reference': added_near, 'missing_loops_built': False,
             'missing_heavy_atoms_repaired': read_json(original/'audit.json').get('missing_heavy_atoms_repaired'),
             'backbone_nitrogen_types': dict(ntypes),
             'files': {name:sha256(output/name) for name in ('receptor.pdbqt','prepared.pdb','receptor.json','meeko_input.pdb')}}
    write_json(output/'audit.json',audit)
    write_json(output/'box.json',{'center':center.tolist(),'size':[24.]*3})
    entry=read_json(base/'sources/entry.json')
    manifest={'reference_sdf':str(base/'reference.sdf'),'receptor_pdbqt':str(output/'receptor.pdbqt'),
              'prepared_pdb':str(output/'prepared.pdb'),'center':center.tolist(),'smiles':Chem.MolToSmiles(mol),
              'canonical_smiles':Chem.MolToSmiles(mol),'state':'neutral_CCD','formalcharge':Chem.GetFormalCharge(mol),
              'formal_charge':Chem.GetFormalCharge(mol),'chemistry_usable':True,'species':'Rattus norvegicus','chain':'A',
              'construct':read_json(base/'construct_audit.json'),'ligand_audit':read_json(base/'ligand_audit.json'),
              'hetero_audit':read_json(base/'hetero_audit.json'),'receptor_audit':audit,
              'metadata':{'pdb_id':'8F6P','ligand_id':'XHO','citation':entry.get('rcsb_primary_citation'),
                          'resolution_A':entry['rcsb_entry_info'].get('resolution_combined'),
                          'source_records':read_json(base/'source_manifest.json'),
                          'scope':'Rat structural/mechanistic reference only; protein-only receptor excludes contacting 6OU lipid A2214 at 3.504 A'},
              'production_code_hashes_before':frozen,'production_code_hashes_after':code_hashes(),
              'script_sha256':sha256(__file__),
              'artifacts_sha256':{'reference_sdf':sha256(base/'reference.sdf'),'receptor_pdbqt':sha256(output/'receptor.pdbqt')}}
    if frozen!=code_hashes(): raise ValueError('Production code changed')
    write_json(base/'prepare_manifest.json',manifest)
    print(f'Prepared {base / "prepare_manifest.json"}',flush=True)


if __name__=='__main__': run()
