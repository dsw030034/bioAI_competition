"""Append observed 8F6P POPE A2214 as a fixed co-receptor for matched diagnostics.

No protein coordinates, atom types, or bytes are changed. Both heads have net zero
charge: plausible NH3+/phosphate- zwitterion and neutral CCD NH2/phosphate-OH.
All 49 lipid heavy atoms have deposited coordinates; no tail is reconstructed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import urllib.request

import gemmi
from meeko import MoleculePreparation, PDBQTWriterLegacy
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from scgd.geometry import pdb_atoms
from scgd.runtime import code_hashes, read_json, sha256, versions, write_json

PANEL=BASE/'outputs/nav15_multireference_20261002/rat_8f6p'
DEFAULT_OUTPUT=BASE/'outputs/nav15_environment_states_20261002/ranolazine_lipid'
CCD_URL='https://files.rcsb.org/ligands/download/6OU.cif'
STATES=('zwitterion','ccd_neutral')


def observed_lipid(pdb_path):
    rows=[line for line in Path(pdb_path).read_text().splitlines()
          if line.startswith('HETATM') and line[17:20]=='6OU' and line[21]=='A'
          and line[22:26].strip()=='2214' and line[76:78].strip() not in ('H','D')]
    atoms={}
    for line in rows:
        name=line[12:16].strip()
        if name in atoms: raise ValueError(f'Duplicate/alternate lipid atom {name}')
        atoms[name]=np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])])
    return atoms


def build_lipid(cif_path,pdb_path,state):
    if state not in STATES: raise ValueError(f'Unknown lipid state {state}')
    block=gemmi.cif.read_file(str(cif_path)).sole_block()
    atoms=[list(row) for row in block.find('_chem_comp_atom.',['atom_id','type_symbol','charge','pdbx_stereo_config']) if row[1]!='H']
    bonds=[list(row) for row in block.find('_chem_comp_bond.',['atom_id_1','atom_id_2','value_order','pdbx_stereo_config'])]
    observed=observed_lipid(pdb_path)
    names=[row[0] for row in atoms]
    if len(observed)!=49 or set(observed)!=set(names):
        raise ValueError(f'Expected complete observed POPE graph; missing={set(names)-set(observed)}, extra={set(observed)-set(names)}')
    rw,indices=Chem.RWMol(),{}
    for name,element,charge,stereo in atoms:
        atom=Chem.Atom(element); atom.SetFormalCharge(int(charge))
        if state=='zwitterion' and name=='N29': atom.SetFormalCharge(1)
        if state=='zwitterion' and name=='O25': atom.SetFormalCharge(-1); atom.SetNoImplicit(True)
        atom.SetProp('ccd_atom_id',name)
        info=Chem.AtomPDBResidueInfo(); info.SetName(f'{name:>4}'); info.SetResidueName('6OU')
        info.SetResidueNumber(2214); info.SetChainId('A'); info.SetIsHeteroAtom(True); atom.SetMonomerInfo(info)
        indices[name]=rw.AddAtom(atom)
    for first,second,order,stereo in bonds:
        if first in indices and second in indices:
            rw.AddBond(indices[first],indices[second],{'SING':Chem.BondType.SINGLE,'DOUB':Chem.BondType.DOUBLE}[order])
    mol=rw.GetMol(); Chem.SanitizeMol(mol)
    conf=Chem.Conformer(mol.GetNumAtoms()); conf.Set3D(True)
    for name,index in indices.items(): conf.SetAtomPosition(index,observed[name])
    mol.AddConformer(conf)
    Chem.AssignStereochemistryFrom3D(mol); Chem.AssignStereochemistry(mol,cleanIt=True,force=True)
    if mol.GetAtomWithIdx(indices['C20']).GetProp('_CIPCode')!='R': raise ValueError('Lipid C20 does not match CCD R configuration')
    unsaturation=[]
    for first,second,order,expected in bonds:
        if first in indices and second in indices and expected in ('E','Z'):
            actual=str(mol.GetBondBetweenAtoms(indices[first],indices[second]).GetStereo()).replace('STEREO','')
            mismatch=actual!=expected
            if mismatch and not (set((first,second))=={'C40','C41'} and expected=='Z' and actual=='E'):
                raise ValueError(f'Unreviewed lipid double-bond mismatch {first}-{second}: {actual} vs {expected}')
            unsaturation.append({'bond':[first,second],'ccd':expected,'observed_geometry':actual,
                                 'chemical_geometry_consistent':not mismatch,
                                 'policy':'Retain deposited coordinates and observed E tag for fixed occupancy diagnostic; do not claim chemically consistent native cis-POPE.' if mismatch else 'CCD and deposited geometry agree'})
    if Chem.GetFormalCharge(mol)!=0: raise ValueError('Both intended lipid microstates must have net charge zero')
    if mol.GetAtomWithIdx(indices['N29']).GetTotalNumHs()!=(3 if state=='zwitterion' else 2): raise ValueError('Invalid ethanolamine H count')
    if mol.GetAtomWithIdx(indices['O25']).GetTotalNumHs()!=(0 if state=='zwitterion' else 1): raise ValueError('Invalid phosphate H count')
    mol=Chem.AddHs(mol,addCoords=True,addResidueInfo=True)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum()==1: atom.GetPDBResidueInfo().SetName(f'H{atom.GetIdx()-len(observed)+1:03d}')
    props=AllChem.MMFFGetMoleculeProperties(mol)
    if props is None: raise ValueError('No MMFF parameters for POPE')
    ff=AllChem.MMFFGetMoleculeForceField(mol,props)
    for index in indices.values(): ff.AddFixedPoint(index)
    ff.Initialize()
    if ff.Minimize(maxIts=3000)!=0: raise ValueError('Fixed-heavy lipid hydrogen optimization failed to converge')
    drift=max(float(np.linalg.norm(mol.GetConformer().GetAtomPosition(index)-observed[name])) for name,index in indices.items())
    if drift>1e-9: raise ValueError('Hydrogen optimization moved deposited lipid heavy coordinates')
    setups=MoleculePreparation().prepare(mol)
    if len(setups)!=1: raise ValueError('Unexpected multiple lipid Meeko setups')
    pdbqt,ok,error=PDBQTWriterLegacy.write_string(setups[0])
    if not ok: raise ValueError(error)
    atom_lines=[line for line in pdbqt.splitlines() if line.startswith(('ATOM  ','HETATM'))]
    typed={line[12:16].strip():line.split()[-1] for line in atom_lines}
    serialized={line[12:16].strip():np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])])
                for line in atom_lines if line.split()[-1] not in ('H','HD')}
    if set(serialized)!=set(observed) or any(not np.array_equal(serialized[name],xyz) for name,xyz in observed.items()):
        raise ValueError('Serialized lipid coordinates or atom names differ from experiment')
    expected_n='N' if state=='zwitterion' else 'NA'
    if typed['N29']!=expected_n or typed['O25']!='OA': raise ValueError(f'Unexpected headgroup Vina atom types {typed}')
    info={'state':state,'component':'CCD-labeled POPE / phosphatidylethanolamine 16:0/18:1, 6OU; deposited tail E/Z conflict',
          'ccd_name':block.find_value('_chem_comp.name'),'ccd_formal_charge':int(block.find_value('_chem_comp.pdbx_formal_charge')),
          'formal_charge':Chem.GetFormalCharge(mol),'formula':rdMolDescriptors.CalcMolFormula(mol),
          'canonical_smiles':Chem.MolToSmiles(Chem.RemoveHs(mol)),'observed_heavy_atoms':len(observed),
          'full_ccd_heavy_atoms':len(names),'omitted_atom_ids':[],'cut_heavy_bonds':[],'cap':'none; complete observed heavy graph',
          'headgroup':{'N29_formal_charge':mol.GetAtomWithIdx(indices['N29']).GetFormalCharge(),
                       'N29_hydrogens':mol.GetAtomWithIdx(indices['N29']).GetTotalNumHs(includeNeighbors=True),
                       'O25_formal_charge':mol.GetAtomWithIdx(indices['O25']).GetFormalCharge(),
                       'O25_hydrogens':mol.GetAtomWithIdx(indices['O25']).GetTotalNumHs(includeNeighbors=True)},
          'C20_CIP':'R','double_bond_geometry':unsaturation,'atom_types':typed,
          'chemical_geometry_consistent':False,'usable_for_fixed_environment_diagnostic':True,
          'heavy_coordinate_max_drift_A':drift,'pdbqt_heavy_coordinates_exact':True,
          'hydrogen_optimization_converged':True,'partial_charge_sum_after_H_merge':sum(float(line[70:76]) for line in atom_lines),
          'rationale':('Primary plausible phosphatidylethanolamine zwitterion NH3+ / phosphate O25-; no bound-state pKa determination.'
                       if state=='zwitterion' else 'Sensitivity model uses neutral CCD NH2 / phosphate O25-H; not claimed physiological.'),
          'vina_note':'Default Vina is typed, non-electrostatic scoring; changes in formal charge act through donor/acceptor types and hydrogens, not an explicit Coulomb term. E/Z tags do not change aliphatic C40/C41 atom types or introduce a receptor geometry strain term; retained coordinates determine occupancy.'}
    return mol,atom_lines,info


def minimum_distances(first_xyz,second_xyz):
    distances=np.linalg.norm(np.asarray(first_xyz)[:,None,:]-np.asarray(second_xyz)[None,:,:],axis=2)
    return {'minimum_heavy_distance_A':float(distances.min()),'heavy_pairs_below_2A':int((distances<2).sum())}


def prepare(output):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    manifest_path=output/'prepare_manifest.json'
    if manifest_path.exists(): raise FileExistsError('Preparation already frozen; use a fresh output directory')
    baseline=read_json(PANEL/'prepare_manifest.json')
    receptor=Path(baseline['receptor_pdbqt']); prepared=Path(baseline['prepared_pdb']); reference=Path(baseline['reference_sdf'])
    raw=PANEL/'sources/8F6P.pdb'; cif=output/'6OU_ccd.cif'
    if not cif.exists():
        with urllib.request.urlopen(CCD_URL,timeout=60) as response: cif.write_bytes(response.read())
    write_json(output/'ccd_provenance.json',{'url':CCD_URL,'sha256':sha256(cif),'retrieved_utc':datetime.now(timezone.utc).isoformat()})
    frozen=code_hashes(); original=receptor.read_bytes()
    if not original.endswith(b'\n'): raise ValueError('Source protein PDBQT lacks final newline; avoid modifying source bytes')
    last_serial=max(int(line[6:11]) for line in original.decode('utf-8').splitlines() if line.startswith(('ATOM','HETATM')))
    ligand_xyz=Chem.RemoveHs(Chem.SDMolSupplier(str(reference))[0]).GetConformer().GetPositions()
    protein_xyz=np.asarray(list(pdb_atoms(prepared).values()))
    variants={}
    for state in STATES:
        directory=output/state; directory.mkdir(exist_ok=True)
        mol,typed_lines,audit=build_lipid(cif,raw,state)
        sdf=directory/'observed_6ou.sdf'
        with Chem.SDWriter(str(sdf)) as writer: writer.write(mol)
        Chem.MolToPDBFile(mol,str(directory/'observed_6ou.pdb'))
        lines=['ATOM  '+f'{last_serial+i:5d}'+line[11:] for i,line in enumerate(typed_lines,1)]
        suffix=('\n'.join(lines)+'\n').encode('ascii')
        combined=original+suffix
        if combined[:len(original)]!=original: raise ValueError('Original receptor byte prefix changed')
        if any(line.startswith(('ROOT','BRANCH','TORSDOF')) for line in combined.decode('ascii').splitlines()): raise ValueError('Unexpected ligand directives in rigid co-receptor')
        receptor_out=directory/'receptor_with_lipid.pdbqt'; receptor_out.write_bytes(combined)
        (directory/'lipid_rigid.pdbqt').write_bytes(suffix)
        lipid_xyz=Chem.RemoveHs(mol).GetConformer().GetPositions()
        audit.update(original_protein_pdbqt_bytes_unchanged=True,original_protein_bytes=len(original),
                     protein_lipid_geometry=minimum_distances(protein_xyz,lipid_xyz),
                     reference_lipid_geometry=minimum_distances(ligand_xyz,lipid_xyz),
                     receptor_sha256=sha256(receptor_out),lipid_sdf_sha256=sha256(sdf))
        obs=observed_lipid(raw)
        audit['reviewed_tail_conflict']={'ccd_double_bond':'C40=C41 Z','deposited_double_bond':'C40=C41 E',
                                        'C39_C40_C41_C42_dihedral_deg':-179.90676380195586,
                                        'atom_minimum_reference_distances_A':{name:float(np.linalg.norm(ligand_xyz-obs[name],axis=1).min()) for name in ('C39','C40','C41','C42','C47','C48','C49')},
                                        'scope':'Fixed deposited-coordinate occupancy control only; not restoration of a validated physical membrane environment.'}
        write_json(directory/'preparation_audit.json',audit)
        variants[state]={'receptor_pdbqt':str(receptor_out),'protein_prepared_pdb':str(prepared),
                         'lipid_sdf':str(sdf),'lipid_pdb':str(directory/'observed_6ou.pdb'),
                         'audit_json':str(directory/'preparation_audit.json'),'receptor_sha256':sha256(receptor_out),
                         'formal_charge':0,'protein_lipid_geometry':audit['protein_lipid_geometry'],
                         'reference_lipid_geometry':audit['reference_lipid_geometry']}
        print(f'{state}: {audit["formula"]}, N={audit["atom_types"]["N29"]}, ligand distance={audit["reference_lipid_geometry"]}',flush=True)
    if code_hashes()!=frozen: raise RuntimeError('Production files changed')
    manifest={'diagnostic_only':True,'chemical_geometry_consistent':False,'usable_for_fixed_environment_diagnostic':True,
              'pdb_id':'8F6P','lipid_residue':'6OU A2214',
              'species':'Rattus norvegicus','states':list(STATES),'variants':variants,'center':baseline['center'],'size':24.,
              'reference_sdf':str(reference),'protein_receptor_pdbqt':str(receptor),'protein_prepared_pdb':str(prepared),
              'protein_only_pairing':{'receptor_pdbqt':str(receptor),'prepared_pdb':str(prepared),'sha256':sha256(receptor)},
              'contact_validation_policy':'Use original protein prepared PDB for unchanged reference contacts. Compute lipid overlaps separately using lipid_sdf.',
              'matched_ligand_inputs_reusable':True,'ligand_input_policy':'Reuse exact frozen neutral/protonated ligand PDBQT inputs for each environment; the receptor changes only.',
              'source_hashes':{'protein_receptor':sha256(receptor),'prepared_pdb':sha256(prepared),'reference_sdf':sha256(reference),
                               'raw_8f6p':sha256(raw),'6ou_ccd':sha256(cif),'baseline_prepare_manifest':sha256(PANEL/'prepare_manifest.json')},
              'source_urls':{'raw_8f6p':'https://files.rcsb.org/download/8F6P.pdb','6ou_ccd':CCD_URL},
              'script_sha256':sha256(__file__),'production_code_hashes':frozen,'versions':versions(),
              'limitations':['Deposited C40=C41 geometry is E, conflicting with the CCD Z identity. This is an explicit fixed-coordinate occupancy diagnostic, not a validated native POPE or physical membrane restoration.',
                             'Fixed complete deposited lipid is a structural control, not a fluid membrane model.',
                             'Net-zero CCD neutral and zwitterion heads are model assumptions, not measured bound protonation.',
                             'All original rat engineered receptor preparation limitations remain; no human selectivity validation.',
                             'Same ligand inputs/seeds give matched diagnostic controls, not independent held-out validation.']}
    write_json(manifest_path,manifest)
    (output/'prepare_nav15_ranolazine_lipid.py').write_bytes(Path(__file__).read_bytes())
    print(f'Prepared {manifest_path}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    prepare(parser.parse_args().output)
