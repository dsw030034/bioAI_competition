"""Post-selection ring coverage and actual-grid boundary audit; no docking."""
from pathlib import Path
import sys
import itertools
import math

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdMolTransforms

from nav15_conformer_sampling import ring_torsions
from scgd.runtime import read_json, write_json, sha256

RING_ATOMS={'quinuclidine_bridge':[2,3,4,5,6,7,10,11],
            'quinoline':[12,13,14,15,16,17,18,19,20,21]}


def ring_rmsd(left,right,indices):
    return float(rdMolAlign.AlignMol(Chem.Mol(left),right,atomMap=[(i,i) for i in indices]))


def delta_degrees(a,b):
    return abs((a-b+180)%360-180)


def coverage(output):
    root=Path(output)
    report={'method':'Postselection ordered-atom ring-component alignment and circular torsion delta; no role in sampling, docking or ranking.',
            'atom_indices_zero_based':RING_ATOMS,'script_sha256':sha256(__file__),'states':{}}
    all_coverage=read_json(root/'coverage_audit.json')
    for state in ('nav15','nav15_qdn_protonated'):
        sampled=read_json(root/'ligands'/state/'report.json')
        entries=sampled['selected_entries']
        mols=[Chem.SDMolSupplier(e['sdf'])[0] for e in entries]
        native=Chem.SDMolSupplier(str(BASE/'data/reference_validation'/f'{state}.sdf'))[0]
        native_torsions={tuple(e['atoms']):e['torsion_deg'] for e in ring_torsions(native,0)}
        values=[e['aligned_heavy_rmsd_to_experimental_A'] for e in all_coverage if e['reference']==state]
        detail={'selected':len(mols),'global_pairwise_aligned_heavy_min_A':sampled['selected_min_pairwise_rmsd_A'],
                'global_vs_experimental_aligned_heavy_range_A':[min(values),max(values)]}
        for ring,indices in RING_ATOMS.items():
            pairwise=[ring_rmsd(a,b,indices) for a,b in itertools.combinations(mols,2)]
            vs_ref=[ring_rmsd(a,native,indices) for a in mols]
            quads=[q for q in native_torsions if set(q)<=set(indices)]
            native_delta=[delta_degrees(rdMolTransforms.GetDihedralDeg(m.GetConformer(),*q),native_torsions[q])
                          for m in mols for q in quads]
            between_delta=[delta_degrees(rdMolTransforms.GetDihedralDeg(a.GetConformer(),*q),
                                         rdMolTransforms.GetDihedralDeg(b.GetConformer(),*q))
                           for a,b in itertools.combinations(mols,2) for q in quads]
            detail[ring]={'selected_pairwise_aligned_rmsd_A_range':[min(pairwise),max(pairwise)],
                          'vs_experimental_aligned_rmsd_A_range':[min(vs_ref),max(vs_ref)],
                          'max_selected_ring_torsion_pair_delta_deg':max(between_delta),
                          'absolute_experimental_ring_torsion_delta_deg_range':[min(native_delta),max(native_delta)]}
        report['states'][state]=detail
    report['interpretation']='Global geometry diversity is mainly ring-to-ring and external torsions; minimized bridge-ring coordinates remain within a narrow family. The experiment does not exclude a ring-conformation sampling limitation.'
    write_json(root/'ring_coverage_audit.json',report)
    if (root/'results.json').exists():
        rows=read_json(root/'results.json')
        bounds=[]
        for row in rows:
            if row['status']!='ok':
                continue
            nominal=20 if row['box']=='pore20' else 28
            actual=math.ceil(nominal/.375)*.375
            for mode in row['all_modes']:
                margin=mode['boundary']['nominal_minimum_face_margin_A']+(actual-nominal)/2
                bounds.append({'reference':row['reference'],'conformer_id':row['conformer_id'],
                               'box':row['box'],'vina_seed':row['vina_seed'],'rank':mode['rank'],
                               'requested_size_A':nominal,'actual_grid_extent_A':actual,
                               'nominal_face_margin_A':mode['boundary']['nominal_minimum_face_margin_A'],
                               'actual_grid_face_margin_A':margin,'near_actual_grid_edge_0p05A':margin<=.05})
        write_json(root/'boundary_grid_audit.json',{'spacing_A':.375,'force_even_voxels':False,
                   'method':'Vina1.2.7 default spacing and ceil voxel count; saved coordinates are rounded to 0.001A.',
                   'rows':bounds})
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    result=coverage(args.output)
    for state,item in result['states'].items():
        print(state,'bridge pairwise',item['quinuclidine_bridge']['selected_pairwise_aligned_rmsd_A_range'],
              'whole-molecule min',item['global_pairwise_aligned_heavy_min_A'])
