"""Checks that projection audits measure geometry without hidden refinement."""
import numpy as np
import pytest

from audit_nav15_projection_clashes import clash_audit,rigid_transform,original_coordinate_audit


def atom(name,xyz):
    return {'identifier':f'A:429:GLU:{name}','residue':'A:429:GLU','atom_name':name,'xyz':xyz}


def test_pairs_use_strict_two_angstrom_threshold():
    protein=[atom('CA',[0,0,0]),atom('CB',[2,0,0])]
    result=clash_audit([[0,0,0]],['C1'],protein)
    assert result['minimum_heavy_distance_A']==0
    assert result['heavy_atom_pairs_below_2A']==1
    assert result['clashing_ligand_atoms']==1
    assert result['residue_pair_counts_below_2A']=={'A:429:GLU':1}


def test_common_rigid_transformation_preserves_pair_distances():
    ligand=np.array([[0.,0.,0.],[0.,2.,0.]])
    xyz=np.array([[1.,0.,0.],[4.,0.,0.]])
    protein=[atom('CA',xyz[0].tolist()),atom('CB',xyz[1].tolist())]
    fit={'rotation_row_vector':[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],
         'translation_A':[7.,3.,-5.]}
    before=clash_audit(ligand,['C1','C2'],protein)
    after=clash_audit(rigid_transform(ligand,fit),['C1','C2'],protein,rigid_transform(xyz,fit))
    assert before['minimum_heavy_distance_A']==pytest.approx(after['minimum_heavy_distance_A'])
    assert before['heavy_atom_pairs_below_2A']==after['heavy_atom_pairs_below_2A']
    assert before['pairs_below_2A']==after['pairs_below_2A']


def test_terminal_name_permutation_requires_exact_coordinates():
    observed=[atom('O',[1.,2.,3.])]
    prepared=[atom('O',[4.,5.,6.]),atom('OXT',[1.,2.,3.])]
    result=original_coordinate_audit(prepared,observed)
    assert result['maximum_original_coordinate_displacement_A']==0
    assert result['terminal_atom_name_permutations'][0]['prepared_atom']=='A:429:GLU:OXT'
    assert result['prepared_atoms_without_original_coordinate_mapping']==['A:429:GLU:O']
    moved=original_coordinate_audit([atom('O',[4.,5.,6.]),atom('OXT',[1.01,2.,3.])],observed)
    assert moved['maximum_original_coordinate_displacement_A']>1
    assert not moved['terminal_atom_name_permutations']
