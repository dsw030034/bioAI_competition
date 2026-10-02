"""Independent checks for reference-frame transformation and residue mapping."""
import numpy as np
import gemmi
import pytest

from align_nav15_known_sites import fit, correspondence, compare, box_coverage


def test_rigid_fit_preserves_distances_and_handedness():
    source=np.array([[0.,0.,0.],[1.,0.,0.],[0.,2.,0.],[0.,0.,3.]])
    angle=.71
    rotation=np.array([[np.cos(angle),-np.sin(angle),0.],
                       [np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    target=source@rotation+np.array([4.,-8.,3.])
    estimated,translation,audit=fit(source,target)
    assert np.linalg.det(estimated)==pytest.approx(1.,abs=1e-12)
    assert np.allclose(source@estimated+translation,target,atol=1e-12)
    assert audit['ca_rmsd_A']<1e-12
    assert np.allclose(np.linalg.norm(source[:,None]-source[None,:],axis=-1),
                       np.linalg.norm(target[:,None]-target[None,:],axis=-1))


def test_sequence_insertion_does_not_shift_following_residues():
    def chain(name,seq,numbers):
        result=gemmi.Chain(name)
        for index,(resname,number) in enumerate(zip(seq,numbers)):
            res=gemmi.Residue();res.name=resname;res.seqid=gemmi.SeqId(number,' ')
            atom=gemmi.Atom();atom.name='CA';atom.element=gemmi.Element('C')
            atom.pos=gemmi.Position(index,0,0);res.add_atom(atom);result.add_residue(res)
        return result
    target=chain('B',['ALA','CYS','ASP','GLU','PHE','GLY','HIS','ILE','LYS','LEU'],range(369,379))
    source=chain('A',['ALA','CYS','ASP','GLU','PHE','TRP','GLY','HIS','ILE','LYS','LEU'],range(10,21))
    pairs,_=correspondence(source,target)
    assert len(pairs)==10
    assert all(p['identity'] for p in pairs)
    assert pairs[5]['source_residue']=='A:16:GLY'
    assert pairs[5]['target_residue']=='B:374:GLY'


def test_different_ligands_compare_without_atomwise_correspondence():
    protein=np.array([[0.,0.,0.],[12.,0.,0.]])
    ids=np.array(['B:402:PHE','B:934:PHE'])
    left=np.array([[1.,0.,0.],[2.,0.,0.]])
    right=np.array([[2.,1.,0.],[3.,1.,0.],[4.,1.,0.]])
    result=compare(left,right,protein,ids)
    assert result['contact_overlap_count_4A']==1
    assert result['contact_jaccard_4A']==1
    assert result['minimum_heavy_atom_distance_A']==pytest.approx(1.)
    assert result['centroid_distance_A']==pytest.approx(np.sqrt(3.25))
    assert not any('rmsd' in key for key in result)


def test_grid_rounding_is_distinguished_from_nominal_box():
    result=box_coverage([[10.1,0.,0.]],[0.,0.,0.],20.)
    assert result['actual_grid_extent_A']==20.25
    assert result['nominal']['heavy_atoms_outside']==1
    assert result['actual_grid']['heavy_atoms_outside']==0
    assert result['actual_grid']['minimum_heavy_atom_face_margin_A']==pytest.approx(.025)


def test_24_angstrom_box_has_exact_voxel_extent():
    result=box_coverage([[11.,0.,0.]],[0.,0.,0.],24.)
    assert result['actual_grid_extent_A']==24.
    assert result['nominal']==result['actual_grid']
