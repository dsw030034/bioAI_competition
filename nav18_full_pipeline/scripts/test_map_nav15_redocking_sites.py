import argparse
import json

import numpy as np
import pytest

from map_nav15_redocking_sites import spatial_comparison,run


def test_spatial_comparison_accepts_different_molecules_without_atom_mapping():
    protein=np.array([[0.,0.,0.],[10.,0.,0.]])
    identifiers=np.array(['B:402:PHE','B:934:PHE'])
    ligand=np.array([[1.,0.,0.],[2.,0.,0.]])
    site=np.array([[2.,0.,0.],[3.,0.,0.],[4.,0.,0.]])
    result=spatial_comparison(ligand,site,protein,identifiers)
    assert result['centroid_distance_A']==pytest.approx(1.5)
    assert result['minimum_ligand_heavy_distance_A']==0
    assert result['contact_jaccard_4A']==1
    assert not any('rmsd' in key for key in result)


def test_partial_reference_is_refused_before_result_directory_creation(tmp_path):
    alignment=tmp_path/'alignment.json';alignment.write_text('{}')
    folder=tmp_path/'references/rat_6uz0/redocking';folder.mkdir(parents=True)
    (folder/'report.json').write_text(json.dumps({'complete':False,'successful_jobs':4,'expected_jobs':15}))
    (folder/'results.json').write_text('[]')
    output=tmp_path/'result'
    with pytest.raises(ValueError,match='Refuse partial'):
        run(argparse.Namespace(alignment=alignment,references=tmp_path/'references',output=output))
    assert not output.exists()
