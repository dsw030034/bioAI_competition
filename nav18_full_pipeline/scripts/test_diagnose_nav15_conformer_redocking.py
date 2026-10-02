"""Check score-only ensemble selection and unchanged reference pose thresholds."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from diagnose_nav15_conformer_redocking import STATES, VINA_SEEDS, summarize
from diagnose_nav15_conformer_redocking import verify_input_artifacts
from scgd.runtime import sha256
import pytest


def state_inputs(n=5):
    return {state:{'selected_entries':[{'conformer_id':f'c{i}'} for i in range(n)]} for state in STATES}


def good_row(conf,seed,passed=True,affinity=-6.0):
    return {'reference':'nav15','conformer_id':conf,'box':'pore20','vina_seed':seed,
            'status':'ok','top1_passed':passed,'any_output_pose_passed':True,
            'top1_rmsd_A':1.0 if passed else 10.0,'best_of_modes_rmsd_A':1.0,
            'affinity_kcal_mol':affinity,'top1_boundary':{'near_nominal_boundary_within_0p05A':False}}


def test_ensemble_uses_original_score_even_if_native_pose_is_available():
    rows=[good_row(f'c{i}',seed,passed=i!=4,affinity=-8.0 if i==4 else -6.0)
          for seed in VINA_SEEDS for i in range(5)]
    report=summarize(rows,state_inputs(),60)
    group=report['summary'][0]
    assert group['top1_pass_fraction']==.8
    assert group['pose_criteria_met_for_this_condition']
    assert group['ensemble_score_selected_top1_pass_count']==0
    assert all(r['selected_conformer_id']=='c4' for r in group['ensemble_seed_results'])
    assert report['production_gate_released'] is False


def test_failed_job_stays_in_denominator_and_prevents_complete_condition():
    rows=[good_row(f'c{i}',seed) for seed in VINA_SEEDS for i in range(5)]
    rows[-1]={'reference':'nav15','conformer_id':'c4','box':'pore20',
              'vina_seed':VINA_SEEDS[-1],'status':'failed'}
    group=summarize(rows,state_inputs(),60)['summary'][0]
    assert group['top1_pass_fraction']==14/15
    assert group['successful_jobs']==14
    assert not group['pose_criteria_met_for_this_condition']
    assert group['ensemble_complete_seed_count']==2


def test_four_distinct_conformers_cannot_satisfy_sampling_requirement():
    rows=[good_row(f'c{i}',seed) for seed in VINA_SEEDS for i in range(4)]
    group=summarize(rows,state_inputs(4),48)['summary'][0]
    assert group['top1_pass_fraction']==1.0
    assert not group['adequate_distinct_conformer_sampling']
    assert not group['pose_criteria_met_for_this_condition']


def test_resume_rejects_modified_assessment_input(tmp_path):
    path=tmp_path/'selected.sdf'
    path.write_text('original frozen assessment geometry')
    integrity={'artifacts':{'selected.sdf':sha256(path)}}
    verify_input_artifacts(tmp_path,integrity)
    path.write_text('different assessment geometry')
    with pytest.raises(ValueError,match='artifact changed'):
        verify_input_artifacts(tmp_path,integrity)
