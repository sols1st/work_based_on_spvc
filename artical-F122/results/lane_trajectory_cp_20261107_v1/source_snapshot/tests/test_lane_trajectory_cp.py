"""Statistical-unit and finite-sample quantile regressions; not empirical proofs."""
import numpy as np
import pytest

from Aebs.lane_keeping.cp import conformal_radius
from Aebs.lane_keeping.trajectory_cp import trajectory_scores, initial_draws, coverage_summary


def test_maximum_covers_all_steps_and_coordinates():
    errors=np.zeros((4,80,2))
    errors[0,79,1]=-.12
    errors[1,0,0]=.06
    errors[2,41,1]=.2
    np.testing.assert_array_equal(trajectory_scores(errors),[.12,.06,.2,0.])
    assert len(trajectory_scores(errors))==4  # not 320 independent samples


def test_95_percent_of_frames_need_not_cover_any_complete_run():
    errors=np.zeros((100,80,2));errors[:,:4,0]=1.
    assert (abs(errors).max(2)<=.1).mean()==.95
    assert not (trajectory_scores(errors)<=.1).any()


def test_split_cp_order_statistic_and_infinity_atom():
    q,k=conformal_radius(np.arange(199,dtype=float),.05)
    assert (q,k)==(189.,190)
    q,k=conformal_radius(np.arange(18,dtype=float),.05)
    assert np.isinf(q) and k==19
    q,k=conformal_radius(np.arange(19,dtype=float),.05)
    assert (q,k)==(18.,19)
    # Existing 540-frame result rank does not change after the edge-case fix.
    assert conformal_radius(np.arange(540,dtype=float),.05)[1]==514


@pytest.mark.parametrize('values,alpha', [([], .05),([1,np.nan],.05),([-1,2],.05),([1,2],0),([1,2],1)])
def test_invalid_calibration_is_not_accepted(values,alpha):
    with pytest.raises(ValueError):conformal_radius(values,alpha)


def test_incomplete_or_malformed_errors_are_not_dropped():
    with pytest.raises(ValueError):trajectory_scores(np.zeros((80,2)))
    errors=np.zeros((3,80,2));errors[2,79,1]=np.nan
    with pytest.raises(ValueError):trajectory_scores(errors)


def test_disjoint_reproducible_rollout_draws_without_inserted_corners():
    x,k,seeds=initial_draws(2026110701,199)
    y,l,other=initial_draws(2026110702,100)
    np.testing.assert_array_equal(x,initial_draws(2026110701,199)[0])
    assert len(set(seeds)&set(other))==0
    assert (abs(x[:,0])<.25).all() and (abs(x[:,1])<.06).all()
    assert set(k)=={-.01,0.,.01}


def test_binomial_interval_only_for_independent_holdout():
    scores=np.arange(100)/100
    assert 'binomial_two_sided_95_interval' not in coverage_summary(scores,.94)
    result=coverage_summary(scores,.94,independent=True)
    assert result['covered']==95
    low,hi=result['binomial_two_sided_95_interval']
    assert low<.95<hi


def bridge_fixture():
    protocol={'horizon':80,'binding':{'actor_sha256':'actor','source_sha256':{'core.py':'plant'}},
              'distribution':{'speed':4.,'dt':.1,'kappas':[-.01,0.,.01],
                              'process_noise':0,'wheelbase':2.6,
                              'initial_low':[-.25,-.06],'initial_high':[.25,.06]}}
    calibration={'radius':.09,'finite_radius':True,'alpha':.05}
    certificate={'passed':True,'actor_sha256':'actor','verification':{'radius':.09},
        'initial':{'passed':True,'bound':.1},'unsafe':{'passed':True,'bound':23.},
        'checks':[{'passed':True,'unresolved':0,'kappa':k} for k in [-.01,0.,.01]],
        'task':{'speed':4.,'dt':.1,'kappas':[-.01,0.,.01],'process_noise':0,
                'wheelbase':2.6,'initial_halfwidths':[.25,.06]},
        'source_sha256':{'core.py':'plant'}}
    return protocol,calibration,certificate


def test_composition_requires_covering_radius_and_matching_controller():
    from Aebs.lane_keeping.compose_trajectory_cp import check_bridge
    p,c,v=bridge_fixture()
    assert check_bridge(p,c,v)['passed']
    v['verification']['radius']=.08
    assert not check_bridge(p,c,v)['passed']
    v['verification']['radius']=.1;v['actor_sha256']='changed'
    assert not check_bridge(p,c,v)['passed']


def test_composition_rejects_unresolved_or_unchecked_curvature():
    from Aebs.lane_keeping.compose_trajectory_cp import check_bridge
    p,c,v=bridge_fixture()
    v['checks'][0]['unresolved']=1
    assert not check_bridge(p,c,v)['passed']
    v['checks']=v['checks'][1:]
    assert not check_bridge(p,c,v)['passed']


def test_composition_rejects_changed_plant_initial_support_or_infinite_quantile():
    from Aebs.lane_keeping.compose_trajectory_cp import check_bridge
    p,c,v=bridge_fixture();p['distribution']['speed']=8.
    assert not check_bridge(p,c,v)['passed']
    p['distribution']['speed']=4.;p['distribution']['initial_high'][0]=.3
    assert not check_bridge(p,c,v)['passed']
    p['distribution']['initial_high'][0]=.25;c['finite_radius']=False;c['radius']=None
    assert not check_bridge(p,c,v)['passed']
