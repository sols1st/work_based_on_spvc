import json
import numpy as np
import pytest
import torch
from Aebs.lane_keeping import state_dependent_cp as m
from Aebs.lane_keeping.verify_state_cp import box_radius,radius


def sample(n,error):
    return dict(states=np.zeros((n,4,2)),errors=np.full((n,3,2),error))


def test_basis_coordinate_mapping_and_positive_extension():
    vertices=np.array([[-.85,-.30],[-.85,.30],[.85,-.30],[.85,.30]])
    np.testing.assert_allclose(m.basis(vertices),np.eye(4))
    points=np.random.default_rng(7).uniform(-2,2,(100,2));b=m.basis(points)
    assert (b>=0).all()
    np.testing.assert_allclose(b.sum(1),1.)


def test_holdout_uses_its_own_errors_and_all_steps():
    cal=sample(3,.01);hold=sample(3,.02);hold['errors'][1,2,1]=.5
    c=np.ones(4)*.1
    np.testing.assert_allclose(m.scores(cal,c),[.1,.1,.1])
    np.testing.assert_allclose(m.scores(hold,c),[.2,5.,.2])
    hold['errors'][0,0,0]=np.nan
    with pytest.raises(ValueError):m.scores(hold,c)


def test_box_radius_encloses_actual_state_dependent_radii():
    rng=np.random.default_rng(4)
    lo=torch.tensor(rng.uniform(-1,.5,(40,2)));hi=lo+torch.tensor(rng.uniform(0,.4,(40,2)))
    contract=dict(coefficients=[.005,.137,.056,.005],q_hat=1.72)
    upper=box_radius(lo,hi,contract)
    for _ in range(20):
        x=lo+torch.tensor(rng.random((40,2)))*(hi-lo)
        assert (radius(x,contract)<=upper).all()


def test_disjoint_protocol_exact_cp_rank_no_cap_and_holdout_cannot_change_threshold(tmp_path):
    src=tmp_path/'source';src.mkdir()
    for name in ['protocol.json','actor.pt','sbc.pt','patch_projection.pt']:(src/name).write_text('{}')
    data=sample(199,.03);data['errors'][99:]=np.linspace(.01,.12,100)[:,None,None]
    np.savez(src/'calibration.npz',**data)
    results=[]
    for i,error in enumerate([.02,1.]):
        np.savez(src/'holdout.npz',**sample(100,error))
        out=tmp_path/str(i);m.main(['--source',str(src),'--output-dir',str(out)])
        c=json.loads((out/'perception_contract.json').read_text());p=json.loads((out/'protocol.json').read_text())
        assert not set(p['fit_ids'])&set(p['calibration_ids'])
        assert len(p['fit_ids'])==99 and len(p['calibration_ids'])==100
        assert c['rank']==96 and c['cap'] is None
        scores=np.load(out/'calibration_scores.npz')['scores']
        assert c['q_hat']==np.sort(scores)[95]
        results.append(json.loads((out/'report.json').read_text()))
    assert results[0]['contract']['coefficients']==results[1]['contract']['coefficients']
    assert results[0]['contract']['q_hat']==results[1]['contract']['q_hat']
    assert results[0]['holdout_regression']['covered']==100
    assert results[1]['holdout_regression']['covered']==0
