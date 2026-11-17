import numpy as np
import pytest

from Aebs.visual_hover import plant
from Aebs.visual_hover.cp_boxes import DESIGN, audit, features
from Aebs.visual_hover.certificate import verify
from Aebs.conformal.trajectory_boxes import fit_shapes


def test_uniform_reproduces_original_wind_proof_and_threshold():
    shape = fit_shapes(np.zeros((2,3,2)),np.ones((2,3,3)))['fixed_uniform']
    for q in [0., .21, .77, .81]:
        a = audit(shape,q)
        assert a['passed'] == verify(DESIGN,q)['passed']
        original = verify(DESIGN,q)['margins']['transformed_velocity_update']
        assert a['certification_metric'] == pytest.approx(plant.WIND+original/plant.DT,abs=1e-13)


def test_grid_accounts_for_transformed_speed_and_endpoint_membership():
    fixed = {'name':'fixed_uniform','xcuts':[],'ycuts':[], 'halfwidths':np.full((1,1,2,3),.6).tolist()}
    widths = np.zeros((1,2,2,3))
    widths[:,0] = .6
    widths[:,1] = .1
    grid = dict(fixed, name='state_grid',ycuts=[.21],halfwidths=widths.tolist())
    assert audit(grid,.5)['certification_metric'] == pytest.approx(.595)
    assert audit(fixed,.5)['certification_metric'] == pytest.approx(.42)
    grid['xcuts']=[plant.POS_LIMIT]
    grid['halfwidths'] = np.concatenate([widths,np.full_like(widths,4.)],axis=0).tolist()
    assert not audit(grid,.5)['passed']


def test_features_follow_current_altitude_and_worst_transformed_velocity():
    state = np.array([[.1,-.3,.7,.02,.08,-.19]])
    expected=np.stack([state[:,2],np.abs(state[:,3:]+.7*state[:,:3]).max(-1)],-1)
    np.testing.assert_array_equal(features(state),expected)
