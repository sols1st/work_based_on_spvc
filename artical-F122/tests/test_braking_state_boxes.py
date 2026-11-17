"""The distance guard must use the positive width where the car actually is."""
import numpy as np
import pytest
import torch

from Aebs.curve_braking.models import Barrier
from Aebs.curve_braking import cp_boxes
from Aebs.curve_braking.verify import verify_checkpoint


@pytest.fixture(autouse=True)
def original_barrier(tmp_path, monkeypatch):
    model = Barrier().double()
    with torch.no_grad():
        model.free.fill_(-3.)
    torch.save(model.state_dict(), tmp_path/'barrier.pt')
    monkeypatch.setattr(cp_boxes, 'BASE', tmp_path)


def test_fixed_uniform_is_original_guard():
    shape = dict(name='fixed_uniform', xcuts=[], ycuts=[], halfwidths=[[[[1]*3, [1]*3]]])
    for q in (.1, .675, .75):
        local = cp_boxes.audit(shape, q)
        original = verify_checkpoint(cp_boxes.BASE/'barrier.pt', q)
        assert local['passed'] == original['passed']
        assert local['certification_metric'] == original['checks']['brake_floor_guard_slack_m']


def test_large_short_range_errors_need_no_global_worst_case():
    # Near the obstacle at d<4, the camera can overestimate by as much as 2 m;
    # d+2 never reaches 7.5 there. A global box would incorrectly pair 6.15+2.
    width = np.ones((2,1,2,3))
    width[1,0,1,2] = .1
    shape = dict(name='state_signed', xcuts=[4.], ycuts=[], halfwidths=width.tolist())
    result = cp_boxes.audit(shape,1.)
    assert result['passed']
    np.testing.assert_allclose(result['certification_metric'], 7.5-6.35)
    assert result['global_envelope_slack_m'] < 0
