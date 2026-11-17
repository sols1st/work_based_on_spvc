import numpy as np
import pytest

from Aebs.conformal.trajectory_boxes import METHODS, calibrate, cuts, fit_shapes, trajectory_scores, width_bounds, widths


def test_signed_boxes_and_trajectory_rank():
    rng = np.random.default_rng(11)
    f = rng.normal(size=(199, 13, 2))
    e = rng.normal(size=(199, 13, 3)) + [1., -.5, 0.]
    shapes = fit_shapes(f, e)
    assert tuple(shapes) == METHODS
    for shape in shapes.values():
        neg, pos = widths(shape, f)
        scores = trajectory_scores(shape, f, e)
        threshold = calibrate(shape, f, e)
        assert threshold['rank'] == 190
        assert threshold['q'] == sorted(scores)[189]
        q = .91
        np.testing.assert_array_equal(scores <= q, ((e >= -q*neg) & (e <= q*pos)).all(axis=(1,2)))
    assert not np.allclose(*widths(shapes['fixed_signed'], f[0, 0]))
    with pytest.raises(ValueError):
        calibrate(shapes['fixed_axis'], f[:3], e[:3])


def test_enclosures_include_cut_points_interior_extrema_and_tails():
    rng = np.random.default_rng(20)
    f = rng.uniform(-2, 2, (79, 24, 2))
    e = rng.normal(size=(79,24,3))*np.exp(f[:,:,[0,1,0]])
    for shape in fit_shapes(f, e).values():
        x, y = cuts(shape)
        for lo, hi in [([-np.inf,-np.inf],[np.inf,np.inf]), ([-.2,-.9],[.8,.5]),
                       ([-10.,-8.],[-4.,-4.]), ([0,0],[0,0])]:
            points = [np.unique(np.clip(np.r_[np.linspace(max(a,-15),min(b,15),31), knots],a,b))
                      for a,b,knots in zip(lo,hi,(x,y))]
            grid = np.stack(np.meshgrid(*points), axis=-1)
            sup = width_bounds(shape,lo,hi)
            for val, bound in zip(widths(shape,grid), sup):
                assert np.all(val <= bound + 1e-12)


def test_degenerate_development_features_and_one_sided_residuals():
    f = np.ones((30,8,2))
    e = np.zeros((30,8,2))
    e[...,0] = 1
    for shape in fit_shapes(f,e).values():
        neg, pos = width_bounds(shape,[-np.inf]*2,[np.inf]*2)
        assert (neg>0).all() and (pos>0).all()
        assert np.isfinite(trajectory_scores(shape, f, e)).all()
