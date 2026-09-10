import numpy as np
import torch

from Aebs.semantic.conformal import (
    MondrianSplitConformalRegressor,
    SplitConformalRegressor,
    StateConditionalErrorContract,
)
from Aebs.semantic.model import SemanticEncoder


def test_semantic_encoder_shapes_and_positive_scale():
    model = SemanticEncoder(input_dim=16, hidden_dims=(8, 4))
    mean, scale = model(torch.randn(5, 4, 4))
    assert mean.shape == (5, 1)
    assert scale.shape == (5, 1)
    assert torch.all(scale > 0)


def test_split_conformal_uses_finite_sample_order_statistic():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    mean = np.zeros_like(y)
    scale = np.ones_like(y)
    calibrator = SplitConformalRegressor.fit(y, mean, scale, alpha=0.25)
    # ceil((4 + 1) * .75) = 4, so the score is the largest residual.
    assert calibrator.quantile == 3.0
    lower, upper = calibrator.interval(np.array([1.0]), np.array([2.0]))
    np.testing.assert_allclose(lower, [-5.0])
    np.testing.assert_allclose(upper, [7.0])


def test_mondrian_conformal_selects_quantile_from_prediction_bin():
    mean = np.arange(8, dtype=float)
    y = mean + np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=float)
    scale = np.ones_like(mean)
    calibrator = MondrianSplitConformalRegressor.fit(y, mean, scale, alpha=0.5, n_bins=2)
    lower, upper = calibrator.interval(np.array([1.0, 6.0]), np.ones(2))
    np.testing.assert_allclose(upper, [3.0, 10.0])
    np.testing.assert_allclose(lower, [-1.0, 2.0])


def test_state_conditional_error_contract_bounds_estimate():
    state = np.array([0.1, 0.2, 0.7, 0.8])
    truth = state.copy()
    estimate = truth + np.array([0.1, 0.2, 0.3, 0.4])
    contract = StateConditionalErrorContract.fit(
        truth, estimate, state, bin_edges=np.array([0.0, 0.5, 1.0]), alpha=0.5
    )
    np.testing.assert_allclose(contract.radii, [0.2, 0.4])
    lower, upper = contract.estimate_interval(np.array([0.25, 0.75]))
    np.testing.assert_allclose(lower, [0.05, 0.35])
    np.testing.assert_allclose(upper, [0.45, 1.15])
