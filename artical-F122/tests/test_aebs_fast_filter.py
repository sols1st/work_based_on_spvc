import numpy as np

from Aebs.semantic.conformal import StateConditionalErrorContract
from Aebs.mvp.grid_certificate_lp import bilinear_indices_weights
from Aebs.mvp.robust_sbc import certificate_masks, discrete_stopping_distance
from Aebs.semantic.robust_controller import evaluate
from Aebs.semantic.safety_filter import SafetyFilteredController
from Aebs.system.env import AebsEnv
from Aebs.system.outcomes import (
    OUT_OF_DOMAIN,
    STOPPED_SAFE_OUTSIDE_GOAL,
    SUCCESS,
    UNSAFE,
    classify_terminal_outcome,
)


class ConstantController:
    def __init__(self, action_value):
        self.action_value = float(action_value)

    def predict(self, observations, deterministic=True):
        values = np.asarray(observations)
        if values.ndim == 1:
            return np.array([self.action_value], dtype=np.float32), None
        return np.full((len(values), 1), self.action_value, dtype=np.float32), None


def test_terminal_outcomes_are_mutually_exclusive_at_boundaries():
    assert classify_terminal_outcome(6.0, 0.5) == SUCCESS
    assert classify_terminal_outcome(6.0, 0.5001) == UNSAFE
    assert classify_terminal_outcome(7.0, 0.0) == STOPPED_SAFE_OUTSIDE_GOAL
    assert classify_terminal_outcome(4.9, 0.1) == OUT_OF_DOMAIN
    assert classify_terminal_outcome(10.0, 1.0) is None


def test_environment_terminates_success_once_and_truncates_timeout():
    successful_env = AebsEnv(std1=1.0)
    successful_env.state = np.array([6.01, 0.4], dtype=np.float32)
    _, reward, terminated, truncated, info = successful_env.step(
        np.array([0.0], dtype=np.float32)
    )
    assert terminated is True
    assert truncated is False
    assert info["outcome"] == SUCCESS
    assert reward > 2.0

    timeout_env = AebsEnv(std1=1.0, max_episode_steps=1)
    timeout_env.state = np.array([10.0, 1.0], dtype=np.float32)
    _, _, terminated, truncated, info = timeout_env.step(
        np.array([0.0], dtype=np.float32)
    )
    assert terminated is False
    assert truncated is True
    assert info["outcome"] == "timeout"


def test_adaptive_filter_selects_radius_from_observed_distance():
    controller = SafetyFilteredController(
        baseline=ConstantController(0.0),
        distance_scale=2.0,
        perception_radius_m=3.0,
        radius_bin_edges_m=(5.0, 10.0, 16.0),
        radius_values_m=(0.1, 0.8),
        distance_buffer_m=0.0,
    )
    observations = np.array([[4.0, 1.0], [6.0, 1.0]], dtype=np.float32)
    _, diagnostics = controller.predict_with_diagnostics(observations)
    np.testing.assert_allclose(diagnostics["perception_radius_m"], [0.1, 0.8])
    assert controller.radius_mode == "observed_distance_bin"


def test_anti_stall_filter_does_not_brake_below_target_speed_in_one_step():
    controller = SafetyFilteredController(
        baseline=ConstantController(0.0),
        distance_scale=1.0,
        perception_radius_m=0.1,
        safety_distance_m=6.0,
        target_speed=0.45,
        distance_buffer_m=0.0,
        prevent_filter_overshoot=True,
        dt=0.05,
    )
    action, diagnostics = controller.predict_with_diagnostics(
        np.array([6.1, 0.55], dtype=np.float32)
    )
    next_speed = 0.55 - float(action[0]) * 0.05
    assert next_speed >= 0.45 - 1e-7
    assert diagnostics["overshoot_cap_active"] == 1.0


def test_corrected_evaluator_separates_unsafe_from_safe_early_stop():
    contract = StateConditionalErrorContract(
        alpha=0.05,
        bin_edges=(5.0, 16.0),
        radii=(0.0,),
        calibration_sizes=(40,),
    )
    unsafe = evaluate(ConstantController(0.0), contract, 1.0, "exact", 4, 7)
    stopped = evaluate(ConstantController(3.0), contract, 1.0, "exact", 4, 7)

    assert unsafe["outcome_counts"]["unsafe"] == 4
    assert unsafe["timeout_rate"] == 0.0
    assert stopped["outcome_counts"]["stopped_safe_outside_goal"] == 4
    assert stopped["unsafe_rate"] == 0.0
    assert sum(unsafe["outcome_counts"].values()) == 4
    assert sum(stopped["outcome_counts"].values()) == 4


def test_bilinear_grid_interpolation_indices_and_weights():
    points = np.array([[5.0, 0.0], [16.0, 3.0], [7.75, 0.75]])
    indices, weights = bilinear_indices_weights(points, distance_scale=1.0, grid_size=3)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    assert indices[0, 0] == 0
    assert weights[0, 0] == 1.0
    assert indices[1, 3] == 8
    assert weights[1, 3] == 1.0
    np.testing.assert_allclose(weights[2], [0.25, 0.25, 0.25, 0.25])


def test_certificate_speed_boundary_matches_controller_outcome():
    np.testing.assert_allclose(
        discrete_stopping_distance(np.array([0.5, 0.65]), 0.5, 3.0),
        [0.0, 0.0325],
    )
    states = np.array([[5.5, 0.5], [5.5, 0.5001]], dtype=np.float32)
    masks = certificate_masks(
        states,
        distance_scale=1.0,
        terminal_speed_threshold=0.5,
        goal_distance_m=6.0,
        goal_speed=0.5,
        unsafe_distance_low_m=5.0,
        unsafe_distance_high_m=6.0,
        unsafe_speed=0.5,
        use_recoverable_domain=True,
        max_braking=3.0,
    )
    np.testing.assert_array_equal(masks["terminal"], [True, False])
    np.testing.assert_array_equal(masks["unsafe"], [False, True])
