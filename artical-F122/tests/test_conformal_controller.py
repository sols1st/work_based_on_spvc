import numpy as np

from Aebs.conformal.calibrate_controller import (
    conformal_summary,
    nearest_semantic_prediction,
    select_tolerance_rank,
    trajectory_safety_score,
)


def test_nearest_semantic_prediction_uses_closest_labeled_frame():
    prediction = nearest_semantic_prediction(
        np.array([5.1, 6.8, 8.4]),
        np.array([5.0, 7.0, 9.0]),
        np.array([1.0, 2.0, 3.0]),
    )
    assert np.array_equal(prediction, np.array([1.0, 2.0, 3.0]))


def test_trajectory_score_matches_unsafe_conjunction():
    scores = trajectory_safety_score(
        np.array([5.9, 5.9, 6.1]), np.array([0.6, 0.4, 0.6])
    )
    assert scores[0] > 0
    assert scores[1] <= 0
    assert scores[2] <= 0


def test_459_samples_support_99_percent_at_99_percent_confidence():
    rank, failure_probability = select_tolerance_rank(459, 0.01, 0.01)
    assert rank == 1
    assert failure_probability <= 0.01


def test_nonpositive_worst_score_passes():
    summary = conformal_summary(-np.linspace(0.01, 1.0, 459), 0.01, 0.01)
    assert summary["rank_from_largest_l"] == 1
    assert summary["q_hat"] < 0
    assert summary["passes_q_hat_nonpositive"]
