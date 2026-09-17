import json

import numpy as np
import pytest

from Aebs.mvp.finite_state_sbc import load_closure_evidence, solve_abstract_sbc


def test_supermartingale_lp_has_zero_conditional_bound():
    result, matrix, rhs, _ = solve_abstract_sbc()
    assert result.success
    assert np.max(matrix @ result.x - rhs) <= 1e-9
    assert result.x[0] == pytest.approx(0.0)
    assert result.x[1] == pytest.approx(0.0)
    assert result.x[2] == pytest.approx(1.0)
    assert result.x[3] == pytest.approx(0.0)


def test_closure_evidence_rejects_unresolved_cells(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "status": "partial_local_verification",
                "student": (
                    "results/mvp/14_standalone_ppo_dual_boundary_repair/"
                    "standalone_ppo_repaired.zip"
                ),
                "partition": {"unresolved_cells": 1},
                "region": {
                    "distance_low_m": 5.0,
                    "distance_high_m": 16.0,
                    "speed_low_m_per_s": 0.5,
                    "speed_high_m_per_s": 3.0,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid closure evidence"):
        load_closure_evidence(path)
