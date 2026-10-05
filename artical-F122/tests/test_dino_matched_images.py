import numpy as np
import torch

from Aebs.dino_latent.diagnose_matched_images import paired_observations, one_step
from Aebs.system.env import AebsEnv


def test_exact_image_index_and_distance_stay_paired():
    latent = np.repeat(np.array([[10], [20], [30]], dtype=np.float32), 32, axis=1)
    distances = np.array([5.2, 8.1, 15.0], dtype=np.float32)
    q = torch.nn.Linear(1, 32)
    ids, d, v, oi, oq = paired_observations(latent, distances, [2, 0], [0.5, 3.0], q, 2.0)
    assert ids.tolist() == [2, 2, 0, 0]
    np.testing.assert_array_equal(d, distances[ids])
    np.testing.assert_array_equal(oi[:, :32], latent[ids])
    np.testing.assert_array_equal(oi[:, -1], oq[:, -1])
    with torch.no_grad():
        np.testing.assert_allclose(oq[:, :32], q(torch.from_numpy(d[:, None] / 2)).numpy())


def test_one_step_uses_same_state_for_both_actions():
    env = AebsEnv(3.1827207)
    braking = one_step(env, 6.04, 1.0, 3.0)
    accelerating = one_step(env, 6.04, 1.0, -3.0)
    assert braking['distance_m'] == accelerating['distance_m']
    assert abs(braking['speed_mps'] - 0.85) < 1e-6
    assert abs(accelerating['speed_mps'] - 1.15) < 1e-6
    assert braking['outcome'] == accelerating['outcome'] == 'unsafe'
