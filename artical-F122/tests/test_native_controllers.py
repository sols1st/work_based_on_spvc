import unittest
import numpy as np
import torch
from Aebs.system.env import AebsEnv
from Aebs.dino_latent.benchmark_native_controllers import (
    encode_native, NativeInterface, NativePPOEnv, batch_step,
)
from Aebs.dino_latent.models import PhysicalToLatent
from Aebs.dino_latent.calibrate_transition_contract import next_state_residual


def fixture(dimension=16):
    q = PhysicalToLatent(latent_dim=dimension)
    ck = dict(result=dict(dimension=dimension, family="raw"), q_state_dict=q.state_dict(),
              distance_mean=torch.tensor(10.), distance_std=torch.tensor(3.))
    z = np.arange(6 * dimension, dtype=np.float32).reshape(6, dimension)
    ds = np.array([5., 7., 7., 9., 11., 13.], dtype=np.float32)
    rows = [dict(weather="a", color="b") for _ in ds]
    return ck, NativeInterface(ck, z, ds, rows, np.arange(5))


class NativeControllerTests(unittest.TestCase):
    def test_native_q_preserves_dimension_and_normalization(self):
        ck, interface = fixture()
        with torch.no_grad(): expected = interface.q(torch.tensor([[0.], [1.]]))
        torch.testing.assert_close(torch.from_numpy(interface.proxy([10., 13.])), expected)
        self.assertEqual(interface.proxy([10.]).shape, (1, 16))

    def test_nearest_only_uses_train_and_original_ties(self):
        ck, interface = fixture()
        latent, error, source = interface.nearest([8., 13.], "a|b")
        np.testing.assert_array_equal(source, [1, 4])
        np.testing.assert_allclose(error, [1., 2.])
        np.testing.assert_array_equal(latent, interface.latents[source])

    def test_no_probe_or_true_distance_in_observation(self):
        ck, interface = fixture(64)
        env = NativePPOEnv(interface, 3.)
        env.active_mode = "surrogate"
        obs = env.observe(np.array([10. / 3., 2.], np.float32))
        self.assertEqual(obs.shape, (65,))
        self.assertEqual(obs[-1], 2.)

    def test_batch_step_matches_original_environment(self):
        rng = np.random.default_rng(7)
        states = np.column_stack((rng.uniform(5, 16, 100) / 3.18, rng.uniform(0, 3, 100))).astype(np.float32)
        actions = rng.uniform(-4, 4, 100).astype(np.float32)
        ns, rewards, outcomes = batch_step(states, actions, 3.18)
        for i in range(100):
            env = AebsEnv(3.18); env.state = states[i].copy()
            expected, reward, terminal, truncated, info = env.step([actions[i]])
            np.testing.assert_allclose(ns[i], expected, atol=1e-6, rtol=0)
            self.assertAlmostEqual(rewards[i], reward, places=6)
            self.assertEqual(outcomes[i], info.get("outcome", ""))

    def test_speed_residual_sign_clipping_and_no_distance_residual(self):
        r = next_state_residual(np.array([7., 7.]), np.array([1., .02]),
                                np.array([3., 3.]), np.array([-3., 0.]))
        np.testing.assert_allclose(r[:, 0], 0.)
        np.testing.assert_allclose(r[:, 1], [-.3, -.02], atol=1e-8)

    def test_raw_feature_encoding_ignores_distance_probe(self):
        raw = np.arange(3 * 384, dtype=np.float32).reshape(3, 384)
        ck = dict(result=dict(family="raw", dimension=384), feature_mean=torch.zeros(384),
                  feature_std=torch.ones(384) * 2, latent_mean=torch.ones(384),
                  latent_std=torch.ones(384) * 3, probe_state_dict={"unusable": None})
        z = encode_native(raw, ck, {})
        np.testing.assert_allclose(z, (raw / 2 - 1) / 3, atol=1e-6)


if __name__ == "__main__": unittest.main()
