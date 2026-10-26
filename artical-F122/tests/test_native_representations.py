import unittest
import numpy as np
import torch
from Aebs.dino_latent.benchmark_native_representations import (
    FeatureCodec, vicreg_loss, validate_split, train_stats, fit_linear_probe,
    linear_readout, native_diagnostics, error_summary, fit_network,
)


class NativeRepresentationTests(unittest.TestCase):
    def test_codec_dimensions_and_vae_mean(self):
        for dimension in (16, 32, 64, 128, 384):
            for variational in (False, True):
                model = FeatureCodec(dimension, variational)
                value, variance = model.encode(torch.randn(4, 384))
                self.assertEqual(tuple(value.shape), (4, dimension))
                self.assertEqual(variance is None, not variational)
                self.assertTrue(torch.isfinite(model.reconstruction_loss(torch.randn(4, 384))))

    def test_split_rejects_overlap_and_test(self):
        rows = [dict(split="train", group_id="a"), dict(split="validation", group_id="b"),
                dict(split="validation", group_id="c")]
        split = dict(train=[0], selection=[1], internal_audit=[2])
        validate_split(split, rows)
        with self.assertRaises(ValueError):
            validate_split(dict(train=[0], selection=[1], internal_audit=[1]), rows)
        rows[2]["group_id"] = "a"
        with self.assertRaises(ValueError): validate_split(split, rows)
        rows[2].update(group_id="c", split="test")
        with self.assertRaises(ValueError): validate_split(split, rows)

    def test_normalization_is_train_only(self):
        x = torch.tensor([[1., 2.], [3., 4.], [1000., 1000.]])
        mean, std = train_stats(x, [0, 1])
        torch.testing.assert_close(mean, torch.tensor([2., 3.]))
        torch.testing.assert_close(std, torch.full((2,), 2. ** .5))

    def test_linear_probe_reads_native_not_r2(self):
        torch.manual_seed(7)
        z = torch.randn(100, 16)
        y = 2. * z[:, :1] - 3. * z[:, 1:2] + .7
        weights = fit_linear_probe(z, y, np.arange(80), ridge=1e-9)
        torch.testing.assert_close(linear_readout(z[80:], weights), y[80:], rtol=1e-5, atol=1e-5)

    def test_vicreg_penalizes_collapse_and_backpropagates(self):
        value = torch.randn(16, 8, requires_grad=True)
        loss = vicreg_loss(value, value + .1)
        loss.backward()
        self.assertGreater(float(value.grad.norm()), 0.)
        self.assertGreater(float(vicreg_loss(torch.zeros(16, 8), torch.zeros(16, 8))), 20.)
        with self.assertRaises(ValueError): vicreg_loss(value[:1], value[:1])

    def test_native_relative_metric_scale_invariance(self):
        z = torch.tensor([[1., 2.], [1., 2.], [3., 4.], [3., 4.]])
        groups = np.array(["a", "a", "b", "b"])
        first = native_diagnostics(z, z + .1, groups)
        second = native_diagnostics(z * 100, (z + .1) * 100, groups)
        self.assertAlmostEqual(first["relative_image_q_rmse"], second["relative_image_q_rmse"], places=5)
        self.assertEqual(first["within_distance_variance_fraction"], 0.)
        self.assertTrue(native_diagnostics(z * 0, z * 0, groups)["collapsed"])

    def test_fitting_probe_does_not_backpropagate_encoder(self):
        torch.manual_seed(7)
        encoder = torch.nn.Linear(5, 16)
        x = encoder(torch.randn(20, 5)).detach()
        probe = torch.nn.Linear(16, 1)
        fit_network(probe, x, torch.randn(20, 1), np.arange(15), np.arange(15, 20), 2, 4)
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))
        self.assertTrue(all(not p.requires_grad for p in probe.parameters()))

    def test_signed_errors_and_tails(self):
        result = error_summary(np.array([-2., 1., 0.]))
        self.assertEqual(result["mae"], 1.)
        self.assertEqual(result["max"], 2.)
        self.assertAlmostEqual(result["bias"], -1 / 3)
        with self.assertRaises(ValueError): error_summary(np.array([np.nan]))


if __name__ == "__main__": unittest.main()
