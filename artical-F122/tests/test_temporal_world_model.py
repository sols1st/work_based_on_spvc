import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from Aebs.dino_latent.temporal_world_model import (
    TemporalPredictor, advance_speed, split_trajectories, window_indices, rollout,
)
from Aebs.dino_latent.run_transformer_comparison import summary, finalize_existing, learning_arms
from Aebs.dino_latent.prepare_expanded_data import sha256


class TemporalWorldModelTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_whole_trajectory_split(self):
        train, val = split_trajectories(459)
        self.assertEqual((len(train), len(val)), (367, 92))
        self.assertFalse(set(train) & set(val))
        np.testing.assert_array_equal(train, split_trajectories(459)[0])

    def test_windows_never_cross_gap_or_trajectory(self):
        mask = np.ones((2, 12), dtype=bool)
        mask[0, 5] = False
        windows = window_indices(mask, [0], 3, 3)
        self.assertTrue(len(windows))
        for episode, end in windows:
            self.assertEqual(episode, 0)
            self.assertTrue(mask[episode, end-2:end+4].all())
        self.assertEqual(window_indices(np.ones((1, 6), bool), [0]).tolist(), [[0, 2]])

    def test_future_causal_mask(self):
        model = TemporalPredictor(8).eval()
        # The production head starts at zero; nonzero weights are necessary
        # here, otherwise persistence would trivially hide a broken mask.
        torch.nn.init.normal_(model.head.weight, std=.1)
        z, v, a = torch.randn(2, 3, 8), torch.rand(2, 3), torch.randn(2, 3)
        expected = model.sequence(z, v, a)
        z2, v2, a2 = z.clone(), v.clone(), a.clone()
        z2[:, 2] += 100; v2[:, 2] += 20; a2[:, 2] -= 2
        actual = model.sequence(z2, v2, a2)
        torch.testing.assert_close(expected[:, :2], actual[:, :2], rtol=1e-5, atol=1e-5)
        self.assertFalse(torch.allclose(expected[:, 2], actual[:, 2]))

    def test_shapes_and_gradients_both_native_spaces(self):
        for dimension, kind in ((32,"mlp"),(32,"transformer"),(384,"transformer")):
            model = TemporalPredictor(dimension,kind)
            z = torch.randn(2, 3, dimension)
            pred = rollout(model,z,torch.ones(2,3),torch.zeros(2,5),3)
            self.assertEqual(pred.shape,(2,3,dimension))
            (pred-torch.randn_like(pred)).square().mean().backward()
            self.assertGreater(float(model.head.weight.grad.norm()),0.)

    def test_persistence_and_action_causality(self):
        z = torch.randn(2,3,8)
        a = torch.zeros(2,5)
        expected = rollout(None,z,torch.ones(2,3),a,3)
        torch.testing.assert_close(expected,z[:,-1:, :].expand(-1,3,-1))
        model = TemporalPredictor(8).eval()
        torch.nn.init.normal_(model.head.weight, std=.1)
        before = rollout(model,z,torch.ones(2,3),a,3)
        a[:,4] = 3
        after = rollout(model,z,torch.ones(2,3),a,3)
        torch.testing.assert_close(before[:,:2],after[:,:2])
        self.assertFalse(torch.allclose(before[:,2],after[:,2]))

    def test_speed_clipping(self):
        actual = advance_speed(torch.tensor([.01,2.99,1.]),torch.tensor([3.,-3.,99.]))
        torch.testing.assert_close(actual,torch.tensor([0.,3.,.85]))

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError): split_trajectories(2)
        with self.assertRaises(ValueError): window_indices(np.ones(3),[0])
        with self.assertRaises(ValueError):
            rollout(None,torch.zeros(2,3,8),torch.zeros(2,3),torch.zeros(2,3),3)

    def test_signed_bias_is_not_mae(self):
        value = summary([-1.,1.])
        self.assertEqual(value["mean"],0.)
        self.assertEqual(value["mae"],1.)

    def test_no_speed_really_ignores_speed_over_three_steps(self):
        model = TemporalPredictor(8,speed_mode="none").eval()
        torch.nn.init.normal_(model.head.weight,std=.1)
        z,a = torch.randn(2,3,8),torch.randn(2,5)
        first = rollout(model,z,torch.zeros(2,3),a,3)
        second = rollout(model,z,torch.full((2,3),float("nan")),a,3)
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        self.assertEqual(model.input.in_features,9)

    def test_history_only_ignores_actions_and_speed(self):
        model = TemporalPredictor(8,speed_mode="none",use_action=False).eval()
        torch.nn.init.normal_(model.head.weight,std=.1)
        z = torch.randn(2,3,8)
        a = rollout(model,z,torch.ones(2,3),torch.zeros(2,5),3)
        b = rollout(model,z,torch.zeros(2,3),torch.full((2,5),3.),3)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(model.input.in_features,8)

    def test_separate_speed_is_used_without_concatenation(self):
        model = TemporalPredictor(8,speed_mode="add").eval()
        torch.nn.init.normal_(model.head.weight,std=.1)
        z,a = torch.randn(2,3,8),torch.zeros(2,3)
        first = model(z,torch.zeros(2,3),a)
        second = model(z,torch.ones(2,3)*3,a)
        self.assertFalse(torch.allclose(first,second))
        self.assertEqual(model.input.in_features,9)
        second.square().mean().backward()
        self.assertGreater(float(model.speed_embedding.weight.grad.norm()),0.)

    def test_default_concat_compatibility(self):
        torch.manual_seed(19)
        first = TemporalPredictor(8)
        torch.manual_seed(19)
        second = TemporalPredictor(8,speed_mode="concat",use_action=True)
        self.assertEqual(first.state_dict().keys(),second.state_dict().keys())
        for name,v in first.state_dict().items(): torch.testing.assert_close(v,second.state_dict()[name])
        self.assertEqual(len(learning_arms("speed_ablation")),8)

    def test_finalizer_rechecks_string_paths_without_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder/"source.json"
            source.write_text("frozen")
            names = ("mlp32","transformer32","transformer384")
            for name in names: (folder/(name+".pt")).write_text("unchanged candidate")
            results = {name:dict(runtime_seconds=1.) for name in names}
            results.update(persistence32={},current_q_known_distance={})
            original = dict(config=dict(source_sha256={str(source):sha256(source)}),results=results)
            (folder/"metrics.json").write_text(json.dumps(original))
            finished = finalize_existing(folder)
            self.assertTrue(finished["source_files_unchanged"])
            self.assertEqual(finished["runtime_seconds"],3.)
            self.assertEqual(finished["results"],original["results"])
            source.write_text("changed")
            with self.assertRaises(ValueError): finalize_existing(folder)


if __name__ == "__main__":
    unittest.main()
