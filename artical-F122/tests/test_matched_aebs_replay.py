import numpy as np
import torch

from Aebs.experiments.matched_aebs_library import SCALE, clean_test
from Aebs.experiments.matched_aebs_replay import collect


def test_common_library_excludes_fit_and_development_pixel_aliases_across_budgets():
    small = {'files': ['train', 'dev'], 'split': ['train', 'validation']}
    large = {'files': ['copy_train', 'copy_dev', 'test_a', 'test_b'],
             'split': ['test', 'test', 'test', 'test']}
    hashes = {'train': 'x', 'dev': 'y', 'copy_train': 'x', 'copy_dev': 'y',
              'test_a': 'z', 'test_b': 'z'}
    assert clean_test([small, large], hashes) == ['test_a', 'test_b']


def test_replay_scores_live_distance_on_last_step_even_when_episode_is_unsafe():
    class Coast(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('C', torch.tensor([[1., 0.]]))

        def forward_observation(self, observation, speed):
            return torch.zeros_like(speed)

    table = dict(weather=np.repeat(['a', 'b', 'c', 'd'], 3),
                 color=np.tile(['a', 'b', 'c'], 4),
                 distance_m=np.full(12, 10.0))
    # The selected image's label is 10m and the camera estimates 9.5m.
    # At the live state 6.1m the end-to-end error is 3.4m, not the camera's
    # 0.5m frame error. The one-step unsafe termination must stay in the score.
    result = collect(Coast(), table, np.arange(12), np.full(12, 9.5 / SCALE),
                     np.array([[6.1, 3.0]]), np.array([0]))
    np.testing.assert_array_equal(result['outcomes'], ['unsafe'])
    np.testing.assert_array_equal(result['steps'], [1])
    np.testing.assert_allclose(result['scores'], [3.4], atol=1e-6)
    np.testing.assert_allclose(result['max_lookup_error_m'], [3.9], atol=1e-6)
