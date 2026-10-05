import h5py
import numpy as np
import torch
import pytest

from Aebs.dino_latent.holdout_experiment import grouped_split
from Aebs.dino_latent.models import PhysicalToLatent
from Aebs.dino_latent.train_latent_ppo import LatentAebsEnv


def test_groups_and_duplicate_distances_never_cross_partitions():
    distances = np.repeat(np.linspace(5, 16, 200, dtype=np.float32), 2)
    split, groups = grouped_split(distances)
    assert split == grouped_split(distances)[0]
    assert sorted(sum(split.values(), [])) == list(range(len(distances)))
    sets = [set(groups[index]) for index in split.values()]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))


def test_lookup_cannot_access_excluded_image(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save({"distance_scale_m": 1.0, "latent_dimension": 32,
                "physical_to_latent_state_dict": PhysicalToLatent().state_dict()}, checkpoint)
    data = tmp_path / "images.h5"
    with h5py.File(data, "w") as stream:
        stream["y_train"] = np.array([5, 6, 7], dtype=np.float32)
    cache = tmp_path / "latent.npz"
    np.savez(cache, latent=np.array([np.full(32, v) for v in [10, 20, 30]], dtype=np.float32))
    env = LatentAebsEnv(checkpoint, "image_nearest", data, cache, image_indices=[0, 2])
    observation = env.reset_to(6, 2)
    assert observation.shape == (33,)
    assert observation[0] != 20
    assert env.lookup_errors_m == [1.0]
    assert observation[-1] == 2
    np.savez(cache, latent=np.ones((3, 32), dtype=np.float32), available_indices=[0, 2])
    with pytest.raises(ValueError, match="deliberately excluded"):
        LatentAebsEnv(checkpoint, "image_nearest", data, cache, image_indices=[1])
    subset_env = LatentAebsEnv(checkpoint, "image_nearest", data, cache, image_indices=[0])
    assert subset_env.reset_to(5, 1).shape == (33,)
