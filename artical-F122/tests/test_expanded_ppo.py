import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import torch
from Aebs.dino_latent.models import PhysicalToLatent
from Aebs.dino_latent.diagnose_expanded_failures import nearest_image_record
from Aebs.dino_latent.diagnose_interpolated_replay import InterpolatedAppearanceEnv
from Aebs.dino_latent.run_robust_ppo import RobustAppearanceEnv
from Aebs.dino_latent.run_expanded_ppo import AppearanceEnv


class ExpandedPPOTests(unittest.TestCase):
    def test_appearance_replay_and_validation_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root/"rep.pt"
            torch.save(dict(distance_scale_m=3.,latent_dimension=32,
                            physical_to_latent_state_dict=PhysicalToLatent().state_dict()),checkpoint)
            with h5py.File(root/"data.h5","w") as f:
                f.create_dataset("y_train",data=[8.,8.,8.])
            np.savez(root/"latent.npz",latent=np.stack([np.ones(32),np.ones(32)*2,np.ones(32)*99]),
                     available_indices=np.arange(3))
            rows = [dict(split=s,weather=w,color="red") for s,w in
                    [("train","clear"),("train","rain"),("validation","clear")]]
            env = AppearanceEnv(checkpoint,root/"data.h5",root/"latent.npz",rows,"image_nearest","rain|red")
            self.assertEqual(len(env.image_latents),2)
            np.testing.assert_array_equal(env.source_indices, [0, 1])
            self.assertTrue(np.all(env.reset_to(8,1)[:32] == 2))
            obs,_ = env.reset(seed=7)
            self.assertTrue(np.all(obs[:32] == 2))
            self.assertEqual(env.action_space.shape,(1,))
            self.assertEqual(env.observation_space.shape,(33,))

    def test_nearest_record_maps_train_local_index_to_cache_row(self):
        class FakeEnv:
            pools = {"overcast|light_gray": np.asarray([0, 2])}
            image_distances_m = np.asarray([7.0, 8.0, 9.0], dtype=np.float32)
            source_indices = np.asarray([10, 11, 12], dtype=np.int64)

        rows = [
            {"filename": "unused.png", "weather": "x", "color": "x"}
            for _ in range(13)
        ]
        rows[12] = {
            "filename": "target.png", "weather": "overcast", "color": "light_gray"
        }
        result = nearest_image_record(
            FakeEnv(), rows, "overcast|light_gray", distance_m=8.8
        )
        self.assertEqual(result["cache_row"], 12)
        self.assertEqual(result["filename"], "target.png")
        self.assertAlmostEqual(result["label_distance_m"], 9.0)
        self.assertAlmostEqual(result["lookup_error_m"], 0.2, places=6)

    def test_interpolated_replay_uses_adjacent_train_latents_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root/"rep.pt"
            torch.save(dict(distance_scale_m=10.,latent_dimension=32,
                            physical_to_latent_state_dict=PhysicalToLatent().state_dict()),checkpoint)
            with h5py.File(root/"data.h5","w") as f:
                f.create_dataset("y_train",data=[7.,9.,8.])
            np.savez(root/"latent.npz",
                     latent=np.stack([np.ones(32),np.ones(32)*3,np.ones(32)*99]),
                     available_indices=np.arange(3))
            rows = [dict(split="train",weather="overcast",color="gray"),
                    dict(split="train",weather="overcast",color="gray"),
                    dict(split="validation",weather="overcast",color="gray")]
            env = InterpolatedAppearanceEnv(
                checkpoint,root/"data.h5",root/"latent.npz",rows,"overcast|gray"
            )
            observation = env.reset_to(8.,1.)
            np.testing.assert_allclose(observation[:32],2.,rtol=0.,atol=1e-6)
            self.assertAlmostEqual(env.interpolation_spans_m[-1],2.)
            np.testing.assert_array_equal(env.source_indices,[0,1])

    def test_robust_residual_path_uses_train_residual_at_current_q_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root/"rep.pt"
            torch.save(dict(distance_scale_m=10.,latent_dimension=32,
                            physical_to_latent_state_dict=PhysicalToLatent().state_dict()),checkpoint)
            with h5py.File(root/"data.h5","w") as f:
                f.create_dataset("y_train",data=[7.,9.,8.])
            np.savez(root/"latent.npz",
                     latent=np.stack([np.ones(32),np.ones(32)*3,np.ones(32)*99]),
                     available_indices=np.arange(3))
            rows = [dict(split="train",weather="overcast",color="gray"),
                    dict(split="train",weather="overcast",color="gray"),
                    dict(split="validation",weather="overcast",color="gray")]
            env = RobustAppearanceEnv(
                checkpoint,root/"data.h5",root/"latent.npz",rows,
                training=False,appearance="overcast|gray",
                fixed_mode="residual_corrected",fixed_scale=1.,
            )
            with torch.no_grad():
                q_current = env.q_model(torch.tensor([[.8]],dtype=torch.float32)).numpy()[0]
            expected = q_current + env.clipped_residuals[0]
            observation = env.reset_to(8.,1.)
            np.testing.assert_allclose(observation[:32],expected,rtol=0.,atol=1e-6)
            self.assertEqual(len(env.image_latents),2)
            self.assertGreater(env.residual_radius_p95,0.)


if __name__ == "__main__":
    unittest.main()
