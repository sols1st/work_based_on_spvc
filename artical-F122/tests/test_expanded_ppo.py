import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import torch
from Aebs.dino_latent.models import PhysicalToLatent
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
            self.assertTrue(np.all(env.reset_to(8,1)[:32] == 2))
            obs,_ = env.reset(seed=7)
            self.assertTrue(np.all(obs[:32] == 2))
            self.assertEqual(env.action_space.shape,(1,))
            self.assertEqual(env.observation_space.shape,(33,))


if __name__ == "__main__":
    unittest.main()
