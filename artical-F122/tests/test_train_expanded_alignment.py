import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from Aebs.dino_latent.models import SafetyDecoder, SafetyProjection, PhysicalToLatent
from Aebs.dino_latent.train_expanded_alignment import training_ids, train, metrics
from Aebs.dino_latent.train_expanded_r2_group_alignment import (
    exact_distance_groups, grouped_loss,
)


class AlignmentTests(unittest.TestCase):
    def rows(self):
        return [dict(filename=str(i),split="train" if i < 8 else "validation",
                     weather="original" if i%2 == 0 else "rain",color="255,0,0",
                     distance_m=5+i*.5,distance_bin_025m=i,group_id=str(i)) for i in range(12)]

    def test_subsets(self):
        self.assertEqual(training_ids(self.rows(),True),[0,2,4,6])
        self.assertEqual(training_ids(self.rows()),list(range(8)))

    def test_zero_error_metrics(self):
        rng = np.random.default_rng(7)
        z = rng.normal(size=(12,32))
        d = np.arange(12,dtype=float)
        result = metrics(z,z,d,d,d,self.rows())
        self.assertEqual(result["relative_latent_rmse"],0)
        self.assertEqual(result["residual_l2_p95"],0)

    def test_synthetic_training_and_save(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(seed=7,batch_size=4,eval_every=1,q_steps=2,output_dir=Path(directory))
            raw = torch.randn(12,384)
            for name in ["R0_original","R1_all_appearances"]:
                result = train(name,raw,self.rows(),args,2,{"synthetic_test":True})
                self.assertTrue((Path(directory)/name/"dino_safety_latent.pt").is_file())
                self.assertEqual(result["validation"]["samples"],4)
                self.assertFalse(result["test_evaluated"])

    def test_r2_groups_exact_distance_and_loss_is_finite(self):
        rows = []
        for group, distance in (("a",5.0),("b",7.0)):
            for appearance in range(3):
                rows.append(dict(group_id=group,distance_m=distance))
        groups = exact_distance_groups(rows,list(range(6)))
        self.assertEqual([len(group) for group in groups],[3,3])
        features = torch.randn(6,384)
        distance = torch.tensor([row["distance_m"] for row in rows])
        loss = grouped_loss(
            SafetyProjection(),SafetyDecoder(),PhysicalToLatent(),
            features,distance,1.0,groups,[0,1],1.0,1.0,
        )
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
