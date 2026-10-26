import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from Aebs.dino_latent.models import SafetyProjection, PhysicalToLatent
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.export_native_r2_collection import (
    EXPORT_KIND, fold_native_coordinates, require_export_trajectory_identity,
)


class NativeCollectionExportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.p = SafetyProjection().eval()
        self.q = PhysicalToLatent().eval()
        self.rep = dict(projection_state_dict=self.p.state_dict(), feature_mean=torch.zeros(384),
                        feature_std=torch.ones(384), distance_scale_m=3.16)
        self.native = dict(result=dict(family="r2", dimension=32), q_state_dict=self.q.state_dict(),
                           latent_mean=torch.randn(32), latent_std=torch.rand(32)+.2,
                           distance_mean=10.2, distance_std=3.1)

    def test_image_affine_fold_equivalent(self):
        exported = fold_native_coordinates(self.rep, self.native)
        new = SafetyProjection().eval(); new.load_state_dict(exported["projection_state_dict"])
        x = torch.randn(256, 384)
        torch.testing.assert_close(new(x), (self.p(x)-self.native["latent_mean"])/self.native["latent_std"],
                                   atol=3e-6, rtol=3e-6)
        self.assertNotIn("auxiliary_decoder_state_dict", exported)

    def test_q_distance_affine_fold_equivalent(self):
        exported = fold_native_coordinates(self.rep, self.native)
        new = PhysicalToLatent().eval(); new.load_state_dict(exported["physical_to_latent_state_dict"])
        d = torch.linspace(5, 16, 256)[:, None]
        torch.testing.assert_close(new(d/3.16), self.q((d-10.2)/3.1), atol=1e-6, rtol=1e-6)

    def test_no_source_mutation(self):
        a, b = copy.deepcopy(self.rep), copy.deepcopy(self.native)
        fold_native_coordinates(self.rep, self.native)
        for k, value in a["projection_state_dict"].items():
            self.assertTrue(torch.equal(value, self.rep["projection_state_dict"][k]))
        for k, value in b["q_state_dict"].items():
            self.assertTrue(torch.equal(value, self.native["q_state_dict"][k]))

    def test_reject_invalid_scale_and_wrong_family(self):
        self.native["latent_std"][0] = 0
        with self.assertRaises(ValueError): fold_native_coordinates(self.rep, self.native)
        self.native["result"]["family"] = "pca"
        with self.assertRaises(ValueError): fold_native_coordinates(self.rep, self.native)

    def test_old_trajectories_cannot_calibrate_new_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep, policy = Path(tmp)/"rep", Path(tmp)/"policy"
            rep.write_bytes(b"frozen representation"); policy.write_bytes(b"frozen policy")
            kind = dict(deployment_export=EXPORT_KIND)
            with self.assertRaises(ValueError): require_export_trajectory_identity({}, kind, rep, policy)
            attrs = dict(provenance_json=json.dumps(dict(representation_sha256=sha256(rep),
                         controller_sha256=sha256(policy), model_updates=False)))
            require_export_trajectory_identity(attrs, kind, rep, policy)
            attrs["provenance_json"] = attrs["provenance_json"].replace(sha256(policy), "old")
            with self.assertRaises(ValueError): require_export_trajectory_identity(attrs, kind, rep, policy)


if __name__ == "__main__": unittest.main()
