import copy
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from Aebs.dino_latent.cache_expanded_features import RGBImages, select_records, initialize_h5
from Aebs.dino_latent.prepare_expanded_data import sha256, split_groups


class CacheTests(unittest.TestCase):
    def test_h5_roundtrip_and_alignment(self):
        import h5py
        import numpy as np
        records = [dict(filename="%d.png" % i, group_id=str(i), weather="original",
                        color="255,0,0", split=s, sha256="example", distance_m=5+i*.25,
                        distance_bin_025m=i) for i,s in enumerate(["train", "validation"])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"features.h5"
            with h5py.File(path,"w") as stream:
                features = initialize_h5(stream, records, {"test": True}, directory)
                self.assertFalse(stream.attrs["complete"])
                features[:] = np.ones((2,384), dtype=np.float32)
                stream.attrs["complete"] = True
            with h5py.File(path,"r") as stream:
                self.assertEqual(stream["features"].shape, (2,384))
                self.assertEqual(stream["train_indices"][:].tolist(), [0])
                self.assertEqual(stream["validation_indices"][:].tolist(), [1])
                self.assertEqual(stream["split"].asstr()[:].tolist(), ["train","validation"])
                self.assertEqual(stream["distance_m"][:].tolist(), [5,5.25])
                self.assertEqual(stream.attrs["test_encoded"], 0)
                self.assertTrue(stream.attrs["complete"])

    def manifest(self):
        groups = split_groups()
        rows = [dict(filename="%02d.png" % g, distance_m=5+g*.25,
                     group_id=str(g), distance_bin_025m=g, split=s)
                for s, ids in groups.items() for g in ids]
        return dict(schema="spvc_expanded_split_v1", groups=groups, records=rows)

    def test_select_excludes_test(self):
        rows, counts = select_records(self.manifest())
        self.assertEqual(len(rows), 35)
        self.assertEqual(counts, dict(train=26, validation=9, test=9))
        self.assertEqual([r["feature_row"] for r in rows], list(range(35)))
        self.assertNotIn("test", {r["split"] for r in rows})

    def test_split_tampering_rejected(self):
        m = self.manifest()
        m["records"][0]["split"] = "test"
        with self.assertRaises(ValueError):
            select_records(m)

    def test_dataset_rejects_test(self):
        with self.assertRaises(ValueError):
            RGBImages(Path("."), [dict(split="test")])

    def test_rgb_order_scale_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = root/"red.png"
            Image.new("RGB", (640,640), (255,0,0)).save(p)
            row = dict(filename=p.name, split="train", sha256=sha256(p))
            dataset = RGBImages(root, [row])
            value = dataset[0]
            self.assertEqual(tuple(value.shape), (3,640,640))
            self.assertEqual(value[:,0,0].tolist(), [1,0,0])
            Image.new("RGB", (640,640), (0,0,255)).save(p)
            with self.assertRaises(ValueError):
                dataset[0]


if __name__ == "__main__":
    unittest.main()
