import tempfile
import unittest
from pathlib import Path
from Aebs.dino_latent.prepare_expanded_data import distance_group, split_groups, image_metadata, sha256


class PrepareTests(unittest.TestCase):
    def test_distance_endpoints(self):
        self.assertEqual([distance_group(d) for d in [5, 5.25, 10, 15.75, 16]], [0, 1, 20, 43, 43])

    def test_invalid_distances(self):
        for d in [float("nan"), float("inf"), 4.99, 16.01]:
            with self.assertRaises(ValueError):
                distance_group(d)

    def test_roundoff_and_duplicate_ten(self):
        self.assertEqual(distance_group(5.25-1e-12), 1)
        self.assertEqual(distance_group(10), distance_group(10.0))

    def test_partition(self):
        s = split_groups()
        self.assertEqual({k: len(v) for k,v in s.items()}, dict(train=26, validation=9, test=9))
        self.assertEqual(sorted(sum(s.values(), [])), list(range(44)))
        for low, high in [(0,10), (10,20), (20,30), (30,44)]:
            self.assertTrue(all(any(low <= g < high for g in ids) for ids in s.values()))
        self.assertEqual(s, split_groups(7))
        self.assertNotEqual(s, split_groups(8))

    def test_png_decode_constant_and_hash(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"a.png"
            Image.new("RGB", (640,640), (0,0,0)).save(p)
            result = image_metadata(p)
            self.assertTrue(result["near_constant"])
            self.assertEqual(result["sha256"], sha256(p))

    def test_reject_bad_size_or_corruption(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"a.png"
            Image.new("RGB", (32,32)).save(p)
            with self.assertRaises(ValueError):
                image_metadata(p)
            p.write_bytes(b"not an image")
            with self.assertRaises(Exception):
                image_metadata(p)


if __name__ == "__main__":
    unittest.main()
