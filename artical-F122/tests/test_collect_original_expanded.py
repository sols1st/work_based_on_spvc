import unittest
from Aebs.connect.collect_original_expanded import distances, SCREENSHOT_REGION, TARGET


class OriginalCaptureTests(unittest.TestCase):
    def test_original_sampling_endpoints(self):
        values = distances(200)
        self.assertEqual(len(values), 400)
        self.assertEqual([values[i] for i in [0, 199, 200, 399]], [5, 10, 10, 16])

    def test_expanded_grid(self):
        values = distances(1000)
        self.assertEqual(len(values), 2000)
        self.assertEqual(values, sorted(values))

    def test_original_geometry(self):
        self.assertEqual(SCREENSHOT_REGION, dict(top=50, left=40, width=2520, height=1550))
        self.assertEqual(TARGET, dict(x=42.846504, y=-193.132416, z=0.275307))

    def test_invalid_grid(self):
        with self.assertRaises(ValueError):
            distances(1)


if __name__ == "__main__":
    unittest.main()
