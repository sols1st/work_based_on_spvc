import unittest
from Aebs.connect.collect_spvc_expanded import MAP_INDEX, SPAWN_INDEX, distances, BASE_WEATHER


class SpvcCaptureTests(unittest.TestCase):
    def test_original_map_and_spawn(self):
        self.assertEqual((MAP_INDEX, SPAWN_INDEX), (3, 4))

    def test_original_400_distances(self):
        d = distances(200)
        self.assertEqual(len(d), 400)
        self.assertEqual([d[i] for i in [0, 199, 200, 399]], [5, 10, 10, 16])

    def test_dense_plan_and_weather(self):
        self.assertEqual(len(distances(1000))*4*3, 24000)
        self.assertEqual(BASE_WEATHER["sun_altitude_angle"], 60)
        self.assertEqual(BASE_WEATHER["sun_azimuth_angle"], 90)

    def test_invalid_sampling(self):
        with self.assertRaises(ValueError):
            distances(1)


if __name__ == "__main__":
    unittest.main()
