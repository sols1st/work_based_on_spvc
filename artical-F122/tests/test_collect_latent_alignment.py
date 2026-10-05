import unittest
from collections import Counter
from Aebs.connect.collect_latent_alignment import build_plan, distance_grid


class DenseCaptureTests(unittest.TestCase):
    def setUp(self):
        self.scenes = [dict(id="s%d" % i, spawn_index=i, split=s)
                       for i, s in enumerate(["train", "validation", "test"])]

    def test_endpoints_and_invalid_grid(self):
        self.assertEqual(distance_grid(5, 5.25, .1), [5, 5.1, 5.2, 5.25])
        for step in [0, -1, float("nan")]:
            with self.assertRaises(ValueError):
                distance_grid(5, 16, step)

    def test_variants_splits_and_no_duplicates(self):
        plan = build_plan(self.scenes, [5, 16])
        self.assertEqual(Counter(r["split"] for r in plan),
                         dict(train=180, validation=180, test=180, test_ood=60))
        self.assertEqual(len({r["sample_id"] for r in plan}), len(plan))

    def test_shards_disjoint_complete_and_keep_groups(self):
        plan = build_plan(self.scenes, [5, 6, 16])
        parts = [build_plan(self.scenes, [5, 6, 16], 3, i) for i in range(3)]
        self.assertEqual(sum(map(len, parts)), len(plan))
        self.assertEqual({r["sample_id"] for r in plan},
                         {r["sample_id"] for part in parts for r in part})
        groups = [{r["group_id"] for r in part} for part in parts]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

    def test_ids_stable_under_denser_grid(self):
        a = build_plan(self.scenes, [5, 16])
        b = build_plan(self.scenes, [5, 6, 16])
        self.assertTrue({r["sample_id"] for r in a} <= {r["sample_id"] for r in b})

    def test_smoke_covers_all_scenes(self):
        plan = build_plan(self.scenes, [5, 16], smoke=True)
        self.assertEqual({r["scene_id"] for r in plan}, {s["id"] for s in self.scenes})
        self.assertEqual({r["distance_reference_m"] for r in plan}, {8})

    def test_preview_is_small_and_labeled(self):
        plan = build_plan(self.scenes, [5, 16], preview=True)
        self.assertEqual(len(plan), 9)
        self.assertEqual({r["distance_reference_m"] for r in plan}, {5, 8, 16})
        self.assertEqual({r["purpose"] for r in plan}, {"scene_preview_not_training"})
        self.assertEqual({r["weather_name"] for r in plan}, {"clear_noon"})


if __name__ == "__main__":
    unittest.main()
