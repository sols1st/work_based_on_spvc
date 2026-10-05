"""Offline invariants for CARLA paired collection (no CARLA dependency)."""

import contextlib
import io
import json
import queue
import struct
import tempfile
import unittest
import zlib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from Aebs.connect.collect_semantic_pairs import audit, digest, load_scenes, make_plan, matching_frame, tick_sensors


SCENES = [dict(id="s%d" % i, spawn_index=i, split=split)
          for i, split in enumerate(["train", "validation", "test"])]


class CollectionTests(unittest.TestCase):
    def test_tick_returns_same_frame_actor_snapshot(self):
        channel = queue.Queue()
        image = SimpleNamespace(frame=11)
        channel.put(image)
        snapshot = SimpleNamespace(frame=11)
        world = SimpleNamespace(tick=lambda timeout: 11, get_snapshot=lambda: snapshot)
        frame, images, actual = tick_sensors(world, {"rgb": (None, channel)}, 1)
        self.assertEqual(frame, 11)
        self.assertIs(images["rgb"], image)
        self.assertIs(actual, snapshot)

    def test_tick_rejects_other_client_advance(self):
        channel = queue.Queue()
        channel.put(SimpleNamespace(frame=11))
        world = SimpleNamespace(tick=lambda timeout: 11, get_snapshot=lambda: SimpleNamespace(frame=12))
        with self.assertRaises(RuntimeError):
            tick_sensors(world, {"rgb": (None, channel)}, 1)

    def test_plan_counts_and_heldout_weather(self):
        plan = make_plan(SCENES, [5+i*0.25 for i in range(45)])
        self.assertEqual(len(plan), 3780)
        self.assertEqual(Counter(r["split"] for r in plan),
                         dict(train=1080, validation=1080, test=1080, test_ood=540))
        self.assertEqual(len({r["sample_id"] for r in plan}), len(plan))
        for row in plan:
            if row["weather_name"].endswith("_ood"):
                self.assertEqual(row["scene_split"], "test")

    def test_pair_group_keeps_distance_and_split(self):
        plan = make_plan(SCENES, [5, 6])
        groups = {}
        for row in plan:
            values = groups.setdefault(row["group_id"], set())
            values.add((row["distance_reference_m"], row["scene_split"]))
        self.assertTrue(all(len(values) == 1 for values in groups.values()))
        self.assertEqual(Counter(r["group_id"] for r in plan)["s0_d000"], 24)

    def test_rejects_shared_anchor_between_train_and_test(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenes.json"
            scenes = [dict(id="train", split="train", spawn_index=0),
                      dict(id="test", split="test", spawn_index=0)]
            path.write_text(json.dumps(dict(scenes=scenes)))
            with self.assertRaises(ValueError):
                load_scenes(path)

    def test_frame_match_discards_old_frame(self):
        channel = queue.Queue()
        channel.put(SimpleNamespace(frame=10))
        wanted = SimpleNamespace(frame=11)
        channel.put(wanted)
        self.assertIs(matching_frame(channel, 11, 1), wanted)

    def test_frame_match_rejects_future_frame(self):
        channel = queue.Queue()
        channel.put(SimpleNamespace(frame=12))
        with self.assertRaises(RuntimeError):
            matching_frame(channel, 11, 1)

    def test_frame_match_times_out(self):
        with self.assertRaises(queue.Empty):
            matching_frame(queue.Queue(), 11, 0.01)

    def test_audit_detects_modified_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def chunk(name, data):
                return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name+data))
            png = (b"\x89PNG\r\n\x1a\n" +
                   chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)) +
                   chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00")) + chunk(b"IEND", b""))
            path = root / "one.png"
            path.write_bytes(png)
            row = dict(sample_id="one", group_id="one", scene_split="train", split="train",
                       frame=1, width=1, height=1,
                       sensors=dict(rgb=dict(path="one.png", frame=1, sha256=digest(path))))
            (root / "samples.jsonl").write_text(json.dumps(row)+"\n")
            (root / "summary.json").write_text(json.dumps(dict(status="complete", planned_samples=1)))
            with contextlib.redirect_stdout(io.StringIO()):
                audit(root)
            path.write_bytes(png+b"modified")
            with self.assertRaises(ValueError):
                audit(root)


if __name__ == "__main__":
    unittest.main()
