"""Dense paired CARLA capture for latent alignment; never trains or changes maps.

Reuses collect_semantic_pairs' synchronized sensors, geometry and cleanup.
Offline planning/export need only the Python standard library.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from decimal import Decimal
from pathlib import Path

from Aebs.connect import collect_semantic_pairs as capture


def distance_grid(low, high, step):
    if not all(math.isfinite(v) for v in (low, high, step)) or low <= 0 or high < low or step <= 0:
        raise ValueError("Require finite 0 < low <= high and step > 0")
    a, b, s = map(lambda v: Decimal(str(v)), (low, high, step))
    n = int((b-a)//s)
    return sorted(set([float(a+i*s) for i in range(n+1)] + [float(b)]))


def profiles():
    weather = dict(capture.WEATHERS)
    weather.update(
        clear_low_sun=dict(cloudiness=0, sun_altitude_angle=25, sun_azimuth_angle=270),
        light_rain=dict(cloudiness=50, precipitation=20, precipitation_deposits=20,
                        wetness=30, sun_altitude_angle=50))
    nominal = dict(height_m=2.0, lateral_m=0, pitch_deg=0, yaw_deg=0)
    poses = {"nominal": nominal,
             "height_low": {**nominal, "height_m": 1.7},
             "height_high": {**nominal, "height_m": 2.3},
             "yaw_left": {**nominal, "yaw_deg": -2},
             "yaw_right": {**nominal, "yaw_deg": 2}}
    return weather, poses, ["255,0,0", "40,80,160", "220,220,220"]


def build_plan(scenes, distances, shard_count=1, shard_index=0, smoke=False, preview=False):
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid shard index/count")
    if len({format(d, ".6f") for d in distances}) != len(distances):
        raise ValueError("Distances must be unique at 1 micrometre precision")
    weather, poses, colors = profiles()
    # Group-based sharding preserves all appearance variants of each state.
    old = capture.WEATHERS, capture.POSES, capture.COLORS
    try:
        capture.WEATHERS, capture.POSES, capture.COLORS = weather, poses, colors
        plan = capture.make_plan(scenes, [5.0, 8.0, 16.0] if preview else ([8.0] if smoke else distances))
    finally:
        capture.WEATHERS, capture.POSES, capture.COLORS = old
    rows = []
    for row in plan:
        if preview and (row["weather_name"] != "clear_noon" or
                        row["camera_pose_name"] != "nominal" or row["color"] != colors[0]):
            continue
        row["purpose"] = "scene_preview_not_training" if preview else "alignment_dataset"
        # Distance-based IDs remain stable when adding intermediate distances.
        group = "%s_d%s" % (row["scene_id"], format(row["distance_reference_m"], ".6f"))
        color_index = colors.index(row["color"])
        row.update(group_id=group, sample_id="%s_c%d_%s_%s" % (
            group, color_index, row["weather_name"], row["camera_pose_name"]))
        shard = int(hashlib.sha256(group.encode()).hexdigest()[:16], 16) % shard_count
        if shard == shard_index:
            rows.append(row)
    return rows


def export_labels(root):
    """Separate split CSVs, paths relative to dataset root; never random-split."""
    capture.audit(root)
    rows = [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()]
    if any(r.get("purpose") == "scene_preview_not_training" for r in rows):
        raise ValueError("Preview is for visual inspection only; do not export training labels")
    fields = ["filename", "distance_m", "group_id", "scene_id", "split", "sample_id"]
    for split in sorted({r["split"] for r in rows}):
        with (root / ("labels_%s.csv" % split)).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                if r["split"] == split:
                    writer.writerow(dict(filename=r["sensors"]["rgb"]["path"],
                                         distance_m=r["distance_center_longitudinal_m"],
                                         **{k: r[k] for k in fields[2:]}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", type=Path, default=Path(__file__).with_name("latent_alignment_scenes.example.json"))
    p.add_argument("--output", type=Path, default=Path("Aebs/carla_latent_alignment/full01"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument("--expected-map", default="")
    p.add_argument("--target-blueprint", default="vehicle.tesla.model3")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--fov", type=float, default=90)
    p.add_argument("--distance-min", type=float, default=5)
    p.add_argument("--distance-max", type=float, default=16)
    p.add_argument("--distance-step", type=float, default=0.1)
    p.add_argument("--focus-min", type=float, default=5.5)
    p.add_argument("--focus-max", type=float, default=7.5)
    p.add_argument("--focus-step", type=float, default=0.025)
    p.add_argument("--settle-ticks", type=int, default=10)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--depth", action="store_true")
    p.add_argument("--smoke", action="store_true", help="All appearance profiles/scenes at 8 m only")
    p.add_argument("--preview", action="store_true", help="Unreviewed scene QA only: red car, clear noon, nominal camera at 5/8/16 m; no training CSVs")
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--export", type=Path, help="Audit existing complete shard and regenerate split CSVs")
    args = p.parse_args()
    if args.export:
        export_labels(args.export)
        return
    if args.inspect:
        capture.collect(args, [], [])
        return
    if (args.settle_ticks < 1 or min(args.width, args.height) < 1
            or not math.isfinite(args.timeout) or args.timeout <= 0 or not 0 < args.fov < 180):
        p.error("Invalid sensor/timing parameters")
    scenes = capture.load_scenes(args.scenes)
    scene_config = json.loads(args.scenes.read_text())
    if args.preview and (args.smoke or args.shard_count != 1 or args.shard_index != 0):
        p.error("Preview cannot be combined with smoke or sharding")
    if scene_config.get("expected_map") and args.expected_map != scene_config["expected_map"] and not args.dry_run:
        p.error("--expected-map must match the scene configuration")
    distances = sorted(set(distance_grid(args.distance_min, args.distance_max, args.distance_step)
                           + [d for d in distance_grid(args.focus_min, args.focus_max, args.focus_step)
                              if args.distance_min <= d <= args.distance_max]))
    plan = build_plan(scenes, distances, args.shard_count, args.shard_index, args.smoke, args.preview)
    if not plan:
        p.error("Empty shard; reduce shard count")
    report = dict(protocol="latent_alignment_dense_v1", samples=len(plan),
                  png_files=len(plan)*(2 if args.depth else 1),
                  split_counts=dict(Counter(r["split"] for r in plan)),
                  distances_per_scene_before_sharding=3 if args.preview else (1 if args.smoke else len(distances)),
                  preview_only=args.preview,
                  state_groups=len({r["group_id"] for r in plan}),
                  shard_index=args.shard_index, shard_count=args.shard_count,
                  output=str(args.output), static_capture=True,
                  note="Not exhaustive real-world coverage; inspect all scenes before full capture")
    print(json.dumps(report, indent=2), flush=True)
    if args.dry_run:
        return
    if scene_config.get("reviewed_for_capture") is not True and not args.preview:
        p.error("Inspect scene anchors first, then set reviewed_for_capture=true in your scene JSON")
    args.collector_protocol = "latent_alignment_dense_v1"
    args.collector_sha256 = capture.digest(Path(__file__))
    args.scene_config_sha256 = capture.digest(args.scenes)
    args.plan_sha256 = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    capture.WEATHERS, capture.POSES, capture.COLORS = profiles()
    capture.collect(args, scenes, plan)
    (args.output / "coverage_plan.json").write_text(json.dumps(report, indent=2))
    (args.output / "collector_sha256.txt").write_text(capture.digest(Path(__file__))+"\n")
    if args.preview:
        capture.audit(args.output)
    else:
        export_labels(args.output)


if __name__ == "__main__":
    main()
