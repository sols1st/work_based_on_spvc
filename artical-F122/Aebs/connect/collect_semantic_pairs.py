"""Collect paired (physical geometry, visual conditions, image) CARLA data.

Independent of the training pipeline. --dry-run and --audit need only Python.
Use a dedicated CARLA world: this client owns synchronous ticks while capturing.
"""

import argparse
import hashlib
import json
import math
import queue
import struct
import time
from collections import Counter
from pathlib import Path


WEATHERS = {
    "clear_noon": dict(cloudiness=0, sun_altitude_angle=60),
    "overcast": dict(cloudiness=90, sun_altitude_angle=45),
    "low_sun": dict(cloudiness=20, sun_altitude_angle=12, sun_azimuth_angle=180),
    "rain": dict(cloudiness=90, precipitation=60, precipitation_deposits=60,
                 wetness=70, sun_altitude_angle=40),
    "fog_ood": dict(cloudiness=80, fog_density=60, fog_distance=5,
                    fog_falloff=0.2, sun_altitude_angle=30),
    "night_ood": dict(cloudiness=30, sun_altitude_angle=-15),
}
POSES = {
    "nominal": dict(height_m=1.6, lateral_m=0, pitch_deg=0, yaw_deg=0),
    "left_up": dict(height_m=1.7, lateral_m=-0.1, pitch_deg=-2, yaw_deg=2),
    "right_down": dict(height_m=1.5, lateral_m=0.1, pitch_deg=2, yaw_deg=-2),
}
COLORS = ["255,0,0", "40,80,160"]
WEATHER_BASE = dict(cloudiness=0, precipitation=0, precipitation_deposits=0,
                    wind_intensity=0, fog_density=0, fog_distance=0,
                    fog_falloff=0.2, wetness=0, sun_altitude_angle=60,
                    sun_azimuth_angle=90)


def load_scenes(path):
    scenes = json.loads(path.read_text(encoding="utf-8"))["scenes"]
    ids, indices = set(), set()
    for scene in scenes:
        if scene["id"] in ids or scene["spawn_index"] in indices:
            raise ValueError("Scene ids and spawn indices must be unique within one run")
        if not scene["id"].replace("_", "").replace("-", "").isalnum():
            raise ValueError("Use letters, numbers, underscore or hyphen for scene id")
        if scene["split"] not in ("train", "validation", "test"):
            raise ValueError("Scene split must be train, validation, or test")
        ids.add(scene["id"])
        indices.add(scene["spawn_index"])
    if not scenes:
        raise ValueError("No scenes configured")
    return scenes


def make_plan(scenes, distances, limit=0):
    rows = []
    for scene in scenes:
        for color_index, color in enumerate(COLORS):
            for weather_name in WEATHERS:
                # OOD weather is never collected on training/validation scenes.
                if weather_name.endswith("_ood") and scene["split"] != "test":
                    continue
                for pose_name, pose in POSES.items():
                    for distance_index, distance in enumerate(distances):
                        group = "%s_d%03d" % (scene["id"], distance_index)
                        rows.append(dict(
                            sample_id="%s_c%d_%s_%s" % (group, color_index, weather_name, pose_name),
                            group_id=group, scene_id=scene["id"],
                            scene_split=scene["split"], spawn_index=scene["spawn_index"],
                            split="test_ood" if weather_name.endswith("_ood") else scene["split"],
                            distance_reference_m=float(distance), color=color,
                            weather_name=weather_name, camera_pose_name=pose_name,
                            camera_pose=pose,
                        ))
    return rows[:limit] if limit else rows


def matching_frame(sensor_queue, frame, timeout):
    deadline = time.monotonic() + timeout
    while True:
        data = sensor_queue.get(timeout=max(0.001, deadline - time.monotonic()))
        if data.frame == frame:
            return data
        if data.frame > frame:
            raise RuntimeError("Sensor skipped requested frame %d (received %d)" % (frame, data.frame))
        if time.monotonic() >= deadline:
            raise TimeoutError("Sensor frame timeout")


def transform_dict(transform):
    return dict(x=transform.location.x, y=transform.location.y, z=transform.location.z,
                pitch=transform.rotation.pitch, yaw=transform.rotation.yaw, roll=transform.rotation.roll)


def tick_sensors(world, sensors, timeout):
    """Advance once and return sensor data and the matching actor snapshot."""
    frame = world.tick(timeout)
    images = {name: matching_frame(channel, frame, timeout)
              for name, (_, channel) in sensors.items()}
    snapshot = world.get_snapshot()
    if snapshot.frame != frame:
        raise RuntimeError("Another client advanced the world during capture")
    return frame, images, snapshot


def camera_transform(carla, target_transform, distance, pose):
    yaw = math.radians(target_transform.rotation.yaw)
    forward = (math.cos(yaw), math.sin(yaw))
    right = (-forward[1], forward[0])
    loc = target_transform.location
    return carla.Transform(carla.Location(
        x=loc.x-distance*forward[0]+pose["lateral_m"]*right[0],
        y=loc.y-distance*forward[1]+pose["lateral_m"]*right[1],
        z=loc.z+pose["height_m"],
    ), carla.Rotation(pitch=pose["pitch_deg"], yaw=target_transform.rotation.yaw+pose["yaw_deg"]))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    rows = [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()]
    ids, groups, counts = set(), {}, Counter()
    for row in rows:
        if row["sample_id"] in ids:
            raise ValueError("Duplicate sample id")
        ids.add(row["sample_id"])
        # test and test_ood are both held-out; the same state can occur in both.
        split_family = row["scene_split"]
        previous = groups.setdefault(row["group_id"], split_family)
        if previous != split_family:
            raise ValueError("Paired group leaked across split families")
        for name, item in row["sensors"].items():
            if item["frame"] != row["frame"]:
                raise ValueError("Sensor/world frame mismatch")
            path = root / item["path"]
            with path.open("rb") as stream:
                header = stream.read(24)
            if header[:8] != b"\x89PNG\r\n\x1a\n":
                raise ValueError("Invalid PNG: %s" % path)
            if struct.unpack(">II", header[16:24]) != (row["width"], row["height"]):
                raise ValueError("Image dimensions differ from manifest")
            if digest(path) != item["sha256"]:
                raise ValueError("Image checksum mismatch")
        counts[row["split"]] += 1
    summary = json.loads((root / "summary.json").read_text())
    complete = summary["status"] == "complete" and len(rows) == summary["planned_samples"]
    print(json.dumps(dict(manifest_checks="passed", samples=len(rows),
                          counts=dict(counts), collection_complete=complete,
                          note="Visual scene quality and target visibility require human inspection"), indent=2))
    if not complete:
        raise RuntimeError("Collection incomplete; inspect summary.json")


def collect(args, scenes, plan):
    import carla
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    if args.inspect:
        print("server=%s map=%s" % (client.get_server_version(), world.get_map().name))
        for index, transform in enumerate(world.get_map().get_spawn_points()):
            print(json.dumps(dict(spawn_index=index, transform=transform_dict(transform))))
        return
    if not args.expected_map or world.get_map().name.split("/")[-1] != args.expected_map:
        raise ValueError("Supply --expected-map matching the running map; this script never loads a map")
    settings_before = world.get_settings()
    if settings_before.synchronous_mode or settings_before.no_rendering_mode:
        raise RuntimeError("Use a dedicated asynchronous, rendering-enabled CARLA world")
    # Prevent a pre-existing dynamic actor from destroying the controlled pairing.
    if any(a.type_id.startswith(("vehicle.", "walker.", "sensor.")) for a in world.get_actors()):
        raise RuntimeError("World contains vehicles/walkers/sensors. Use an empty dedicated server")
    spawns = world.get_map().get_spawn_points()
    for scene in scenes:
        index = scene["spawn_index"]
        if not isinstance(index, int) or not 0 <= index < len(spawns):
            raise ValueError("Invalid spawn index: %s" % index)
        if abs(spawns[index].rotation.pitch) > 2 or abs(spawns[index].rotation.roll) > 2:
            raise ValueError("Choose a near-level road anchor for the static AEBS protocol")
    args.output.mkdir(parents=True, exist_ok=False)
    weather_before = world.get_weather()
    config = dict(protocol="static_paired_visual_nuisance_v1", args={k: str(v) if isinstance(v, Path) else v
                  for k, v in vars(args).items()}, scenes=scenes, weather_profiles=WEATHERS,
                  camera_profiles=POSES, colors=COLORS, client_version=client.get_client_version(),
                  server_version=client.get_server_version(), map=world.get_map().name,
                  planned_samples=len(plan), script_sha256=digest(Path(__file__)))
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    owned, sensors, target = [], {}, None
    saved, failure = 0, None
    started = time.monotonic()
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        library = world.get_blueprint_library()
        for name, type_id in ([('rgb', 'sensor.camera.rgb')] +
                              ([('depth', 'sensor.camera.depth')] if args.depth else [])):
            blueprint = library.find(type_id)
            for key, value in dict(image_size_x=args.width, image_size_y=args.height,
                                   fov=args.fov, sensor_tick=0.0).items():
                blueprint.set_attribute(key, str(value))
            if name == "rgb":
                # Remove teleport-dependent blur; still collect real weather/light changes.
                if blueprint.has_attribute("motion_blur_intensity"):
                    blueprint.set_attribute("motion_blur_intensity", "0.0")
            actor = world.spawn_actor(blueprint, spawns[scenes[0]["spawn_index"]])
            owned.append(actor)
            channel = queue.Queue()
            actor.listen(channel.put)
            sensors[name] = (actor, channel)
        current_target = None
        with (args.output / "samples.jsonl").open("w", encoding="utf-8") as manifest:
            for row in plan:
                target_key = (row["spawn_index"], row["color"])
                if current_target != target_key:
                    if target is not None:
                        target.destroy()
                        owned.remove(target)
                    blueprint = library.find(args.target_blueprint)
                    blueprint.set_attribute("color", row["color"])
                    target = world.try_spawn_actor(blueprint, spawns[row["spawn_index"]])
                    if target is None:
                        raise RuntimeError("Target spawn failed; choose a different inspected anchor")
                    owned.append(target)
                    target.set_simulate_physics(False)
                    target.set_light_state(carla.VehicleLightState.NONE)
                    current_target = target_key
                    # A newly spawned actor may not yet be in the client's last
                    # tick cache. Never position cameras using that stale cache.
                    _, _, spawn_snapshot = tick_sensors(world, sensors, args.timeout)
                    target_state = spawn_snapshot.find(target.id)
                    if target_state is None:
                        raise RuntimeError("New target missing from synchronized snapshot")
                    target_transform = target_state.get_transform()
                weather_values = {**WEATHER_BASE, **WEATHERS[row["weather_name"]]}
                world.set_weather(carla.WeatherParameters(**weather_values))
                pose = camera_transform(carla, target_transform,
                                        row["distance_reference_m"], row["camera_pose"])
                for actor, _ in sensors.values():
                    actor.set_transform(pose)
                # Drain each sensor every tick so old frames cannot label a new pose.
                for _ in range(args.settle_ticks + 1):
                    frame, images, snapshot = tick_sensors(world, sensors, args.timeout)
                target_snapshot = snapshot.find(target.id)
                if target_snapshot is None:
                    raise RuntimeError("Target disappeared during capture")
                tf = target_snapshot.get_transform()
                measured_camera = images["rgb"].transform
                yaw = math.radians(tf.rotation.yaw)
                forward = (math.cos(yaw), math.sin(yaw))
                delta = (tf.location.x-measured_camera.location.x,
                         tf.location.y-measured_camera.location.y)
                center_longitudinal = sum(a*b for a, b in zip(delta, forward))
                if abs(center_longitudinal-row["distance_reference_m"]) > 0.01:
                    diagnostic = dict(sample_id=row["sample_id"], frame=frame,
                                      requested_distance_m=row["distance_reference_m"],
                                      measured_distance_m=center_longitudinal,
                                      difference_m=center_longitudinal-row["distance_reference_m"],
                                      target_used_for_camera=transform_dict(target_transform),
                                      target_snapshot=transform_dict(tf),
                                      camera_requested=transform_dict(pose),
                                      camera_image=transform_dict(measured_camera))
                    (args.output / "pose_mismatch.json").write_text(json.dumps(diagnostic, indent=2))
                    raise RuntimeError("Camera/target pose mismatch: requested=%.6f m, actual=%.6f m; see pose_mismatch.json"
                                       % (row["distance_reference_m"], center_longitudinal))
                vertices = target.bounding_box.get_world_vertices(tf)
                rear_longitudinal = min(
                    (v.x-measured_camera.location.x)*forward[0] +
                    (v.y-measured_camera.location.y)*forward[1] for v in vertices)
                record = {**row, "frame": frame, "timestamp": images["rgb"].timestamp,
                          "width": args.width, "height": args.height, "fov_deg": args.fov,
                          "map": world.get_map().name, "weather": weather_values,
                          "target_blueprint": args.target_blueprint, "target_actor_id": target.id,
                          "target_transform": transform_dict(tf),
                          "camera_transform": transform_dict(measured_camera),
                          "distance_center_longitudinal_m": center_longitudinal,
                          "distance_center_horizontal_m": math.hypot(*delta),
                          "distance_rear_bbox_longitudinal_m": rear_longitudinal,
                          "ego_speed_mps": None, "capture_motion": "static_no_ego_vehicle",
                          "target_speed_mps": target_snapshot.get_velocity().length(),
                          "sensors": {}}
                for name, image in images.items():
                    relative = Path(name) / (row["sample_id"] + ".png")
                    path = args.output / relative
                    path.parent.mkdir(exist_ok=True)
                    image.save_to_disk(str(path), carla.ColorConverter.Raw)
                    record["sensors"][name] = dict(path=str(relative), frame=image.frame,
                                                  sha256=digest(path))
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()
                saved += 1
                if saved == 1 or saved % 100 == 0:
                    print("captured %d/%d" % (saved, len(plan)), flush=True)
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        cleanup_errors = []
        for actor in reversed(owned):
            try:
                if actor.type_id.startswith("sensor."):
                    actor.stop()
                actor.destroy()
            except Exception as exc:
                cleanup_errors.append(repr(exc))
        for restore in (lambda: world.set_weather(weather_before),
                        lambda: world.apply_settings(settings_before)):
            try:
                restore()
            except Exception as exc:
                cleanup_errors.append(repr(exc))
        summary = dict(status="complete" if failure is None and saved == len(plan) else "incomplete",
                       planned_samples=len(plan), saved_samples=saved, error=failure,
                       runtime_seconds=time.monotonic()-started, cleanup_errors=cleanup_errors)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, default=Path(__file__).with_name("semantic_pair_scenes.example.json"))
    parser.add_argument("--output", type=Path, default=Path("Aebs/carla_semantic_pairs/run01"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--expected-map", default="")
    parser.add_argument("--target-blueprint", default="vehicle.tesla.model3")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fov", type=float, default=90)
    parser.add_argument("--distance-min", type=float, default=5)
    parser.add_argument("--distance-max", type=float, default=16)
    parser.add_argument("--distance-step", type=float, default=0.25)
    parser.add_argument("--settle-ticks", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Smoke capture only: truncate the registered plan")
    parser.add_argument("--depth", action="store_true", help="Also save synchronized raw packed-depth PNGs")
    parser.add_argument("--inspect", action="store_true", help="Read current map and spawn points; do not change world")
    parser.add_argument("--dry-run", action="store_true", help="Print capture counts without CARLA or writing files")
    parser.add_argument("--audit", type=Path, help="Check an existing output without CARLA")
    args = parser.parse_args()
    if args.audit:
        audit(args.audit)
        return
    if args.inspect:
        collect(args, [], [])
        return
    if (not all(math.isfinite(v) for v in (args.distance_step, args.distance_min,
                                          args.distance_max, args.fov, args.timeout))
            or args.distance_step <= 0 or args.distance_min <= 0 or args.distance_max < args.distance_min
            or args.settle_ticks < 1 or args.limit < 0 or min(args.width, args.height) < 1
            or not 0 < args.fov < 180 or args.timeout <= 0):
        parser.error("Invalid distance, sensor or timing parameters")
    scenes = load_scenes(args.scenes)
    n = int(math.floor((args.distance_max-args.distance_min)/args.distance_step+1e-8)) + 1
    distances = [round(args.distance_min+i*args.distance_step, 8) for i in range(n)]
    plan = make_plan(scenes, distances, args.limit)
    if args.dry_run:
        print(json.dumps(dict(samples=len(plan), images=len(plan)*(2 if args.depth else 1),
                              distance_count=n, split_counts=dict(Counter(r["split"] for r in plan)),
                              static_paired_capture=True, output=str(args.output),
                              note="Example anchors need visual inspection in your running map"), indent=2))
        return
    collect(args, scenes, plan)


if __name__ == "__main__":
    main()
