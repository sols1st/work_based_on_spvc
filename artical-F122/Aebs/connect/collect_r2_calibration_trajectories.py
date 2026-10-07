"""Collect independent synchronized CARLA trajectories for the frozen R2 contract.

Run against a dedicated, rendering-enabled CARLA server.  The script never
loads or changes maps, never reads train/validation/test images, and never
updates DINO, R2, or PPO.  Each trajectory follows the deployed image-latent
controller and stores raw frozen-DINO features plus synchronized physical
states in the strict calibration H5 schema.
"""

import argparse
import json
import math
import queue
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
from stable_baselines3 import PPO

from Aebs.connect.collect_semantic_pairs import (
    WEATHER_BASE, camera_transform, matching_frame, transform_dict,
)
from Aebs.connect.collect_spvc_expanded import WEATHERS
from Aebs.dino_latent.backbone import FrozenDinoV2
from Aebs.dino_latent.cache_expanded_features import repository_hash
from Aebs.dino_latent.calibrate_transition_contract import SCHEMA
from Aebs.dino_latent.models import SafetyProjection
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.system.outcomes import TIMEOUT, classify_terminal_outcome


COLORS = ("255,0,0", "40,80,160", "220,220,220")
NOMINAL_CAMERA = dict(height_m=2.0, lateral_m=0.0, pitch_deg=0.0, yaw_deg=0.0)


def episode_plan(episodes, seed, weather_names=None, colors=COLORS):
    weather_names = tuple(sorted(WEATHERS) if weather_names is None else weather_names)
    colors = tuple(colors)
    if episodes < 1 or not weather_names or not colors:
        raise ValueError("episodes, weathers, and colors must be nonempty")
    if len(set(weather_names)) != len(weather_names) or any(name not in WEATHERS for name in weather_names):
        raise ValueError("weather names must be unique registered R2 profiles")
    if len(set(colors)) != len(colors):
        raise ValueError("colors must be unique")
    rng = np.random.default_rng(seed)
    plan = []
    for index in range(episodes):
        plan.append({
            "trajectory_id": "calibration_seed%d_%06d" % (seed, index),
            "initial_distance_m": float(rng.uniform(15.0, 16.0)),
            "initial_speed_mps": float(rng.uniform(2.5, 3.0)),
            "weather": weather_names[int(rng.integers(len(weather_names)))],
            "color": colors[int(rng.integers(len(colors)))],
        })
    return plan


def advance_state(distance_m, speed_mps, action, dt=0.05):
    action = float(np.clip(action, -3.0, 3.0))
    return (
        float(distance_m - speed_mps * dt),
        float(np.clip(speed_mps - action * dt, 0.0, 3.0)),
    )


def bgra_to_rgb_tensor(raw_data, width, height):
    value = np.frombuffer(raw_data, dtype=np.uint8)
    if value.size != width * height * 4:
        raise ValueError("CARLA RGB frame has unexpected byte count")
    bgra = value.reshape(height, width, 4)
    rgb = bgra[:, :, :3][:, :, ::-1].copy()
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)


def initialize_h5(stream, plan, horizon, provenance):
    episodes = len(plan)
    stream.attrs.update({
        "schema": SCHEMA,
        "complete": False,
        "split_role": "calibration",
        "independent_from_training": True,
        "used_for_model_selection": False,
        "horizon": int(horizon),
        "dino_weights_sha256": provenance["dino_weights_sha256"],
        "provenance_json": json.dumps(provenance),
    })
    strings = h5py.string_dtype(encoding="utf-8")
    stream.create_dataset(
        "trajectory_id", data=np.asarray([row["trajectory_id"] for row in plan], dtype=object),
        dtype=strings,
    )
    stream.create_dataset(
        "weather", data=np.asarray([row["weather"] for row in plan], dtype=object), dtype=strings,
    )
    stream.create_dataset(
        "color", data=np.asarray([row["color"] for row in plan], dtype=object), dtype=strings,
    )
    stream.create_dataset("initial_distance_m", data=[row["initial_distance_m"] for row in plan])
    stream.create_dataset("initial_speed_mps", data=[row["initial_speed_mps"] for row in plan])
    features = stream.create_dataset(
        "dino_features", shape=(episodes, horizon, 384), dtype="float32",
        chunks=(1, min(horizon, 64), 384), compression="lzf",
    )
    distance = stream.create_dataset("distance_m", shape=(episodes, horizon), dtype="float32")
    speed = stream.create_dataset("speed_mps", shape=(episodes, horizon), dtype="float32")
    valid = stream.create_dataset("valid", shape=(episodes, horizon), dtype="bool")
    action = stream.create_dataset("image_action", shape=(episodes, horizon), dtype="float32")
    valid[:] = False
    return features, distance, speed, valid, action


def preflight(args, parser):
    controller_path = args.ppo_dir / "latent_ppo.zip"
    metrics_path = args.ppo_dir / "metrics.json"
    required = [
        args.representation, controller_path, metrics_path,
        args.training_cache_manifest, args.dino_repo / "hubconf.py", args.dino_weights,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("missing inputs: " + ", ".join(missing))
    representation = torch.load(args.representation, map_location="cpu")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    training_cache = json.loads(args.training_cache_manifest.read_text(encoding="utf-8"))
    expected_dino_hash = training_cache.get("provenance", {}).get("dino_weights_sha256")
    if (
        representation.get("variant") != "R2_group_alignment"
        or representation.get("dino_frozen") is not True
        or metrics.get("representation_sha256") != sha256(args.representation)
        or metrics.get("checkpoint_sha256") != sha256(controller_path)
        or metrics.get("test_used") is not False
        or sha256(args.dino_weights) != expected_dino_hash
    ):
        parser.error("R2/PPO/DINO provenance mismatch")
    return representation, controller_path, expected_dino_hash


def collect(args, plan, representation, controller_path, expected_dino_hash):
    import carla

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    torch.set_num_threads(1)
    backbone = FrozenDinoV2(args.dino_repo, args.dino_weights, device).eval()
    projection = SafetyProjection().to(device).eval()
    projection.load_state_dict(representation["projection_state_dict"])
    for parameter in projection.parameters():
        parameter.requires_grad_(False)
    feature_mean = representation["feature_mean"].to(device)
    feature_std = representation["feature_std"].to(device)
    controller = PPO.load(controller_path, device=device_name)
    for model in (backbone, projection):
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("DINO/projection must stay frozen during calibration collection")

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    if not args.expected_map or world.get_map().name.split("/")[-1] != args.expected_map:
        raise ValueError("--expected-map must match the running map; collector never loads a map")
    settings_before = world.get_settings()
    if settings_before.synchronous_mode or settings_before.no_rendering_mode:
        raise RuntimeError("use a dedicated asynchronous, rendering-enabled CARLA world")
    if any(actor.type_id.startswith(("vehicle.", "walker.", "sensor.")) for actor in world.get_actors()):
        raise RuntimeError("dedicated world must not contain vehicles, walkers, or sensors")
    spawns = world.get_map().get_spawn_points()
    if not 0 <= args.spawn_index < len(spawns):
        raise ValueError("invalid spawn index")
    anchor = spawns[args.spawn_index]
    if abs(anchor.rotation.pitch) > 2 or abs(anchor.rotation.roll) > 2:
        raise ValueError("spawn anchor must be on a near-level road")

    provenance = {
        "collector": "R2_independent_online_calibration_trajectories_v1",
        "registered_before_collection": True,
        "calibration_seed": args.seed,
        "episodes": len(plan),
        "horizon": args.horizon,
        "initial_distribution": "distance uniform[15,16], speed uniform[2.5,3]",
        "appearance_distribution": "weather and color sampled independently and uniformly per trajectory",
        "weather_profiles": {name: {**WEATHER_BASE, **WEATHERS[name]} for name in args.weathers},
        "colors": list(args.colors),
        "camera": {**NOMINAL_CAMERA, "width": args.width, "height": args.height, "fov": args.fov},
        "deployment_rollout": "image latent PPO; q path is not used to advance collection states",
        "representation_sha256": sha256(args.representation),
        "controller_sha256": sha256(controller_path),
        "dino_weights_sha256": expected_dino_hash,
        "dino_source_sha256": repository_hash(args.dino_repo),
        "training_cache_manifest_sha256": sha256(args.training_cache_manifest),
        "test_used": False,
        "model_updates": False,
        "device": device_name,
        "map": world.get_map().name,
        "spawn_index": args.spawn_index,
        "server_version": client.get_server_version(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "config.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8"
    )
    partial = args.output_dir / "calibration_trajectories.partial.h5"
    final_h5 = args.output_dir / "calibration_trajectories.h5"
    settings = world.get_settings()
    weather_before = world.get_weather()
    owned = []
    target = sensor = None
    saved_steps = completed = 0
    outcomes = Counter()
    failure = None
    started = time.monotonic()
    try:
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        library = world.get_blueprint_library()
        camera_bp = library.find("sensor.camera.rgb")
        for key, value in {
            "image_size_x": args.width, "image_size_y": args.height,
            "fov": args.fov, "sensor_tick": 0.0,
        }.items():
            camera_bp.set_attribute(key, str(value))
        if camera_bp.has_attribute("motion_blur_intensity"):
            camera_bp.set_attribute("motion_blur_intensity", "0.0")
        sensor = world.spawn_actor(camera_bp, anchor)
        owned.append(sensor)
        image_queue = queue.Queue()
        sensor.listen(image_queue.put)
        trajectory_outcomes = []
        with h5py.File(partial, "w") as stream:
            feature_ds, distance_ds, speed_ds, valid_ds, action_ds = initialize_h5(
                stream, plan, args.horizon, provenance
            )
            for episode, row in enumerate(plan):
                if target is not None:
                    target.destroy()
                    owned.remove(target)
                vehicle_bp = library.find(args.target_blueprint)
                vehicle_bp.set_attribute("color", row["color"])
                target = world.try_spawn_actor(vehicle_bp, anchor)
                if target is None:
                    raise RuntimeError("target spawn failed")
                owned.append(target)
                target.set_simulate_physics(False)
                target.set_light_state(carla.VehicleLightState.NONE)
                world.set_weather(carla.WeatherParameters(
                    **{**WEATHER_BASE, **WEATHERS[row["weather"]]}
                ))
                frame = world.tick(args.timeout)
                matching_frame(image_queue, frame, args.timeout)
                snapshot = world.get_snapshot()
                target_snapshot = snapshot.find(target.id)
                if target_snapshot is None:
                    raise RuntimeError("target missing from synchronized snapshot")
                target_transform = target_snapshot.get_transform()
                distance_m = row["initial_distance_m"]
                speed_mps = row["initial_speed_mps"]
                outcome = None
                for step in range(args.horizon):
                    pose = camera_transform(carla, target_transform, distance_m, NOMINAL_CAMERA)
                    sensor.set_transform(pose)
                    image = None
                    snapshot = None
                    measured_distance = None
                    # CARLA applies actor teleport commands at a tick boundary.
                    # Usually one tick is enough, but after replacing the target
                    # actor an RGB frame can occasionally retain the previous
                    # trajectory's camera transform.  Accept only a frame whose
                    # measured geometry matches this state; never label a stale
                    # frame with the new distance.
                    for _ in range(args.settle_ticks):
                        frame = world.tick(args.timeout)
                        image = matching_frame(image_queue, frame, args.timeout)
                        snapshot = world.get_snapshot()
                        if snapshot.frame != frame:
                            raise RuntimeError("another client advanced the world")
                        target_state = snapshot.find(target.id)
                        if target_state is None:
                            raise RuntimeError("target disappeared")
                        measured_camera = image.transform
                        actual_target = target_state.get_transform()
                        forward = actual_target.get_forward_vector()
                        delta = actual_target.location - measured_camera.location
                        measured_distance = delta.x * forward.x + delta.y * forward.y
                        if abs(measured_distance - distance_m) <= 0.01:
                            break
                        # Reissue the desired pose before the next bounded retry.
                        sensor.set_transform(pose)
                    else:
                        raise RuntimeError(
                            "camera distance mismatch after %d ticks requested=%.6f actual=%.6f" %
                            (args.settle_ticks, distance_m, measured_distance)
                        )
                    tensor = bgra_to_rgb_tensor(image.raw_data, image.width, image.height)
                    with torch.no_grad():
                        feature = backbone(tensor.unsqueeze(0).to(device)).float()
                        latent = projection(
                            (feature - feature_mean) / feature_std
                        ).cpu().numpy()
                    observation = np.concatenate(
                        [latent, np.asarray([[speed_mps]], dtype=np.float32)], axis=1
                    )
                    action = float(controller.predict(observation, deterministic=True)[0].reshape(-1)[0])
                    feature_ds[episode, step] = feature.cpu().numpy()[0]
                    distance_ds[episode, step] = distance_m
                    speed_ds[episode, step] = speed_mps
                    valid_ds[episode, step] = True
                    action_ds[episode, step] = action
                    saved_steps += 1
                    distance_m, speed_mps = advance_state(distance_m, speed_mps, action)
                    outcome = classify_terminal_outcome(distance_m, speed_mps)
                    if outcome is not None:
                        break
                if outcome is None:
                    outcome = TIMEOUT
                trajectory_outcomes.append(outcome)
                outcomes[outcome] += 1
                completed += 1
                stream.flush()
                print(
                    "calibration trajectory %d/%d steps=%d outcome=%s appearance=%s|%s" %
                    (completed, len(plan), step + 1, outcome, row["weather"], row["color"]),
                    flush=True,
                )
            strings = h5py.string_dtype(encoding="utf-8")
            stream.create_dataset(
                "outcome", data=np.asarray(trajectory_outcomes, dtype=object), dtype=strings
            )
            stream.attrs["complete"] = True
        partial.rename(final_h5)
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
        for restore in (
            lambda: world.set_weather(weather_before),
            lambda: world.apply_settings(settings_before),
        ):
            try:
                restore()
            except Exception as exc:
                cleanup_errors.append(repr(exc))
        summary = {
            "status": "complete" if failure is None and completed == len(plan) else "incomplete",
            "planned_trajectories": len(plan),
            "completed_trajectories": completed,
            "saved_valid_steps": saved_steps,
            "outcomes": dict(outcomes),
            "failure": failure,
            "cleanup_errors": cleanup_errors,
            "runtime_seconds": time.monotonic() - started,
            "h5": str(final_h5) if final_h5.is_file() else None,
            "h5_sha256": sha256(final_h5) if final_h5.is_file() else None,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/dino_expanded_v1/10_calibration_trajectories"))
    parser.add_argument("--representation", type=Path, default=Path("results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt"))
    parser.add_argument("--ppo-dir", type=Path, default=Path("results/dino_expanded_v1/09_R2_ppo"))
    parser.add_argument("--training-cache-manifest", type=Path, default=Path("results/dino_expanded_v1/02_dino_h5/cache_manifest.json"))
    parser.add_argument("--dino-repo", type=Path, default=Path("../external/dinov2"))
    parser.add_argument("--dino-weights", type=Path, default=Path("../external/dinov2_weights/dinov2_vits14_pretrain.pth"))
    parser.add_argument("--episodes", type=int, default=459)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--weathers", nargs="+", choices=sorted(WEATHERS), default=sorted(WEATHERS))
    parser.add_argument("--colors", nargs="+", default=list(COLORS))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--expected-map", required=False, default="")
    parser.add_argument("--spawn-index", type=int, default=4)
    parser.add_argument("--target-blueprint", default="vehicle.tesla.model3")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument(
        "--settle-ticks", type=int, default=10,
        help="Maximum ticks to wait for each requested camera pose; accepts the first geometrically matched frame.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        args.output_dir.exists() or args.episodes < 1 or args.horizon < 1
        or args.settle_ticks < 1 or min(args.width, args.height) < 1
        or not math.isfinite(args.timeout) or args.timeout <= 0
        or not 0 < args.fov < 180 or len(set(args.colors)) != len(args.colors)
    ):
        parser.error("use a new output directory and valid collection settings")
    for color in args.colors:
        try:
            values = [int(value) for value in color.split(",")]
            assert len(values) == 3 and all(0 <= value <= 255 for value in values)
        except (ValueError, AssertionError):
            parser.error("colors must be unique R,G,B integers in [0,255]")
    plan = episode_plan(args.episodes, args.seed, args.weathers, args.colors)
    report = {
        "protocol": "R2_independent_online_calibration_trajectories_v1",
        "episodes": args.episodes,
        "horizon": args.horizon,
        "maximum_frames": args.episodes * args.horizon,
        "seed": args.seed,
        "appearance_counts": dict(Counter(row["weather"] + "|" + row["color"] for row in plan)),
        "test_used": False,
        "output": str(args.output_dir),
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.dry_run:
        return
    if args.inspect:
        import carla
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        world = client.get_world()
        print("server=%s map=%s" % (client.get_server_version(), world.get_map().name))
        for index, transform in enumerate(world.get_map().get_spawn_points()):
            print(json.dumps({"spawn_index": index, "transform": transform_dict(transform)}))
        return
    if not args.expected_map:
        parser.error("--expected-map is required for collection; this script never changes maps")
    representation, controller_path, expected_dino_hash = preflight(args, parser)
    collect(args, plan, representation, controller_path, expected_dino_hash)


if __name__ == "__main__":
    main()
