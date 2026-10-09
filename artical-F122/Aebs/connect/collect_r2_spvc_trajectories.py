"""Collect R2 trajectories with the original SPVC full-screen protocol.

This collector deliberately preserves the image acquisition used by
``spvc_carla.py`` and ``collect_spvc_expanded.py``: the original map selected
by ``available_maps[3]``, spawn point 4, a spectator placed behind the target,
the primary monitor captured with mss, and an OpenCV resize to 640x640.

The default role is ``development``.  Development output is intentionally
given a different H5 schema and cannot be consumed by the formal transition
contract calibrator.  Use ``--role calibration`` only after the smoke gate is
passed; that role requires at least 459 fresh trajectories.
"""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
from stable_baselines3 import PPO

from Aebs.connect.collect_r2_calibration_trajectories import (
    COLORS,
    advance_state,
    episode_plan,
    preflight,
)
from Aebs.connect.collect_spvc_expanded import (
    BASE_WEATHER,
    MAP_INDEX,
    SPAWN_INDEX,
    WEATHERS,
)
from Aebs.dino_latent.backbone import FrozenDinoV2
from Aebs.dino_latent.cache_expanded_features import repository_hash
from Aebs.dino_latent.calibrate_transition_contract import SCHEMA as CALIBRATION_SCHEMA
from Aebs.dino_latent.models import SafetyProjection
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.system.outcomes import TIMEOUT, classify_terminal_outcome


DEVELOPMENT_SCHEMA = "spvc_r2_fullscreen_development_trajectories_v1"
CAPTURE_PROTOCOL = "original_spvc_primary_monitor_fullscreen_v1"


def screenshot_bgra_to_model_inputs(raw_bgra, output_width=640, output_height=640):
    """Apply the original OpenCV screenshot pipeline and return BGR plus RGB tensor."""
    import cv2

    value = np.asarray(raw_bgra)
    if value.ndim != 3 or value.shape[2] != 4:
        raise ValueError("mss screenshot must have shape [H,W,4]")
    bgr = cv2.cvtColor(value, cv2.COLOR_BGRA2BGR)
    bgr = cv2.resize(bgr, (int(output_width), int(output_height)))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255.0)
    return bgr, tensor


def initialize_h5(stream, plan, horizon, provenance, role):
    episodes = len(plan)
    schema = CALIBRATION_SCHEMA if role == "calibration" else DEVELOPMENT_SCHEMA
    stream.attrs.update({
        "schema": schema,
        "capture_protocol": CAPTURE_PROTOCOL,
        "complete": False,
        "split_role": role,
        "independent_from_training": role == "calibration",
        "used_for_model_selection": role != "calibration",
        "horizon": int(horizon),
        "dino_weights_sha256": provenance["dino_weights_sha256"],
        "provenance_json": json.dumps(provenance),
    })
    strings = h5py.string_dtype(encoding="utf-8")
    for key in ("trajectory_id", "weather", "color"):
        stream.create_dataset(
            key,
            data=np.asarray([row[key] for row in plan], dtype=object),
            dtype=strings,
        )
    stream.create_dataset(
        "initial_distance_m", data=[row["initial_distance_m"] for row in plan]
    )
    stream.create_dataset(
        "initial_speed_mps", data=[row["initial_speed_mps"] for row in plan]
    )
    features = stream.create_dataset(
        "dino_features",
        shape=(episodes, horizon, 384),
        dtype="float32",
        chunks=(1, min(horizon, 64), 384),
        compression="lzf",
    )
    distance = stream.create_dataset(
        "distance_m", shape=(episodes, horizon), dtype="float32"
    )
    speed = stream.create_dataset(
        "speed_mps", shape=(episodes, horizon), dtype="float32"
    )
    valid = stream.create_dataset(
        "valid", shape=(episodes, horizon), dtype="bool"
    )
    action = stream.create_dataset(
        "image_action", shape=(episodes, horizon), dtype="float32"
    )
    valid[:] = False
    return features, distance, speed, valid, action


def collect(args, plan, representation, controller_path, expected_dino_hash):
    import carla
    import cv2
    import mss

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
    if any(parameter.requires_grad for model in (backbone, projection) for parameter in model.parameters()):
        raise RuntimeError("DINO/projection must stay frozen")

    # Fail before resetting CARLA if the graphical desktop cannot be captured.
    screen = mss.mss()
    if len(screen.monitors) < 2:
        screen.close()
        raise RuntimeError("mss primary monitor is unavailable; run in the CARLA graphical session")
    monitor = screen.monitors[1]
    screenshot_region = {
        key: int(monitor[key]) for key in ("top", "left", "width", "height")
    }
    screen.grab(screenshot_region)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    maps = client.get_available_maps()
    if MAP_INDEX >= len(maps):
        screen.close()
        raise RuntimeError("available_maps[%d] does not exist" % MAP_INDEX)
    selected_map = maps[MAP_INDEX]
    world = client.load_world(selected_map)
    actual_map = world.get_map().name.split("/")[-1]
    if actual_map != args.expected_map:
        screen.close()
        raise RuntimeError(
            "original available_maps[%d] resolved to %s, expected %s" %
            (MAP_INDEX, actual_map, args.expected_map)
        )
    if world.get_settings().synchronous_mode:
        screen.close()
        raise RuntimeError("original SPVC full-screen protocol requires asynchronous CARLA")
    spawns = world.get_map().get_spawn_points()
    if SPAWN_INDEX >= len(spawns):
        screen.close()
        raise RuntimeError("original spawn point %d is unavailable" % SPAWN_INDEX)

    provenance = {
        "collector": "R2_original_SPVC_fullscreen_trajectory_collector_v1",
        "registered_before_collection": True,
        "role": args.role,
        "seed": args.seed,
        "episodes": len(plan),
        "horizon": args.horizon,
        "initial_distribution": "distance uniform[15,16], speed uniform[2.5,3]",
        "appearance_distribution": "weather and color sampled independently and uniformly per trajectory",
        "weather_profiles": {
            name: {**BASE_WEATHER, **WEATHERS[name]} for name in args.weathers
        },
        "colors": list(args.colors),
        "capture_protocol": CAPTURE_PROTOCOL,
        "screenshot_region": screenshot_region,
        "resize": [args.width, args.height],
        "map_selection": "available_maps[%d]" % MAP_INDEX,
        "map": world.get_map().name,
        "spawn_index": SPAWN_INDEX,
        "spectator_height_m": args.height_offset_m,
        "wait_after_teleport_seconds": args.wait_after_teleport,
        "appearance_settle_seconds": args.appearance_settle_seconds,
        "deployment_rollout": "image latent PPO; analytic AEBS state; CARLA renders only",
        "representation_sha256": sha256(args.representation),
        "controller_sha256": sha256(controller_path),
        "dino_weights_sha256": expected_dino_hash,
        "dino_source_sha256": repository_hash(args.dino_repo),
        "training_cache_manifest_sha256": sha256(args.training_cache_manifest),
        "test_used": False,
        "model_updates": False,
        "device": device_name,
        "server_version": client.get_server_version(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "config.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8"
    )
    preview_dir = args.output_dir / "previews"
    if args.save_previews:
        preview_dir.mkdir()

    weather_before = world.get_weather()
    spectator = world.get_spectator()
    target = None
    saved_steps = completed = 0
    outcomes = Counter()
    failure = None
    started = time.monotonic()
    partial = args.output_dir / "trajectories.partial.h5"
    final_h5 = args.output_dir / "trajectories.h5"
    previous_appearance = None
    current_color = None
    try:
        print(
            "Primary monitor %dx%d. Put CARLA FULL SCREEN (F11); starting in %.1f s." %
            (screenshot_region["width"], screenshot_region["height"], args.initial_wait_seconds),
            flush=True,
        )
        time.sleep(args.initial_wait_seconds)
        trajectory_outcomes = []
        with h5py.File(partial, "w") as stream:
            feature_ds, distance_ds, speed_ds, valid_ds, action_ds = initialize_h5(
                stream, plan, args.horizon, provenance, args.role
            )
            for episode, row in enumerate(plan):
                world.set_weather(carla.WeatherParameters(
                    **{**BASE_WEATHER, **WEATHERS[row["weather"]]}
                ))
                # As in the expanded original collector, keep one vehicle for
                # repeated states and replace it only when its color changes.
                if target is None or current_color != row["color"]:
                    if target is not None:
                        target.destroy()
                        target = None
                    vehicle_bp = world.get_blueprint_library().find(args.target_blueprint)
                    vehicle_bp.set_attribute("color", row["color"])
                    target = world.try_spawn_actor(vehicle_bp, spawns[SPAWN_INDEX])
                    if target is None:
                        raise RuntimeError("original SPVC target spawn failed")
                    current_color = row["color"]
                appearance = (row["weather"], row["color"])
                if appearance != previous_appearance:
                    time.sleep(args.appearance_settle_seconds)
                    previous_appearance = appearance
                distance_m = row["initial_distance_m"]
                speed_mps = row["initial_speed_mps"]
                outcome = None
                for step in range(args.horizon):
                    target_transform = target.get_transform()
                    forward = target_transform.get_forward_vector()
                    backward = carla.Location(-forward.x, -forward.y, 0.0)
                    location = (
                        target_transform.location
                        + backward * distance_m
                        + carla.Location(z=args.height_offset_m)
                    )
                    spectator.set_transform(carla.Transform(location, target_transform.rotation))
                    time.sleep(args.wait_after_teleport)
                    raw = np.asarray(screen.grab(screenshot_region))
                    bgr, tensor = screenshot_bgra_to_model_inputs(
                        raw, args.width, args.height
                    )
                    if args.save_previews and step == 0:
                        preview = preview_dir / ("%03d_%s_%s.png" % (
                            episode, row["weather"], row["color"].replace(",", "-"),
                        ))
                        if not cv2.imwrite(str(preview), bgr):
                            raise RuntimeError("preview image write failed")
                    with torch.no_grad():
                        feature = backbone(tensor.unsqueeze(0).to(device)).float()
                        latent = projection(
                            (feature - feature_mean) / feature_std
                        ).cpu().numpy()
                    observation = np.concatenate(
                        [latent, np.asarray([[speed_mps]], dtype=np.float32)], axis=1
                    )
                    action = float(
                        controller.predict(observation, deterministic=True)[0].reshape(-1)[0]
                    )
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
                    "SPVC trajectory %d/%d steps=%d outcome=%s appearance=%s|%s" %
                    (completed, len(plan), step + 1, outcome, row["weather"], row["color"]),
                    flush=True,
                )
            strings = h5py.string_dtype(encoding="utf-8")
            stream.create_dataset(
                "outcome",
                data=np.asarray(trajectory_outcomes, dtype=object),
                dtype=strings,
            )
            stream.attrs["complete"] = True
        partial.rename(final_h5)
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        cleanup_errors = []
        if target is not None:
            try:
                target.destroy()
            except Exception as exc:
                cleanup_errors.append(repr(exc))
        try:
            world.set_weather(weather_before)
        except Exception as exc:
            cleanup_errors.append(repr(exc))
        try:
            screen.close()
        except Exception as exc:
            cleanup_errors.append(repr(exc))
        summary = {
            "status": "complete" if failure is None and completed == len(plan) else "incomplete",
            "role": args.role,
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
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/dino_expanded_v1/12_spvc_online_smoke_v1"),
    )
    parser.add_argument(
        "--representation", type=Path,
        default=Path("results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt"),
    )
    parser.add_argument(
        "--ppo-dir", type=Path,
        default=Path("results/dino_expanded_v1/09_R2_ppo"),
    )
    parser.add_argument(
        "--training-cache-manifest", type=Path,
        default=Path("results/dino_expanded_v1/02_dino_h5/cache_manifest.json"),
    )
    parser.add_argument("--dino-repo", type=Path, default=Path("../external/dinov2"))
    parser.add_argument(
        "--dino-weights", type=Path,
        default=Path("../external/dinov2_weights/dinov2_vits14_pretrain.pth"),
    )
    parser.add_argument("--role", choices=("development", "calibration"), default="development")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument("--weathers", nargs="+", choices=sorted(WEATHERS), default=sorted(WEATHERS))
    parser.add_argument("--colors", nargs="+", default=list(COLORS))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--expected-map", default="Town01_Opt")
    parser.add_argument("--target-blueprint", default="vehicle.tesla.model3")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--height-offset-m", type=float, default=2.0)
    parser.add_argument("--wait-after-teleport", type=float, default=0.3)
    parser.add_argument("--initial-wait-seconds", type=float, default=8.0)
    parser.add_argument("--appearance-settle-seconds", type=float, default=8.0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--save-previews", dest="save_previews", action="store_true")
    parser.add_argument("--no-save-previews", dest="save_previews", action="store_false")
    parser.set_defaults(save_previews=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        args.output_dir.exists() or args.episodes < 1 or args.horizon < 1
        or min(args.width, args.height) < 1
        or not math.isfinite(args.timeout) or args.timeout <= 0
        or min(args.height_offset_m, args.wait_after_teleport) < 0
        or min(args.initial_wait_seconds, args.appearance_settle_seconds) < 0
        or len(set(args.colors)) != len(args.colors)
    ):
        parser.error("use a new output directory and valid settings")
    if args.role == "calibration" and args.episodes < 459:
        parser.error("formal calibration requires at least 459 fresh trajectories")
    for color in args.colors:
        try:
            values = [int(value) for value in color.split(",")]
            assert len(values) == 3 and all(0 <= value <= 255 for value in values)
        except (ValueError, AssertionError):
            parser.error("colors must be unique R,G,B integers in [0,255]")
    plan = episode_plan(args.episodes, args.seed, args.weathers, args.colors)
    print(json.dumps({
        "protocol": CAPTURE_PROTOCOL,
        "role": args.role,
        "episodes": args.episodes,
        "horizon": args.horizon,
        "seed": args.seed,
        "map_selection": "available_maps[%d] expected %s" % (MAP_INDEX, args.expected_map),
        "spawn_index": SPAWN_INDEX,
        "appearance_counts": dict(Counter(
            row["weather"] + "|" + row["color"] for row in plan
        )),
        "output": str(args.output_dir),
        "test_used": False,
    }, indent=2), flush=True)
    if args.dry_run:
        return
    representation, controller_path, expected_dino_hash = preflight(args, parser)
    collect(args, plan, representation, controller_path, expected_dino_hash)


if __name__ == "__main__":
    main()
