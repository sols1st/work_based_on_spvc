"""Expand user spvc_carla.py data without changing its full-screen acquisition.

Run on the CARLA graphical desktop, not a headless SSH session. The explicit
--reload-original-map flag authorizes the original load_world operation.
"""
import argparse
import csv
import json
import time
from pathlib import Path

MAP_INDEX = 3
SPAWN_INDEX = 4
BASE_WEATHER = dict(cloudiness=0.0, precipitation=0.0, precipitation_deposits=0.0,
                    wind_intensity=0.0, fog_density=0.0, wetness=0.0,
                    sun_altitude_angle=60.0, sun_azimuth_angle=90.0)
WEATHERS = {"original": {}, "overcast": dict(cloudiness=90),
            "low_sun": dict(sun_altitude_angle=12),
            "rain": dict(cloudiness=90, precipitation=60, precipitation_deposits=60, wetness=70)}


def distances(n):
    # Preserve the two inclusive linspace segments, including repeated 10 m.
    if n < 2:
        raise ValueError("points-per-segment must be >= 2")
    return [5+5*i/(n-1) for i in range(n)] + [10+6*i/(n-1) for i in range(n)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("Aebs/carla_data_spvc_expanded/run01"))
    p.add_argument("--points-per-segment", type=int, default=1000,
                   help="200 reproduces original 400 distance entries; default 1000 gives 2000")
    p.add_argument("--weathers", nargs="+", choices=list(WEATHERS), default=["original"])
    p.add_argument("--colors", nargs="+", default=["255,0,0"])
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--reload-original-map", action="store_true",
                   help="Authorize original client.load_world(available_maps[3]); resets running world")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    values = distances(args.points_per_segment)
    if len(set(args.weathers)) != len(args.weathers) or len(set(args.colors)) != len(args.colors):
        p.error("Repeated appearance profiles would produce duplicate data")
    for color in args.colors:
        try:
            rgb = [int(c) for c in color.split(",")]
            assert len(rgb) == 3 and all(0 <= c <= 255 for c in rgb)
        except (ValueError, AssertionError):
            p.error("Colors must be R,G,B integers in [0,255]")
    config = dict(protocol="spvc_carla_fullscreen_expanded_v1",
                  samples=len(values)*len(args.weathers)*len(args.colors),
                  points_per_segment=args.points_per_segment,
                  screenshot_region="mss.monitors[1] full screen", resize=[640, 640],
                  map_index=MAP_INDEX, spawn_index=SPAWN_INDEX,
                  height_offset_m=2.0, wait_after_move_seconds=0.3,
                  weather_profiles={w: {**BASE_WEATHER, **WEATHERS[w]} for w in args.weathers},
                  colors=args.colors, map_selection="available_maps[3] (original script)",
                  note="Desktop viewport/window/OS scaling must match original. Not frame-synchronized.")
    print(json.dumps(config, indent=2), flush=True)
    if args.dry_run:
        return
    if not args.reload_original_map:
        p.error("Original script reloads the map. Use --reload-original-map on a dedicated CARLA instance")
    import carla
    import cv2
    import mss
    import numpy as np

    values = np.concatenate((np.linspace(5.0, 10.0, args.points_per_segment),
                             np.linspace(10.0, 16.0, args.points_per_segment)))

    # Check GUI screenshot access before any world reset.
    screen = mss.mss()
    monitor = screen.monitors[1]
    screenshot_region = {key: monitor[key] for key in ("top", "left", "width", "height")}
    screen.grab(screenshot_region)
    config["screenshot_region"] = screenshot_region
    print("Full primary monitor capture: %s. Keep CARLA full screen." % screenshot_region, flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    vehicle = None
    saved, error, cleanup_errors = 0, None, []
    started = time.monotonic()
    try:
        maps = client.get_available_maps()
        world = client.load_world(maps[MAP_INDEX])
        if world.get_settings().synchronous_mode:
            raise RuntimeError("Original screenshot protocol requires an asynchronously ticking world")
        config.update(actual_map=world.get_map().name, server_version=client.get_server_version())
        spawn = world.get_map().get_spawn_points()[SPAWN_INDEX]
        location, rotation = spawn.location, spawn.rotation
        config["target_location"] = dict(x=location.x, y=location.y, z=location.z)
        config["target_rotation"] = dict(pitch=rotation.pitch, yaw=rotation.yaw, roll=rotation.roll)
        (args.output / "config.json").write_text(json.dumps(config, indent=2))
        spectator = world.get_spectator()
        with (args.output / "labels.csv").open("w", newline="") as labels, \
                (args.output / "samples.jsonl").open("w") as manifest:
            writer = csv.writer(labels)
            writer.writerow(["filename", "distance_m"])
            for color in args.colors:
                world.set_weather(carla.WeatherParameters(**BASE_WEATHER))
                bp = world.get_blueprint_library().find("vehicle.tesla.model3")
                bp.set_attribute("color", color)
                vehicle = world.try_spawn_actor(bp, carla.Transform(location, rotation))
                if vehicle is None:
                    raise RuntimeError("Original target spawn failed; do not silently substitute a different scene")
                spectator.set_transform(carla.Transform(location+carla.Location(z=2.0), rotation))
                print("8 seconds: put CARLA FULL SCREEN (F11), as in spvc_carla.py", flush=True)
                time.sleep(8)
                for weather_index, weather in enumerate(args.weathers):
                    world.set_weather(carla.WeatherParameters(**config["weather_profiles"][weather]))
                    if weather_index > 0 or weather != "original":
                        time.sleep(8)  # Let an explicitly requested new appearance settle.
                    for index, distance in enumerate(values):
                        tf = vehicle.get_transform()
                        forward = tf.get_forward_vector()
                        backward = carla.Location(-forward.x, -forward.y, 0)
                        pose = carla.Transform(tf.location+backward*distance+carla.Location(z=2.0), tf.rotation)
                        spectator.set_transform(pose)
                        time.sleep(0.3)
                        img = np.array(screen.grab(screenshot_region))
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                        img = cv2.resize(img, (640, 640))
                        filename = "%06d.png" % saved
                        if not cv2.imwrite(str(args.output / filename), img):
                            raise RuntimeError("Image write failed")
                        writer.writerow([filename, distance])
                        manifest.write(json.dumps(dict(filename=filename, distance_m=distance,
                            group_id="distance_%.9f" % distance, distance_index=index,
                            distance_bin_025m=int((float(distance)-5.0)/0.25+1e-8),
                            weather=weather, color=color, timestamp=time.time(),
                            target_location=dict(x=tf.location.x, y=tf.location.y, z=tf.location.z)))+"\n")
                        labels.flush()
                        manifest.flush()
                        saved += 1
                        if saved == 1 or saved % 100 == 0:
                            print("captured %d/%d" % (saved, config["samples"]), flush=True)
                vehicle.destroy()
                vehicle = None
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        if vehicle is not None:
            try:
                vehicle.destroy()
            except Exception as exc:
                cleanup_errors.append(repr(exc))
        screen.close()
        summary = dict(status="complete" if error is None and saved == config["samples"] else "incomplete",
                       saved_samples=saved, planned_samples=config["samples"], error=error,
                       cleanup_errors=cleanup_errors, runtime_seconds=time.monotonic()-started)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
