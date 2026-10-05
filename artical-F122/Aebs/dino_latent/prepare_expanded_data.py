"""Audit expanded SPVC PNGs and freeze distance-group splits. No model runs.

Audit code uses Pillow and stdlib; package imports also require existing Torch.
Never edits source data or loads DINO weights.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def distance_group(value):
    d = float(value)
    if not math.isfinite(d) or not 5 <= d <= 16:
        raise ValueError("Distance must be finite and in [5,16]")
    # Quantize at 1 nm in metre units to neutralize arithmetic at bin edges.
    rounded = Decimal(str(d)).quantize(Decimal("0.000000001"), rounding=ROUND_HALF_UP)
    return min(43, int((rounded - Decimal(5)) // Decimal("0.25")))


def split_groups(seed=7):
    rng = random.Random(seed)
    groups = {"train": [], "validation": [], "test": []}
    for start, end, ntrain, nval in [(0, 10, 6, 2), (10, 20, 6, 2),
                                     (20, 30, 6, 2), (30, 44, 8, 3)]:
        ids = list(range(start, end))
        rng.shuffle(ids)
        groups["train"].extend(ids[:ntrain])
        groups["validation"].extend(ids[ntrain:ntrain+nval])
        groups["test"].extend(ids[ntrain+nval:])
    return {key: sorted(ids) for key, ids in groups.items()}


def image_metadata(path):
    from PIL import Image, ImageStat
    with Image.open(path) as image:
        if image.format != "PNG" or image.size != (640, 640):
            raise ValueError("Expected 640x640 PNG")
        image.load()  # Full decode, not just header verification.
        rgb = image.convert("RGB")
        gray = rgb.convert("L")
        stats = ImageStat.Stat(gray)
        return dict(sha256=sha256(path), pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                    gray_mean=stats.mean[0], gray_std=stats.stddev[0],
                    near_constant=stats.stddev[0] < 1.0)


def validate_rows(root, config, summary, rows, csv_rows):
    if config.get("protocol") != "spvc_carla_fullscreen_expanded_v1":
        raise ValueError("Unexpected capture protocol")
    n = config["points_per_segment"]
    if not isinstance(n, int) or n < 2:
        raise ValueError("Invalid points_per_segment")
    weather, colors = list(config["weather_profiles"]), config["colors"]
    expected = 2*n*len(weather)*len(colors)
    if (summary.get("status") != "complete" or summary.get("error") is not None or
            summary.get("saved_samples") != expected or summary.get("planned_samples") != expected or
            config.get("samples") != expected or len(rows) != expected or len(csv_rows) != expected):
        raise ValueError("Summary/config/manifest/CSV counts or completion disagree")
    csv_map = {r["filename"]: float(r["distance_m"]) for r in csv_rows}
    if len(csv_map) != expected:
        raise ValueError("Duplicate CSV filenames")
    names, keys, group_values = set(), set(), {}
    for row in rows:
        name = row["filename"]
        if Path(name).name != name or not name.endswith(".png") or name in names:
            raise ValueError("Unsafe or duplicate filename: %s" % name)
        path = root / name
        if path.resolve().parent != root.resolve() or not path.is_file():
            raise ValueError("Missing or out-of-root image: %s" % name)
        names.add(name)
        i, d = row["distance_index"], float(row["distance_m"])
        distance_group(d)
        if not isinstance(i, int) or not 0 <= i < 2*n:
            raise ValueError("Invalid distance index")
        target = 5+5*i/(n-1) if i < n else 10+6*(i-n)/(n-1)
        if (abs(d-target) > 1e-8 or name not in csv_map or
                not math.isfinite(csv_map[name]) or abs(csv_map[name]-d) > 1e-8):
            raise ValueError("Distance sequence or CSV mismatch: %s" % name)
        if row["weather"] not in weather or row["color"] not in colors:
            raise ValueError("Unknown appearance")
        key = (row["weather"], row["color"], i)
        if key in keys:
            raise ValueError("Duplicate appearance/distance index")
        keys.add(key)
        previous = group_values.setdefault(row["group_id"], d)
        if abs(previous-d) > 1e-8:
            raise ValueError("One exact group contains multiple distances")
    if names != set(csv_map) or names != {p.name for p in root.glob("*.png")}:
        raise ValueError("PNG inventory and labels disagree")
    if {distance_group(r["distance_m"]) for r in rows} != set(range(44)):
        raise ValueError("Not all 44 distance intervals populated")


def run(root, output, seed):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status="running", data_root=str(root), seed=seed, training_run=False,
                  model_test_evaluated=False, visual_review="pending")
    try:
        config = json.loads((root / "config.json").read_text())
        summary = json.loads((root / "summary.json").read_text())
        rows = [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines() if line.strip()]
        with (root / "labels.csv").open(newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        validate_rows(root, config, summary, rows, csv_rows)
        groups = split_groups(seed)
        membership = {g: key for key, ids in groups.items() for g in ids}
        counts, combinations, duplicates = Counter(), Counter(), defaultdict(list)
        issues, records, near_constant = [], [], []
        source_hashes = {f: sha256(root/f) for f in ["config.json", "summary.json", "labels.csv", "samples.jsonl"]}
        for i, row in enumerate(sorted(rows, key=lambda r: r["filename"])):
            try:
                metadata = image_metadata(root/row["filename"])
            except Exception as exc:
                issues.append(dict(filename=row["filename"], error=repr(exc)))
                continue
            g = distance_group(row["distance_m"])
            split = membership[g]
            entry = {**row, **metadata, "distance_bin_025m": g, "split": split}
            records.append(entry)
            counts[split] += 1
            combinations["%s|%s|%s" % (split, row["weather"], row["color"])] += 1
            duplicates[metadata["pixel_sha256"]].append(dict(filename=row["filename"], split=split,
                                                            distance_m=row["distance_m"]))
            if metadata["near_constant"]:
                near_constant.append(row["filename"])
            if (i+1) % 1000 == 0 or i+1 == len(rows):
                print("audit images %d/%d" % (i+1, len(rows)), flush=True)
        duplicate_groups = [v for v in duplicates.values() if len(v) > 1]
        cross_split = [v for v in duplicate_groups if len({r["split"] for r in v}) > 1]
        (output/"image_issues.json").write_text(json.dumps(dict(decode_errors=issues,
            near_constant=near_constant, duplicate_pixel_groups=duplicate_groups,
            cross_split_duplicates=cross_split), indent=2))
        report.update(samples=len(rows), decoded=len(records), decode_errors=len(issues),
                      near_constant=len(near_constant), duplicate_pixel_groups=len(duplicate_groups),
                      cross_split_duplicate_groups=len(cross_split), split_counts=dict(counts),
                      split_group_counts={k: len(v) for k,v in groups.items()},
                      appearance_counts=dict(combinations), source_hashes=source_hashes)
        if issues or cross_split or near_constant:
            report["status"] = "needs_data_review"
            print("Data anomalies found: inspect image_issues.json; no split manifest released.", flush=True)
            return 2
        digest = hashlib.sha256()
        for r in records:
            digest.update((r["filename"]+":"+r["sha256"]+"\n").encode())
        manifest = dict(schema="spvc_expanded_split_v1", data_root=str(root), seed=seed,
                        algorithm="python_random_shuffle_stratified_10_10_10_14_v1",
                        source_hashes=source_hashes, image_inventory_sha256=digest.hexdigest(),
                        script_sha256=sha256(Path(__file__)), groups=groups, records=records,
                        visual_review="pending", test_used_for_model_selection=False)
        (output/"split_manifest.json").write_text(json.dumps(manifest, indent=2))
        for split in groups:
            with (output/(split+".csv")).open("w", newline="") as stream:
                fields = ["filename", "distance_m", "group_id", "distance_bin_025m", "weather", "color", "split"]
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(r for r in records if r["split"] == split)
        qa = []
        for weather in config["weather_profiles"]:
            for color in config["colors"]:
                available = [r for r in records if r["weather"] == weather and r["color"] == color]
                for d in [5, 6, 8, 10, 13, 16]:
                    selected = min(available, key=lambda r: abs(r["distance_m"]-d))
                    qa.append({k: selected[k] for k in ["filename", "distance_m", "weather", "color", "split"]})
        (output/"visual_review_samples.json").write_text(json.dumps(qa, indent=2))
        report["status"] = "audit_pass_split_ready_pending_visual_review"
        return 0
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        report["runtime_seconds"] = time.monotonic()-started
        (output/"data_audit.json").write_text(json.dumps(report, indent=2))
        print("[Expanded data audit and grouped split — no training]")
        for key in ["status", "samples", "decoded", "decode_errors", "near_constant", "cross_split_duplicate_groups",
                    "split_group_counts", "split_counts", "error"]:
            if key in report:
                print("%s: %s" % (key, report[key]))
        print("report: %s" % (output/"data_audit.json"))
        print("Latent errors not computed; no DINO/PPO/SBC training or model testing.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("../carla_data_spvc_expanded/full01"))
    p.add_argument("--output-dir", type=Path, default=Path("results/dino_expanded_v1/01_data_prepare"))
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error("Output already exists; use a new directory, do not overwrite a frozen split")
    raise SystemExit(run(args.data_root.resolve(), args.output_dir, args.seed))


if __name__ == "__main__":
    main()
