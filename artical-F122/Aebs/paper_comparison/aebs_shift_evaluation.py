"""Registered matched-pixel AEBS photometric-shift evaluation, without training.

Seven fixed conditions are applied to the same common held-image bank.  The
experiment measures zero-shot behavior under synthetic distribution changes;
it is not new field imagery, a CP recalibration, or a safety certificate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import gaussian_filter
import scipy
import torch
import torch.nn.functional as F

from Aebs.dino_latent.backbone import FrozenDinoV2
from Aebs.experiments.matched_aebs_library import SCALE
from Aebs.vision_sls.model import ObservationProjection
from Combined_network.model import SubNet


BUDGETS = (400, 10000, 24000)
SEEDS = (27411, 27412, 27413)
CONDITIONS = (
    ("identity", "identity", 1.0),
    ("brightness_0p7", "brightness", .7),
    ("brightness_1p3", "brightness", 1.3),
    ("contrast_0p7", "contrast", .7),
    ("contrast_1p3", "contrast", 1.3),
    ("blur_0p5", "gaussian_blur", .5),
    ("blur_1p0", "gaussian_blur", 1.0),
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def transform(raw, kind, parameter):
    raw = np.asarray(raw)
    if raw.dtype != np.uint8 or raw.shape[-2:] != (32, 32):
        raise ValueError("expected original uint8 grayscale 32x32 pixels")
    if kind == "identity":
        return raw.copy()
    values = raw.astype(np.float64)
    if kind == "brightness":
        values *= parameter
    elif kind == "contrast":
        values = 127.5 + parameter * (values - 127.5)
    elif kind == "gaussian_blur":
        sigma = (0,) * (values.ndim - 2) + (parameter, parameter)
        values = gaussian_filter(values, sigma=sigma, mode="reflect", truncate=4.0)
    else:
        raise ValueError(kind)
    return np.rint(np.clip(values, 0, 255)).astype(np.uint8)


def old_heads(source):
    heads = []
    for n in BUDGETS:
        for seed in SEEDS:
            for mode in ("gray32_mlp", "gray32"):
                key = "%s_%d_%d" % (mode, n, seed)
                folder = source / "models" / key
                meta = read(folder / "fit.json")
                if digest(folder / "readout.pt") != meta["model"]:
                    raise ValueError("frozen legacy head mismatch: " + key)
                if meta["protocol_sha256"] != digest(source / "protocol.json"):
                    raise ValueError("legacy training protocol mismatch")
                heads.append(dict(key=key, mode=mode, budget=n, train_seed=seed,
                                  checkpoint=str(folder / "readout.pt"),
                                  checkpoint_sha256=meta["model"],
                                  fit_path=str(folder / "fit.json"),
                                  fit_sha256=digest(folder / "fit.json"),
                                  family="mlp" if mode == "gray32_mlp" else "dino_cls"))
    return heads


def register(a):
    a.output.mkdir(parents=True, exist_ok=False)
    libraries = read(a.library / "libraries.json")
    filenames = libraries["common"]
    if len(filenames) != 3229 or len(set(filenames)) != len(filenames):
        raise ValueError("expected exactly 3229 unique common held IDs")
    ids = np.load(a.source / "ids.npy")
    index = {name: i for i, name in enumerate(ids)}
    rows = np.array([index[name] for name in filenames], dtype=np.int64)
    master = read(a.budgets / "protocol.json")["24000"]
    labels = {name: (d, w, c) for name, d, w, c in
              zip(master["files"], master["distances"], master["weather"], master["color"])}
    distance, weather, color = map(np.array, zip(*[labels[name] for name in filenames]))
    raw = np.load(a.source / "raw_gray.npy", mmap_mode="r")[rows].reshape(-1, 32, 32)
    with np.load(a.library / "image_hashes.npz") as expected:
        hashes = dict(zip(expected["filename"], expected["gray_sha256"]))
    if any(hashlib.sha256(image.tobytes()).hexdigest() != hashes[name]
           for image, name in zip(raw, filenames)):
        raise ValueError("common pixels differ from audited source bytes")
    inputs = [a.source / name for name in ("protocol.json", "cache.json", "ids.npy", "raw_gray.npy", "gray32.npy")]
    inputs += [a.library / "libraries.json", a.library / "image_hashes.npz",
               a.budgets / "protocol.json", a.weights, Path(__file__),
               Path("Aebs/dino_latent/backbone.py"), Path("Aebs/vision_sls/model.py"),
               Path("Combined_network/model.py")]
    record = dict(
        experiment="AEBS frozen common-image zero-shot photometric shifts",
        source=str(a.source), library=str(a.library), budgets_path=str(a.budgets),
        repo=str(a.repo), weights=str(a.weights), image_count=3229,
        source_pixels="same audited uint8 grayscale32 input bytes for all methods",
        condition_order=[name for name, _, _ in CONDITIONS],
        conditions=[dict(name=name, operation=kind, parameter=value) for name, kind, value in CONDITIONS],
        transformation="float64; brightness multiplies intensity; contrast around127.5; Gaussian sigma in original32px coordinates, reflect boundary and truncate4; clip[0,255], np.rint(ties-to-even), uint8; identity unchanged",
        scope="Synthetic test distribution changes applied to existing held CARLA images. No new rendered camera frames, field images, CP guarantee or safety certificate.",
        labels="Physical distance labels unchanged by these photometric transforms; no geometric warp",
        population="Common3229 excludes training/development IDs and matching source gray pixels of all three budgets; held duplicates are retained as registered finite-library records, not claimed independent frames",
        old_heads=old_heads(a.source), distance_scale_m=SCALE,
        metrics=["mae_m", "rmse_m", "p95_absolute_m", "signed_bias_m", "max_absolute_m"],
        groups=["all", "each_weather", "each_weather_color"],
        percentile="numpy.percentile default linear interpolation",
        selection="All seven conditions and every frozen head reported. Test outcomes cannot select heads, hyperparameters, transforms or intensity levels.",
        new_heads="After the independent four-trial development selection has frozen all27 heads, bind checkpoint+fit+training protocol then evaluate unchanged conditions. No shift result is returned to training selection.",
        illustrative_example="Lexicographically first (weather,color) appearance group; smallest absolute physical distance difference from10m, filename breaks ties; all7 conditions; three new families at24k seed27411. Selection does not use predictions.",
        extraction=dict(device=a.device, batch=a.batch, memory_fraction=a.memory_fraction,
                        dtype="float32", tf32=False, cpu_preprocess=True,
                        cls="x_norm_clstoken", patches="x_norm_patchtokens[256,384] reshaped384x16x16; adaptive average pool2/4; row-major patches then384 channels",
                        forwards_per_condition=math.ceil(3229 / a.batch), expected_image_forwards=3229 * 7),
        cost_estimate=dict(new_images=0, transformed_inputs=3229 * 7,
                           cache_float32_bytes=3229 * 7 * (384 + 4 * 384 + 16 * 384) * 4,
                           transformed_uint8_bytes=3229 * 7 * 1024,
                           legacy_head_condition_pairs=18 * 7, new_head_condition_pairs=27 * 7),
        versions=dict(numpy=np.__version__, scipy=scipy.__version__, torch=str(torch.__version__)),
        bindings={str(path): digest(path) for path in inputs},
    )
    np.savez_compressed(a.output / "metadata.npz", filename=np.array(filenames), source_rows=rows,
                        distance_m=distance.astype(np.float64), weather=weather, color=color)
    record["metadata_sha256"] = digest(a.output / "metadata.npz")
    np.save(a.output / "ids.npy", np.array(filenames))
    record["ids_sha256"] = digest(a.output / "ids.npy")
    write(a.output / "protocol.json", record)
    print(json.dumps(dict(registered=str(a.output), cost=record["cost_estimate"])), flush=True)


def validate(output):
    protocol = read(output / "protocol.json")
    for path, expected in protocol["bindings"].items():
        if digest(path) != expected:
            raise ValueError("registered source changed: " + path)
    if digest(output / "metadata.npz") != protocol["metadata_sha256"]:
        raise ValueError("metadata changed")
    if digest(output / "ids.npy") != protocol["ids_sha256"]:
        raise ValueError("image IDs changed")
    for head in protocol["old_heads"]:
        if digest(head["checkpoint"]) != head["checkpoint_sha256"]:
            raise ValueError("old checkpoint changed")
        if digest(head["fit_path"]) != head["fit_sha256"]:
            raise ValueError("old selection record changed")
    return protocol


def prepare(a):
    protocol = validate(a.output)
    with np.load(a.output / "metadata.npz") as meta:
        rows = meta["source_rows"]
    raw = np.load(Path(protocol["source"]) / "raw_gray.npy", mmap_mode="r")[rows].reshape(-1, 32, 32)
    start = time.perf_counter()
    records = []
    for condition in protocol["conditions"]:
        folder = a.output / "conditions" / condition["name"]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "pixels.npy"
        if path.exists():
            raise FileExistsError(str(path))
        value = transform(raw, condition["operation"], condition["parameter"])
        np.save(path, value)
        records.append(dict(condition=condition["name"], path=str(path), sha256=digest(path),
                            shape=list(value.shape), dtype=str(value.dtype),
                            changed_pixels=int(np.count_nonzero(value != raw))))
    write(a.output / "pixels_manifest.json", dict(protocol_sha256=digest(a.output / "protocol.json"),
          conditions=records, wall_seconds=time.perf_counter() - start))
    with np.load(a.output / "metadata.npz") as meta:
        appearance = sorted(set(zip(meta["weather"].tolist(), meta["color"].tolist())))[0]
        candidates = np.flatnonzero((meta["weather"] == appearance[0]) & (meta["color"] == appearance[1]))
        chosen = min(candidates, key=lambda i: (abs(float(meta["distance_m"][i]) - 10.), str(meta["filename"][i])))
        sample = dict(row=int(chosen), filename=str(meta["filename"][chosen]), distance_m=float(meta["distance_m"][chosen]),
                      weather=str(appearance[0]), color=str(appearance[1]), conditions=protocol["condition_order"],
                      rule=protocol["illustrative_example"])
    np.savez_compressed(a.output / "example_images.npz", pixels=np.stack([
        np.load(a.output / "conditions" / name / "pixels.npy", mmap_mode="r")[chosen]
        for name in protocol["condition_order"]]), condition=np.array(protocol["condition_order"]))
    sample["image_file_sha256"] = digest(a.output / "example_images.npz")
    write(a.output / "example_images.json", sample)
    print("prepared_all_conditions", flush=True)


def extract(a):
    protocol = validate(a.output)
    pixels = read(a.output / "pixels_manifest.json")
    if pixels["protocol_sha256"] != digest(a.output / "protocol.json"):
        raise ValueError("pixel protocol mismatch")
    torch.set_num_threads(2)
    cfg = protocol["extraction"]
    device = cfg["device"]
    if device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(cfg["memory_fraction"], torch.device(device))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    encoder = FrozenDinoV2(protocol["repo"], protocol["weights"], device)
    with np.load(a.output / "metadata.npz") as meta:
        rows = meta["source_rows"]
    reference = np.load(Path(protocol["source"]) / "gray32.npy", mmap_mode="r")
    total_start = time.perf_counter()
    records = []
    with torch.inference_mode():
        for item in pixels["conditions"]:
            if digest(item["path"]) != item["sha256"]:
                raise ValueError("changed transformed pixels")
            folder = Path(item["path"]).parent
            if (folder / "cache.json").exists():
                saved = read(folder / "cache.json")
                if saved["protocol_sha256"] != digest(a.output / "protocol.json") or saved["input_sha256"] != item["sha256"]:
                    raise ValueError("cached features belong to different protocol or input")
                for filename, expected in saved["files"].items():
                    if digest(folder / filename) != expected:
                        raise ValueError("cache changed")
                records.append(saved)
                continue
            raw = np.load(item["path"], mmap_mode="r")
            cache = {"cls": np.lib.format.open_memmap(folder / "cls.npy", mode="w+", dtype="float32", shape=(len(raw), 384))}
            for g in (2, 4):
                cache["pool%d" % g] = np.lib.format.open_memmap(folder / ("pool%d.npy" % g), mode="w+", dtype="float32", shape=(len(raw), g * g, 384))
            start = time.perf_counter()
            max_delta = 0.
            for pos in range(0, len(raw), cfg["batch"]):
                x = torch.from_numpy(np.array(raw[pos:pos + cfg["batch"]])).float()[:, None].div_(255.)
                features = encoder.backbone.forward_features(encoder.preprocess(x).to(device))
                cls = features["x_norm_clstoken"].cpu().numpy()
                patch = features["x_norm_patchtokens"]
                if patch.shape[1:] != (256, 384):
                    raise ValueError("unexpected DINO patch shape")
                patch = patch.transpose(1, 2).reshape(-1, 384, 16, 16)
                cache["cls"][pos:pos + len(x)] = cls
                for g in (2, 4):
                    pooled = F.adaptive_avg_pool2d(patch, g).flatten(2).transpose(1, 2)
                    cache["pool%d" % g][pos:pos + len(x)] = pooled.cpu().numpy()
                if item["condition"] == "identity":
                    max_delta = max(max_delta, float(np.abs(cls - reference[rows[pos:pos + len(x)]]).max()))
            for value in cache.values():
                value.flush()
            if item["condition"] == "identity" and max_delta > 2e-4:
                raise ValueError("identity CLS differs from original frozen cache")
            record = dict(condition=item["condition"], input_sha256=item["sha256"],
                          protocol_sha256=digest(a.output / "protocol.json"),
                          files={key + ".npy": digest(folder / (key + ".npy")) for key in cache},
                          wall_seconds=time.perf_counter() - start, image_forwards=len(raw),
                          batches=math.ceil(len(raw) / cfg["batch"]),
                          identity_cls_max_abs_difference=max_delta if item["condition"] == "identity" else None)
            write(folder / "cache.json", record)
            records.append(record)
            print("shift_encoded", item["condition"], round(record["wall_seconds"], 3), flush=True)
    write(a.output / "cache_manifest.json", dict(protocol_sha256=digest(a.output / "protocol.json"),
          conditions=records, wall_seconds=time.perf_counter() - total_start,
          image_forwards=sum(row["image_forwards"] for row in records),
          gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(torch.device(device)) if device.startswith("cuda") else None))
    print("gpu_extraction_completed", flush=True)


def metrics(prediction, truth):
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    absolute = np.abs(error)
    return dict(n=len(error), mae_m=float(absolute.mean()), rmse_m=float(np.sqrt(np.mean(error**2))),
                p95_absolute_m=float(np.percentile(absolute, 95)), signed_bias_m=float(error.mean()),
                max_absolute_m=float(absolute.max()))


def register_new_heads(output, model_root):
    # Only frozen development selections are consumed, never a shift score.
    from Aebs.experiments.spatial_aebs import ARMS
    heads = []
    for n in BUDGETS:
        for seed in SEEDS:
            for mode in ARMS:
                key = "%s_%d_%d" % (mode, n, seed)
                folder = model_root / "models" / key
                fit, frozen = read(folder / "fit.json"), read(folder / "frozen.json")
                sha = digest(folder / "readout.pt")
                if fit["model"] != sha or frozen["model_sha256"] != sha:
                    raise ValueError("new checkpoint identity differs from freeze")
                if fit["frozen_sha256"] != digest(folder / "frozen.json"):
                    raise ValueError("new freeze record changed")
                if fit["test_metrics_accessed"] or frozen["test_metrics_accessed"]:
                    raise ValueError("model selection used test metrics")
                if fit["protocol_sha256"] != digest(model_root / "protocol.json"):
                    raise ValueError("new training protocol mismatch")
                heads.append(dict(key=key, mode=mode, budget=n, train_seed=seed,
                                  grid=fit["rank_selection"].get("grid"), checkpoint=str(folder / "readout.pt"),
                                  checkpoint_sha256=sha, fit_path=str(folder / "fit.json"),
                                  fit_sha256=digest(folder / "fit.json"), frozen_sha256=digest(folder / "frozen.json"),
                                  family={"gray32_mlp": "mlp", "gray32_cls": "dino_cls", "gray32_spatial": "dino_spatial"}[mode]))
    record = dict(training_protocol_sha256=digest(model_root / "protocol.json"),
                  architecture_source_sha256=digest(Path("Aebs/experiments/spatial_aebs.py")),
                  shift_protocol_sha256=digest(output / "protocol.json"), heads=heads,
                  selection="Frozen development-only winners; no shift-condition selection")
    path = output / "new_frozen_heads.json"
    if path.exists() and read(path) != record:
        raise ValueError("new frozen-head registration differs")
    if not path.exists():
        write(path, record)
    return heads


def evaluate(a, new=False):
    protocol = validate(a.output)
    cache = read(a.output / "cache_manifest.json")
    if cache["protocol_sha256"] != digest(a.output / "protocol.json"):
        raise ValueError("feature cache protocol mismatch")
    for item in cache["conditions"]:
        folder = a.output / "conditions" / item["condition"]
        for filename, expected in item["files"].items():
            if digest(folder / filename) != expected:
                raise ValueError("feature cache changed")
    group_name = "new27" if new else "legacy18"
    heads = register_new_heads(a.output, a.new_models) if new else protocol["old_heads"]
    out = a.output / group_name
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    with np.load(a.output / "metadata.npz") as meta:
        truth, weather, color = meta["distance_m"], meta["weather"], meta["color"]
    masks = [("all", "all", np.ones(len(truth), bool))]
    masks.extend(("weather", str(w), weather == w) for w in np.unique(weather))
    masks.extend(("weather_color", str(w) + " | " + str(c), (weather == w) & (color == c))
                 for w, c in sorted(set(zip(weather.tolist(), color.tolist()))))
    summary, predictions = [], []
    start = time.perf_counter()
    for head in heads:
        if new:
            from Aebs.experiments.spatial_aebs import make_model
            model = make_model(head["mode"], head.get("grid"))
        else:
            model = SubNet([1024, 256, 64, 1]) if head["mode"] == "gray32_mlp" else ObservationProjection()
        model.load_state_dict(torch.load(head["checkpoint"], map_location="cpu", weights_only=False))
        model.eval().requires_grad_(False)
        for condition in protocol["condition_order"]:
            folder = a.output / "conditions" / condition
            raw = head["family"] == "mlp"
            name = "pixels" if raw else ("pool%d" % head["grid"] if head["family"] == "dino_spatial" else "cls")
            values = np.load(folder / (name + ".npy"), mmap_mode="r")
            batches = []
            with torch.inference_mode():
                for pos in range(0, len(values), 256):
                    x = torch.from_numpy(np.array(values[pos:pos + 256])).float()
                    if raw:
                        x = x.reshape(len(x), -1).mul_(2 / 255.).sub_(1.)
                    batches.append(model(x).flatten().numpy())
            prediction = np.concatenate(batches)
            path = out / (head["key"] + "__" + condition + ".npy")
            np.save(path, prediction)
            predictions.append(dict(**head, condition=condition, prediction_path=str(path),
                                    prediction_sha256=digest(path), prediction_units="distance_over_scale",
                                    ids_path=str(a.output / "ids.npy"), ids_sha256=digest(a.output / "ids.npy"),
                                    model_path=head["checkpoint"],
                                    distance_scale_m=SCALE))
            for grouping, label, mask in masks:
                summary.append(dict(head=head["key"], family=head["family"], budget=head["budget"],
                                    train_seed=head["train_seed"], condition=condition,
                                    grouping=grouping, group=label, **metrics(prediction[mask].astype(np.float64) * SCALE, truth[mask])))
        print("shift_head_evaluated", group_name, head["key"], flush=True)
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    write(out / "metrics.json", summary)
    write(out / "prediction_manifest.json", dict(
        schema="aebs_fixed_photometric_prediction_manifest_v1", models=predictions,
        metadata_path=str(a.output / "metadata.npz"), metadata_sha256=digest(a.output / "metadata.npz"),
        protocol_sha256=digest(a.output / "protocol.json"), image_count=len(truth),
        condition_order=protocol["condition_order"], distance_scale_m=SCALE,
        input_pixels="shared shifted grayscale32 uint8 for every family",
        scope="Synthetic shifted saved-image replay support; no new CP guarantee or certificate transfer",
        paths_relative_to="artical-F122 working directory", wall_seconds=time.perf_counter() - start,
        scalar_predictions=len(heads) * len(CONDITIONS) * len(truth)))
    print("shift_evaluation_completed", group_name, round(time.perf_counter() - start, 3), flush=True)


def self_test():
    image = np.arange(1024, dtype=np.uint16).reshape(1, 32, 32).astype(np.uint8)
    assert np.array_equal(transform(image, "identity", 1), image)
    for _, kind, parameter in CONDITIONS:
        result = transform(image, kind, parameter)
        assert result.dtype == np.uint8 and result.shape == image.shape
        assert np.array_equal(result, transform(image, kind, parameter))
    constant = np.full((2, 32, 32), 127, dtype=np.uint8)
    assert np.array_equal(transform(constant, "gaussian_blur", 1), constant)
    bright = transform(np.full((1, 32, 32), 255, np.uint8), "brightness", 1.3)
    assert np.all(bright == 255)
    print("transform_self_test_passed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["register", "prepare", "extract", "evaluate-old", "evaluate-new", "self-test"])
    p.add_argument("output", type=Path)
    p.add_argument("--source", type=Path, default=Path("results/aebs_factorial_20261117"))
    p.add_argument("--budgets", type=Path, default=Path("results/aebs_budget_scaling_20261110"))
    p.add_argument("--library", type=Path, default=Path("results/aebs_matched_replay_20261117/library"))
    p.add_argument("--new-models", type=Path, default=Path("results/aebs_spatial_readout_20261118_v1"))
    p.add_argument("--repo", type=Path, default=Path("../external/dinov2"))
    p.add_argument("--weights", type=Path, default=Path("../external/dinov2_weights/dinov2_vits14_pretrain.pth"))
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--memory-fraction", type=float, default=.035)
    a = p.parse_args()
    if a.phase == "self-test":
        self_test()
    elif a.phase == "evaluate-old":
        evaluate(a)
    elif a.phase == "evaluate-new":
        evaluate(a, new=True)
    else:
        {"register": register, "prepare": prepare, "extract": extract}[a.phase](a)


if __name__ == "__main__":
    main()
