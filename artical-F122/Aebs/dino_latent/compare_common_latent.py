"""Frozen common-semantic-space q-vs-cGAN comparison. No training/PPO/SBC/test.

The evaluator is the existing R2 DINO+projection, which q was trained to match.
This is TASK-SPACE consistency, not an independent perceptual or safety metric.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import h5py
import numpy as np
from PIL import Image
import torch

from cGAN.taxi_models_and_data import AebsMLPGenerator
from Aebs.dino_latent.backbone import FrozenDinoV2
from Aebs.dino_latent.cache_expanded_features import repository_hash
from Aebs.dino_latent.models import SafetyProjection, PhysicalToLatent
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.proxy_comparison_math import legacy_pixels
from Aebs.dino_latent.common_latent_metrics import generator_to_pixels, rms_gap, summarize_gaps, make_plots
from Aebs.dino_latent.saved_pair_provenance import check_final_recheck


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    root = Path("results/dino_expanded_v1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=root/"02_dino_h5")
    parser.add_argument("--data-root", type=Path, default=Path("/root/work_based_on_spvc/carla_data_spvc_expanded/full01"))
    parser.add_argument("--representation", type=Path, default=root/"08_R2_group_alignment/dino_safety_latent.pt")
    parser.add_argument("--base-controller", type=Path, default=root/"09_R2_ppo/latent_ppo.zip")
    parser.add_argument("--spvc-dir", type=Path, default=Path("results/dino_spvc_minimal_v1"))
    parser.add_argument("--final-recheck", type=Path, default=Path("results/dino_spvc_minimal_v1_final_recheck/metrics.json"))
    parser.add_argument("--dino-repo", type=Path, default=Path("../external/dinov2"))
    parser.add_argument("--dino-weights", type=Path, default=Path("../external/dinov2_weights/dinov2_vits14_pretrain.pth"))
    parser.add_argument("--legacy-generator", type=Path, default=Path("Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth"))
    parser.add_argument("--legacy-data", type=Path, default=Path("Aebs/data/Downsampled.h5"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mc-draws", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("results/dino_spvc_minimal_v1_common_latent"))
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory exists: %s; choose a NEW --output-dir" % args.output_dir)
    if args.batch_size < 1 or not 1 <= args.mc_draws <= 128:
        parser.error("batch-size must be positive; mc-draws must be 1..128")
    paths = dict(cache_manifest=args.cache_dir/"cache_manifest.json", features=args.cache_dir/"features.h5",
                 representation=args.representation, base_controller=args.base_controller,
                 saved_actor=args.spvc_dir/"latent_ppo_spvc.pt", saved_barrier=args.spvc_dir/"sbc.pt",
                 spvc_metrics=args.spvc_dir/"metrics.json", final_recheck=args.final_recheck,
                 dino_weights=args.dino_weights, legacy_generator=args.legacy_generator, legacy_data=args.legacy_data,
                 evaluator_source=Path(__file__), backbone_source=Path(__file__).with_name("backbone.py"),
                 metric_source=Path(__file__).with_name("common_latent_metrics.py"),
                 pixel_metric_source=Path(__file__).with_name("proxy_comparison_math.py"),
                 model_source=Path(__file__).with_name("models.py"))
    manifest = args.spvc_dir/"minimal_run_manifest.json"
    if manifest.is_file():
        paths["minimal_manifest"] = manifest
    for path in paths.values():
        if not path.is_file():
            parser.error("missing input: " + str(path))
    if not (args.dino_repo/"hubconf.py").is_file():
        parser.error("local DINO repo missing; pass --dino-repo")
    import matplotlib  # Fail before inference if plotting dependency is missing.
    matplotlib.use("Agg")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(1)
    started = time.monotonic()
    hashes = {key: sha256(path) for key, path in paths.items()}
    identity = check_final_recheck(args.representation, args.base_controller, args.spvc_dir, args.final_recheck)
    cache = json.loads(paths["cache_manifest"].read_text())
    rep = torch.load(args.representation, map_location="cpu")
    dino_source_hash = repository_hash(args.dino_repo)
    if (cache.get("schema") != "spvc_expanded_dino_h5_cache_v1" or cache.get("test_encoded") != 0
            or cache["feature_sha256"] != hashes["features"]
            or cache["provenance"]["dino_weights_sha256"] != hashes["dino_weights"]
            or cache["provenance"]["dino_source_sha256"] != dino_source_hash
            or rep.get("variant") != "R2_group_alignment" or rep.get("dino_frozen") is not True
            or rep["provenance"]["cache_manifest_sha256"] != hashes["cache_manifest"]):
        raise ValueError("cache/representation/backbone provenance mismatch")
    records = cache["records"]
    if any(r["split"] not in ("train", "validation") or r["feature_row"] != i for i, r in enumerate(records)):
        raise ValueError("invalid feature rows or test rows present")
    ids = np.asarray([i for i, r in enumerate(records) if r["split"] == "validation"], dtype=int)
    if not len(ids):
        raise ValueError("empty validation split")
    rows = [records[i] for i in ids]
    with h5py.File(paths["features"], "r") as stream:
        if not stream.attrs["complete"] or stream["features"].shape != (len(records), 384):
            raise ValueError("incomplete/invalid cache")
        if (list(stream["filename"].asstr()[:]) != [r["filename"] for r in records]
                or not np.array_equal(stream["distance_m"][:], [r["distance_m"] for r in records])):
            raise ValueError("cache labels/order mismatch")
        features = np.asarray(stream["features"][ids], dtype=np.float32)
    distances = np.asarray([r["distance_m"] for r in rows], dtype=np.float64)
    if not np.isfinite(features).all() or not np.isfinite(distances).all():
        raise ValueError("nonfinite features/distance")
    scale = float(rep["distance_scale_m"])
    mean, std = rep["feature_mean"].to(device), rep["feature_std"].to(device)
    if (not np.isfinite(scale) or scale <= 0 or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all() or not (std > 0).all()):
        raise ValueError("invalid representation normalization")
    projection, q = SafetyProjection().to(device), PhysicalToLatent().to(device)
    projection.load_state_dict(rep["projection_state_dict"], strict=True)
    q.load_state_dict(rep["physical_to_latent_state_dict"], strict=True)
    backbone = FrozenDinoV2(args.dino_repo, args.dino_weights, torch.device(device)).eval()
    generator = AebsMLPGenerator(4, 1).to(device)
    generator.load_state_dict(torch.load(args.legacy_generator, map_location=device), strict=True)
    for net in (projection, q, generator, backbone.backbone):
        net.eval()
        for parameter in net.parameters():
            parameter.requires_grad_(False)
    with h5py.File(args.legacy_data, "r") as stream:
        legacy_scale = float(np.std(np.asarray(stream["y_train"], dtype=np.float32)))
        native_pixels = np.asarray(stream["X_train"], dtype=np.float32)
        if not np.isfinite(native_pixels).all() or native_pixels.min() < 0 or native_pixels.max() > 1:
            raise ValueError("legacy training pixels are not [0,1]")
    if not np.isfinite(legacy_scale) or legacy_scale <= 0:
        raise ValueError("invalid legacy distance scale")

    def project_feature(value):
        return projection((value-mean)/std)

    def encode_pixels(value):
        chunks = []
        with torch.no_grad():
            for start in range(0, len(value), args.batch_size):
                tensor = torch.from_numpy(value[start:start+args.batch_size]).to(device)
                chunks.append(project_feature(backbone(tensor).float()).cpu().numpy())
        return np.concatenate(chunks)

    with torch.no_grad():
        z_real = project_feature(torch.from_numpy(features).to(device)).cpu().numpy()
        z_q = q(torch.from_numpy((distances[:, None]/scale).astype(np.float32)).to(device)).cpu().numpy()
    appearances = np.asarray([r["weather"]+"|"+r["color"] for r in rows])
    # Recompute one real RGB per appearance to check cached preprocessing.
    anchors = {int(np.flatnonzero(appearances == a)[0]) for a in set(appearances.tolist())}
    low_pixels = np.empty((len(rows), 1, 32, 32), dtype=np.float32)
    reference_check = []
    data_root = args.data_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", experiment="frozen_common_latent_q_vs_cgan_v1", training=False,
                  test_used=False, split="validation", device=device, hashes=hashes,
                  dino_source_sha256=dino_source_hash, final_pair_identity=identity,
                  args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  metric="Euclidean distance / sqrt(32), same frozen P(DINO) coordinate system",
                  interpretation="mae is mean nonnegative vector RMS gap; rmse is RMS of those gaps",
                  generator_pixels="(tanh_output+1)/2; inverse of gen_aebs_images training mapping; no minmax",
                  limitations=["evaluator was optimized for this task and q was trained against it; not independent",
                               "existing methods have unequal training data; not architecture-only causality",
                               "generated 32x32 grayscale vs real RGB mismatch is included; reference is not an additive decomposition",
                               "random cGAN z is not paired with a particular weather label",
                               "all draws retained, repeated appearances/draws are not independent samples",
                               "no PPO/SBC execution, new certificate, or closed-loop performance evaluation",
                               "old weights named cGAN come from the repository; this is not a new equal-data baseline"])
    write_json(args.output_dir/"metrics.json", report)
    try:
        for i, row in enumerate(rows):
            path = (data_root/row["filename"]).resolve()
            if path.parent != data_root:
                raise ValueError("image outside registered root")
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != row["sha256"]:
                raise ValueError("RGB changed since cache: " + row["filename"])
            with Image.open(io.BytesIO(content)) as image:
                if image.format != "PNG" or image.size != (640, 640):
                    raise ValueError("unexpected RGB format/size")
                low_pixels[i, 0] = legacy_pixels(image)
                if i in anchors:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)[None]/255.
                    with torch.no_grad():
                        fresh = backbone(torch.from_numpy(rgb).to(device)).float().cpu().numpy()[0]
                    reference_check.append(float(np.max(np.abs(fresh-features[i]))))
                    # Operational reproducibility tolerance, NOT a safety bound.
                    if not np.allclose(fresh, features[i], rtol=1e-4, atol=1e-3):
                        raise ValueError("RGB/cache DINO features disagree; do not compare different encoders")
            if i == 0 or (i+1) % 500 == 0 or i+1 == len(rows):
                print("RGB/hash/downsample %d/%d" % (i+1, len(rows)), flush=True)
        z_down = []
        for start in range(0, len(rows), args.batch_size):
            z_down.append(encode_pixels(low_pixels[start:start+args.batch_size]))
            if start == 0 or start % (args.batch_size*20) == 0:
                print("real 32x32 reference %d/%d" % (min(start+args.batch_size, len(rows)), len(rows)), flush=True)
        z_down = np.concatenate(z_down)
        unique_d, inverse = np.unique(distances, return_inverse=True)
        draws = np.concatenate([np.zeros((1, 4), dtype=np.float32),
                                np.random.default_rng(args.seed).uniform(-1, 1, (args.mc_draws, 4)).astype(np.float32)])
        n_draws = len(draws)
        total = len(unique_d)*n_draws
        generated_pixels = np.empty((total, 1, 32, 32), dtype=np.float32)
        z_generated = np.empty((total, 32), dtype=np.float32)
        raw_min, raw_max = float("inf"), -float("inf")
        for start in range(0, total, args.batch_size):
            index = np.arange(start, min(start+args.batch_size, total))
            y = (unique_d[index//n_draws, None]/legacy_scale).astype(np.float32)
            with torch.no_grad():
                raw = generator(torch.from_numpy(draws[index % n_draws]).to(device),
                                torch.from_numpy(y).to(device)).cpu().numpy()
            raw_min, raw_max = min(raw_min, float(raw.min())), max(raw_max, float(raw.max()))
            pixels = generator_to_pixels(raw)
            generated_pixels[index] = pixels
            z_generated[index] = encode_pixels(pixels)
            if start == 0 or start % (args.batch_size*20) == 0 or index[-1]+1 == total:
                print("unique distance x cGAN draws %d/%d" % (index[-1]+1, total), flush=True)
        z_generated = z_generated.reshape(len(unique_d), n_draws, 32)
        generated_pixels = generated_pixels.reshape(len(unique_d), n_draws, 1024)
        dino_gap = rms_gap(z_q, z_real)
        cgan_gap = rms_gap(z_generated[inverse], z_real[:, None, :])
        down_gap = rms_gap(z_down, z_real)
        native_gap = np.empty_like(cgan_gap)
        for start in range(0, len(rows), args.batch_size):
            stop = min(start+args.batch_size, len(rows))
            native_gap[start:stop] = rms_gap(generated_pixels[inverse[start:stop]], low_pixels[start:stop].reshape(-1, 1, 1024))
        groups = np.asarray([r["distance_bin_025m"] for r in rows])
        comparison = summarize_gaps(dino_gap, cgan_gap, down_gap, groups, appearances, native_gap)
        if ({key: sha256(path) for key, path in paths.items()} != hashes
                or repository_hash(args.dino_repo) != dino_source_hash):
            raise ValueError("inputs changed during evaluation")
        np.savez_compressed(args.output_dir/"samples.npz", feature_row=ids, filename=np.asarray([r["filename"] for r in rows]),
                            appearance=appearances, distance_m=distances, distance_block=groups,
                            z_real=z_real, z_q=z_q, z_real_downsampled=z_down, z_generated_unique=z_generated,
                            unique_distance_m=unique_d, image_to_unique_distance=inverse, latent_draws=draws,
                            dino_gap=dino_gap, cgan_gap=cgan_gap, downsample_gap=down_gap, cgan_pixel_gap=native_gap)
        plots = make_plots(args.output_dir, dino_gap, cgan_gap, down_gap, appearances, comparison)
        report.update(status="development_common_latent_comparison_complete", images=len(rows),
                      rgb_hashes_checked=len(rows), unique_distances=len(unique_d), random_draws=args.mc_draws,
                      generated_dino_encodings=total, rgb_cache_anchor_count=len(anchors),
                      rgb_cache_anchor_max_abs=max(reference_check), generator_raw_range=[raw_min, raw_max],
                      input_files_unchanged=True, comparison=comparison, figures=plots)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        report["runtime_seconds"] = time.monotonic()-started
        write_json(args.output_dir/"metrics.json", report)
    print("\n[Common semantic-space comparison: frozen models, development only]")
    print("images=%d unique_distances=%d space=32D; no training/PPO/SBC/test" % (len(rows), len(unique_d)))
    for kind, stats in comparison["all_selected"]["comparison"].items():
        print("%s: DINO mean=%.6f p95=%.6f max=%.6f | cGAN mean=%.6f p95=%.6f max=%.6f | lower=%s; images_better=%.2f%%" % (
            kind, stats["dino"]["mae"], stats["dino"]["p95_abs"], stats["dino"]["max_abs"],
            stats["legacy"]["mae"], stats["legacy"]["p95_abs"], stats["legacy"]["max_abs"],
            stats["sampled_mae_and_rmse_lower"], 100*stats["fraction_images_dino_below_cgan_mean"]))
    print("real 32x32 reference gap:", comparison["all_selected"]["downsample_reference_gap"])
    print("Native pixel RMS is reported separately; NEVER compare its magnitude to latent RMS.")
    print("Task-trained evaluator; no architecture-only or safety claim.")
    print("metrics:", args.output_dir/"metrics.json")
    print("figures:", ", ".join(plots))


if __name__ == "__main__":
    main()
