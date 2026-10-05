"""Frozen DINO RGB cache for audited train/validation only; no training."""
import argparse
import hashlib
import io
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import h5py
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from Aebs.dino_latent.backbone import FrozenDinoV2
from Aebs.dino_latent.prepare_expanded_data import distance_group, sha256


def select_records(manifest):
    if manifest.get("schema") != "spvc_expanded_split_v1":
        raise ValueError("Unsupported split manifest")
    groups = manifest["groups"]
    if set(groups) != {"train", "validation", "test"}:
        raise ValueError("Unexpected splits")
    if sorted(sum(groups.values(), [])) != list(range(44)):
        raise ValueError("Group allocation overlaps or is incomplete")
    names, exact, counts, chosen = set(), {}, Counter(), []
    for row in manifest["records"]:
        name, split = row["filename"], row["split"]
        if name in names or Path(name).name != name or not name.endswith(".png"):
            raise ValueError("Duplicate or unsafe image name")
        names.add(name)
        group = distance_group(row["distance_m"])
        if split not in groups or group not in groups[split] or row["distance_bin_025m"] != group:
            raise ValueError("Record/group split mismatch")
        prev = exact.setdefault(row["group_id"], split)
        if prev != split:
            raise ValueError("Exact-distance group leaks across splits")
        counts[split] += 1
        if split in ("train", "validation"):
            chosen.append({**row, "feature_row": len(chosen)})
    if any(counts[k] == 0 for k in groups):
        raise ValueError("Empty split")
    return chosen, dict(counts)


class RGBImages(Dataset):
    def __init__(self, root, records):
        self.root = Path(root).resolve()
        if any(r["split"] not in ("train", "validation") for r in records):
            raise ValueError("Test records forbidden in feature dataset")
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        path = (self.root/row["filename"]).resolve()
        if path.parent != self.root:
            raise ValueError("Image outside data root")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("Image changed after audit: %s" % row["filename"])
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or image.size != (640, 640):
                raise ValueError("Unexpected image format/size")
            value = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        return torch.from_numpy(value).permute(2, 0, 1).float().div_(255)


def repository_hash(root):
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if ".git" not in path.parts:
            h.update((str(path.relative_to(root))+":"+sha256(path)+"\n").encode())
    return h.hexdigest()


def initialize_h5(stream, records, provenance, root):
    """Row-aligned raw DINO features and labels; no test rows or normalization."""
    if not records or any(r["split"] not in ("train", "validation") for r in records):
        raise ValueError("H5 requires nonempty train/validation records only")
    stream.attrs["schema"] = "spvc_expanded_dino_h5_v1"
    stream.attrs["complete"] = False
    stream.attrs["data_root"] = str(root)
    stream.attrs["provenance_json"] = json.dumps(provenance)
    stream.attrs["feature_dimension"] = 384
    stream.attrs["test_encoded"] = 0
    stream.attrs["feature_normalization"] = "none"
    features = stream.create_dataset("features", shape=(len(records),384), dtype="float32",
                                     chunks=(min(256,len(records)),384), compression="lzf")
    strings = h5py.string_dtype(encoding="utf-8")
    for key in ["filename", "group_id", "weather", "color", "split", "sha256"]:
        stream.create_dataset(key, data=np.asarray([r[key] for r in records], dtype=object), dtype=strings)
    stream.create_dataset("distance_m", data=np.asarray([r["distance_m"] for r in records], dtype=np.float64))
    stream.create_dataset("distance_bin_025m", data=np.asarray([r["distance_bin_025m"] for r in records], dtype=np.int32))
    for split in ("train", "validation"):
        stream.create_dataset(split+"_indices", data=np.asarray(
            [i for i,r in enumerate(records) if r["split"] == split], dtype=np.int64))
    return features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("results/dino_expanded_v1/01_data_prepare/split_manifest.json"))
    p.add_argument("--output-dir", type=Path, default=Path("results/dino_expanded_v1/02_dino_h5"))
    p.add_argument("--dino-repo", type=Path, default=Path("../external/dinov2"))
    p.add_argument("--dino-weights", type=Path, default=Path("../external/dinov2_weights/dinov2_vits14_pretrain.pth"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = p.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        p.error("Invalid batch-size/workers")
    if args.output_dir.exists():
        p.error("Output exists; use a new directory. Partial caches are not reusable.")
    manifest = json.loads(args.manifest.read_text())
    records, counts = select_records(manifest)
    root = Path(manifest["data_root"]).resolve()
    for name, expected in manifest["source_hashes"].items():
        path = (root/name).resolve()
        if path.parent != root or sha256(path) != expected:
            raise ValueError("Source metadata changed since audit: %s" % name)
    if not (args.dino_repo/"hubconf.py").is_file() or not args.dino_weights.is_file():
        p.error("Local DINO repository/weights missing")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status="running", cached_rows=0, feature_dimension=384, device=device,
                  storage_format="hdf5",
                  split_counts=counts, test_encoded=0, test_evaluated=False,
                  dino_frozen=True, training_run=False, visual_review=manifest.get("visual_review", "pending"))
    try:
        provenance = dict(split_manifest_sha256=sha256(args.manifest),
            dino_weights_sha256=sha256(args.dino_weights), dino_source_sha256=repository_hash(args.dino_repo),
            script_sha256=sha256(Path(__file__)), torch_version=torch.__version__,
            preprocess="PIL RGB /255; existing FrozenDinoV2 bicubic antialias 224; ImageNet normalization",
            feature_normalization="none: fit only on each experiment's permitted training subset",
            args={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
        (args.output_dir/"config.json").write_text(json.dumps(provenance, indent=2))
        model = FrozenDinoV2(args.dino_repo, args.dino_weights, torch.device(device)).eval()
        if any(parameter.requires_grad for parameter in model.backbone.parameters()):
            raise RuntimeError("Backbone not frozen")
        loader = DataLoader(RGBImages(root, records), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=device == "cuda")
        partial = args.output_dir/"features.partial.h5"
        with h5py.File(partial, "w") as stream:
            features = initialize_h5(stream, records, provenance, root)
            stream.attrs["visual_review"] = report["visual_review"]
            offset = 0
            for batch_index, images in enumerate(loader):
                with torch.no_grad():
                    values = model(images.to(device)).detach().float().cpu().numpy()
                if values.shape != (len(images), 384) or not np.isfinite(values).all():
                    raise ValueError("DINO features have invalid shape or nonfinite values")
                features[offset:offset+len(values)] = values
                offset += len(values)
                report["cached_rows"] = offset
                if batch_index == 0 or (batch_index+1) % 50 == 0 or offset == len(records):
                    stream.flush()
                    print("DINO frozen cache %d/%d elapsed=%.1fs" % (offset, len(records), time.monotonic()-started), flush=True)
            if offset != len(records):
                raise RuntimeError("Incomplete feature extraction")
            stream.attrs["complete"] = True
        partial.rename(args.output_dir/"features.h5")
        cache = dict(schema="spvc_expanded_dino_h5_cache_v1", data_root=str(root), records=records,
                     groups=manifest["groups"], provenance=provenance, test_encoded=0,
                     feature_file="features.h5", feature_dataset="features", feature_shape=[len(records),384],
                     feature_sha256=sha256(args.output_dir/"features.h5"))
        (args.output_dir/"cache_manifest.json").write_text(json.dumps(cache, indent=2))
        report["status"] = "cache_complete_pending_visual_review" if report["visual_review"] == "pending" else "cache_complete"
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        report["runtime_seconds"] = time.monotonic()-started
        (args.output_dir/"metrics.json").write_text(json.dumps(report, indent=2))
        print("[Expanded RGB frozen DINO H5 — no training]")
        for k,v in report.items():
            print("%s: %s" % (k,v))
        print("metrics: %s" % (args.output_dir/"metrics.json"))
        print("No latent alignment errors yet; no projection/PPO/SBC training.")


if __name__ == "__main__":
    main()
