"""Export registered shift metrics and the prediction-independent display example."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(root, group):
    folder = root / group
    manifest = read(folder / "prediction_manifest.json")
    models = {(r["key"], r["condition"]): r for r in manifest["models"]}
    records = read(folder / "metrics.json")
    flat = []
    for row in records:
        if row["grouping"] != "all":
            continue
        model = models[row["head"], row["condition"]]
        flat.append(dict(budget=row["budget"], train_seed=row["train_seed"], mode=model["mode"],
                         condition=row["condition"], mae_m=row["mae_m"], rmse_m=row["rmse_m"],
                         p95_abs_m=row["p95_absolute_m"], max_abs_m=row["max_absolute_m"],
                         signed_bias_m=row["signed_bias_m"], n=row["n"], family=row["family"],
                         checkpoint_sha256=model["checkpoint_sha256"]))
    with (folder / "static_flat.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    example = read(root / "example_images.json")
    predictions = []
    for model in manifest["models"]:
        if model["budget"] != 24000 or model["train_seed"] != 27411:
            continue
        path = Path(model["prediction_path"])
        if sha(path) != model["prediction_sha256"]:
            raise ValueError("prediction file changed")
        value = float(np.load(path)[example["row"]]) * float(manifest["distance_scale_m"])
        predictions.append(dict(mode=model["mode"], family=model["family"], condition=model["condition"],
                                prediction_m=value, absolute_error_m=abs(value - example["distance_m"])))
    record = dict(example=example, budget=24000, train_seed=27411, predictions=predictions,
                  prediction_manifest_sha256=sha(folder / "prediction_manifest.json"),
                  metrics_sha256=sha(folder / "metrics.json"), export_source_sha256=sha(__file__))
    (folder / "example_predictions.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(group=group, all_population_rows=len(flat), example_predictions=len(predictions))))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("group", choices=["legacy18", "new27"])
    a = p.parse_args()
    export(a.root, a.group)
