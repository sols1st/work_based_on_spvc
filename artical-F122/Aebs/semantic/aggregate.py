"""Aggregate repeated semantic-abstraction runs without inspecting test data during training."""

import argparse
import json
from pathlib import Path

import numpy as np


FIELDS = {
    "legacy_mae_m": ("legacy_real_image", "mae_m"),
    "semantic_mae_m": ("semantic_point", "mae_m"),
    "semantic_rmse_m": ("semantic_point", "rmse_m"),
    "global_coverage": ("conformal_global", "test_coverage"),
    "global_mean_width_m": ("conformal_global", "mean_width_m"),
    "mondrian_coverage": ("conformal_mondrian", "test_coverage"),
    "mondrian_mean_width_m": ("conformal_mondrian", "mean_width_m"),
    "state_contract_coverage": ("state_conditional_contract", "test_coverage"),
    "state_contract_mean_width_m": ("state_conditional_contract", "mean_width_m"),
    "action_mae": ("control_sufficiency", "action_mae"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    runs = []
    for path in args.metrics:
        with open(path, "r", encoding="utf-8") as stream:
            runs.append(json.load(stream))
    summary = {"run_count": len(runs), "seeds": [run["seed"] for run in runs], "metrics": {}}
    for name, keys in FIELDS.items():
        values = np.asarray([run[keys[0]][keys[1]] for run in runs], dtype=float)
        summary["metrics"][name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
