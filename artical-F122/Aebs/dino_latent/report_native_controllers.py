"""Read-only evidence checks and task-qualified native-controller summary."""
import argparse
import json
from pathlib import Path

from Aebs.dino_latent.prepare_expanded_data import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.results / "metrics.json").read_text())
    if result["status"] != "native_controller_development_comparison_complete" or result["failures"]:
        raise ValueError("All preregistered controller runs must finish first")
    if set(result["results"]) != set(result["config"]["arms"]):
        raise ValueError("Missing controller")
    eligible = []
    for name in result["config"]["arms"]:
        record = result["results"][name]
        ck = args.results / name / "native_ppo.zip"
        if sha256(ck) != record["controller_sha256"]:
            raise ValueError("Saved controller differs from evaluated controller")
        rr = record["reference_residual"]
        if rr["trajectory_score"]["trajectories_checked"] != 459 or rr["fresh_CP_guarantee"]:
            raise ValueError("Reference scope mismatch")
        if record["sampled_task_gate"]:
            eligible.append(name)
    ordering = lambda names, metric: sorted(names, key=lambda name: metric(result["results"][name]))
    radius = lambda r: r["reference_residual"]["same_recipe_radius_mps"]
    mean = lambda r: r["reference_residual"]["step"]["mae"]
    summary = dict(controllers=len(result["results"]), task_qualified=eligible,
                   qualified_order_by_reference_radius=ordering(eligible, radius),
                   qualified_order_by_reference_mae=ordering(eligible, mean),
                   all_order_by_reference_radius=ordering(result["results"], radius),
                   reference_steps=next(iter(result["results"].values()))["reference_residual"]["step"]["count"],
                   actual_timesteps={name: r["actual_timesteps"] for name, r in result["results"].items()},
                   every_saved_controller_hash_matches=True,
                   limitations=["reused off-policy reference trajectories, not independent candidate calibration",
                                "task screen uses train-image library replay, not new online CARLA",
                                "a different trained PPO/q for each representation means system comparison, not isolated encoder causality",
                                "none of these residual radii alone is an SBC or image-loop safety certificate"])
    (args.results / "selection_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print("[Native-controller physical residual comparison]", flush=True)
    print("controllers=%d, task-qualified=%d" % (summary["controllers"], len(eligible)), flush=True)
    for name in ordering(result["results"], radius):
        r = result["results"][name]; step = r["reference_residual"]["step"]
        audit = r["matched"]["internal_audit"]["steps"]
        print("%s radius=%.9f MAE=%.9f max_static=%.9f surrogate=%.2f%% image=%.2f%% image_unsafe=%.2f%% qualified=%s" % (
            name, radius(r), mean(r), audit["max_abs"],
            100 * r["rollouts"]["surrogate"]["success_rate"], 100 * r["rollouts"]["image"]["success_rate"],
            100 * r["rollouts"]["image"]["unsafe_rate"], r["sampled_task_gate"]), flush=True)
    print("Saved controller hashes verified. Reference diagnostics only, not a fresh CP guarantee.")


if __name__ == "__main__": main()
