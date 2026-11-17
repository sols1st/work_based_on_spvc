"""Render the paired deployment and fixed-controller results from their saved runs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Aebs.lane_keeping.core import PHI_MAX


def read_json(path):
    return json.loads(path.read_text())


def plot_camera(ax, results):
    root = results / "aebs_native_cross_20261117_v1" / "fixed_tube_2000"
    report = read_json(root / "report.json")
    threshold = report["threshold_m"]
    for name, label, color, style in [
        ("rgb_native", "Released SPVC head", "#354255", "--"),
        ("rgb_dino", "Target-adapted DINO", "#0b715b", "-"),
    ]:
        with np.load(root / (name + ".npz")) as runs:
            scores = np.sort(runs["max_truth_m"])
        assert np.count_nonzero(scores <= threshold) == report["arms"][name]["in_tube"]
        x = np.r_[0.1, scores, 10.]
        y = np.r_[0, np.arange(1, len(scores) + 1) / len(scores), 1]
        ax.step(x, y, where="post", color=color, ls=style, label=label)
    ax.axvline(threshold, color="#a1442b", lw=1, ls=":")
    ax.text(threshold * 1.08, .48, "Certified tolerance\n0.45 m", color="#903d2a", va="center")
    ax.set(xscale="log", xlim=(.1, 10.), ylim=(-.03,1.04),
           xlabel="Allowed distance error (m, log scale)", ylabel="Fraction of trajectories in tube")
    ax.set_title("(a) AEBS deployment", loc="left")
    ax.legend(loc="center", bbox_to_anchor=(.58, .79), frameon=False, fontsize=8)


def plot_lane(ax, results):
    report = read_json(results / "lane_contract_comparison_20261117_v1" / "refinement.json")
    # Each physical angle is PHI_MAX times the normalized steering input.
    for y, (arm, label) in enumerate([("uniform", "Uniform box"),
                                      ("diagonal", "Scaled box"),
                                      ("joint", "Scaled + action")]):
        entry = report["arms"][arm]
        passed = entry["largest_pass"] * PHI_MAX
        witness = entry["first_witness"] * PHI_MAX
        ax.plot([passed, witness], [y, y], color="#818993", ls=":")
        ax.plot(passed, y, "o", color="#0b715b", label="Verified" if y == 0 else None)
        ax.plot(witness, y, "x", color="#a1442b", mew=1.8, label="Drift witness" if y == 0 else None)
        ax.text(.0615, y-.16, label, fontsize=8, color="#263749")
    ax.set(xlim=(.061, .093), ylim=(2.89,-.45), yticks=[], xlabel="Additional steering error (rad)")
    ax.set_xticks([.0675,.072,.0855,.09], ["0.0675","0.072","0.0855","0.09"], fontsize=8)
    ax.set_title("(b) Lane certificate", loc="left")
    ax.legend(loc="lower center", frameon=False, fontsize=8, ncol=2)


def plot_crane(ax, results):
    fits = read_json(results / "crane_contract_confirmation_20261117_v1" / "summary.json")["fits"]
    for y, (arm, label) in enumerate([("box","Uniform box"), ("box_and_action","Visual + action")]):
        upper = fits[arm]["upper"]
        color = "#0b715b" if fits[arm]["passes"] else "#354255"
        ax.barh(y, upper, height=.42, color=color)
        ax.text(.03, y, label, va="center", color="white", fontsize=8)
        ax.text(upper+.032, y, f"{upper:.4f}", va="center", fontsize=8)
    ax.axvline(1., color="#a1442b", ls=":", lw=1.5)
    ax.text(.98, 1.7, "pass < 1", ha="right", color="#903d2a")
    ax.set(xlim=(0,1.31), ylim=(1.93,-.63), yticks=[], xlabel="Worst constraint / physical limit")
    ax.set_title("(c) Crane at 1.5× initial scale", loc="left")


def main(args):
    plt.rcParams.update({"font.size":9, "axes.titlesize":10, "axes.titleweight":"bold",
                         "svg.fonttype":"none", "font.family":"DejaVu Sans", "axes.axisbelow":True})
    fig, axes = plt.subplots(1,3,figsize=(10.6,3.15), layout="constrained")
    for ax, plot in zip(axes, [plot_camera, plot_lane, plot_crane]):
        plot(ax, args.results)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="x", color="#d9dfe3", lw=.55)
    args.output.mkdir(exist_ok=True, parents=True)
    for ext in ["svg","png"]:
        fig.savefig(args.output / ("paper_benefits." + ext), dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--results", type=Path, default=Path("results"))
    cli.add_argument("--output", type=Path, default=Path("../docs/assets/spvc_benefit_20261117"))
    main(cli.parse_args())
