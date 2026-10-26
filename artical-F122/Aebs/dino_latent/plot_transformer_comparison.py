"""Plot mean/tail trade-offs from a completed comparison, without training."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True)
    args = parser.parse_args()
    data = json.loads((args.results/"metrics.json").read_text())
    if data.get("source_files_unchanged") is not True:
        parser.error("Need completed, source-checked results")
    target = args.results/"error_comparison.png"
    if target.exists(): parser.error("Figure already exists; do not overwrite")
    names = [a["name"] for a in data["config"]["learning_arms"]]
    labels = [n.replace("t32_","32 ").replace("t384_","384 ")
              .replace("mlp32_","MLP32 ").replace("_"," ") for n in names]
    metrics = data["results"]
    fig, axes = plt.subplots(1,3,figsize=(18,6),constrained_layout=True)
    panels = [("distance_error","mae","Future distance MAE (m)"),
              ("successor_speed_gap","mae","Policy successor mean gap (m/s)"),
              ("successor_speed_gap","max","Policy successor maximum gap (m/s)")]
    for ax,(metric,stat,title) in zip(axes,panels):
        values = [metrics[n]["metrics"]["1"][metric][stat] for n in names]
        colors = ["#d88436" if "384" in n else "#377db6" for n in names]
        bars = ax.barh(labels,values,color=colors)
        ax.invert_yaxis(); ax.set_title(title); ax.grid(axis="x",alpha=.2)
        ax.set_axisbelow(True); ax.set_xlim(0,max(values)*1.22)
        for bar,value in zip(bars,values):
            ax.text(value+max(values)*.012,bar.get_y()+bar.get_height()/2,
                    "%.5f"%value,va="center",fontsize=9)
        if metric == "successor_speed_gap":
            base = metrics["current_q_known_distance"]["metrics"]["1"][metric][stat]
            ax.axvline(base,color="#333333",linestyle="--",linewidth=1,label="Current q (known distance)")
            ax.legend(loc="lower right",fontsize=8)
    fig.suptitle("One-step development prediction: same trajectory split; frozen PPO still receives true speed",fontsize=13)
    fig.savefig(target,dpi=160)
    plt.close(fig)
    print(target)


if __name__ == "__main__": main()
