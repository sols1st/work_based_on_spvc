"""Read-only metrics/weights; produce figures for completed development pilots."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS={"mlp32":"MLP-small","mlp32_parameter_matched":"MLP-matched","transformer32":"Transformer",
    "transformer32_no_speed":"Transformer-no-v","gru32":"GRU","rssm32":"Gaussian-RSSM",
    "lifted_linear32":"Lifted-linear","transformer32_task_aware":"Transformer-task",
    "pca32_transformer":"PCA-Transformer","original_R2":"R2-preserved",
    "r2_frozen_refit_q":"R2-refit-q","pca32":"PCA32","feature_ae32":"Feature-AE32","feature_vae32":"Feature-VAE32"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--temporal",type=Path,required=True); p.add_argument("--static",type=Path,required=True)
    a=p.parse_args()
    temporal=json.loads((a.temporal/"metrics.json").read_text())
    static=json.loads((a.static/"metrics.json").read_text())
    decomp=json.loads((a.static/"decomposition.json").read_text())
    if not temporal.get("source_files_unchanged") or not static.get("source_files_unchanged"):
        raise ValueError("Unfinished or changed-source experiment")
    outputs=(a.temporal/"temporal_comparison.png",a.static/"static_comparison.png")
    if any(x.exists() for x in outputs): raise ValueError("New figure paths required")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    names=[r["name"] for r in temporal["config"]["arms"]]
    colors=dict(zip(names,plt.get_cmap("tab10").colors))
    hs=[1,3,temporal["config"]["horizon"]]
    fig,axes=plt.subplots(2,2,figsize=(15,11),layout="constrained")
    for key,ax,title in (("distance_error_m",axes[0,0],"Forecast distance MAE (m)"),
                         ("next_speed_gap_mps",axes[0,1],"Frozen-policy successor-speed MAE (m/s)")):
        for n in names:
            y=[temporal["results"][n]["metrics"][str(h)]["all"][key]["mae"] for h in hs]
            ax.plot(hs,y,marker="o",label=LABELS[n],color=colors[n],linewidth=2 if "task_aware" in n else 1.2)
        ax.set(title=title,xlabel="Open-loop prediction step"); ax.set_xticks(hs); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,ncol=2)
    candidates=[n for n in names if n!="transformer32"]
    ax=axes[1,0]
    for i,n in enumerate(candidates):
        pair=temporal["paired_vs_transformer32"][n][str(hs[-1])]["next_speed_gap_mps"]
        point=pair["mean_paired_delta"]; lo,hi=pair["ci95"]
        ax.errorbar(point,i,xerr=np.array([[point-lo],[hi-point]]),fmt="o",color=colors[n],capsize=4)
    ax.axvline(0,color="black",linestyle="--",linewidth=1)
    ax.set_yticks(range(len(candidates)),[LABELS[n] for n in candidates]); ax.invert_yaxis()
    ax.set(title="Step-10 paired trajectory delta vs Transformer",xlabel="Mean absolute speed-gap difference (m/s); left is better")
    ax.grid(axis="x",alpha=.2)
    ax=axes[1,1]; xpos=np.arange(len(names))
    near=[temporal["results"][n]["metrics"][str(hs[-1])]["near_obstacle_d_le_8_v_ge_0.5"] for n in names]
    ax.bar(xpos,[x["next_speed_gap_mps"]["mae"] for x in near],color=[colors[n] for n in names])
    ax.set_xticks(xpos,[LABELS[n] for n in names],rotation=40,ha="right")
    ax.set(title="Step-10 near-obstacle mean gap",ylabel="Successor-speed MAE (m/s)")
    ax.grid(axis="y",alpha=.2)
    fig.suptitle("Old-pool internal development audit | 69 trajectories | Frozen R2/PPO/SBC | No safety claim",fontsize=13)
    fig.savefig(outputs[0],dpi=170); plt.close(fig)
    names=list(static["results"]); xpos=np.arange(len(names)); palette=["#22577a","#38a3a5","#f0a202","#d95d39","#8f5bb5"]
    fig,axes=plt.subplots(1,3,figsize=(16,5.5),layout="constrained")
    y=[static["results"][n]["common_proxy_vs_original_image_gap"]["mean"] for n in names]
    axes[0].bar(xpos,y,color=palette); axes[0].set(title="Proxy vs original image: common R2 gap",ylabel="Mean per-image latent RMS")
    y=[static["results"][n]["all_pair_proxy_vs_original_speed_gap_mps"]["mae"] for n in names]
    axes[1].bar(xpos,y,color=palette); axes[1].set(title="Proxy vs original image: physical gap",ylabel="Successor-speed MAE (m/s)")
    y=[decomp["results"][n]["native"]["appearance_fraction_of_centered_image_variance"]*100 for n in names]
    axes[2].bar(xpos,y,color=palette); axes[2].set(title="Same-distance appearance spread",ylabel="Fraction of latent variance (%)")
    for ax in axes:
        ax.set_xticks(xpos,[LABELS[n] for n in names],rotation=35,ha="right"); ax.grid(axis="y",alpha=.2)
    fig.suptitle("Static dual-path audit | fixed task-trained R2 readout | no PPO retraining | not independent final test",fontsize=12)
    fig.savefig(outputs[1],dpi=170); plt.close(fig)
    print("figures:",*[str(x) for x in outputs],sep="\n")


if __name__=="__main__": main()
