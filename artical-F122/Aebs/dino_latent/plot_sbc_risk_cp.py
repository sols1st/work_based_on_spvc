"""Scientific figures for the frozen trajectory-risk CP experiments."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from Aebs.dino_latent.experiment_sbc_risk_cp import FrozenBarrier, SCORE_NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline",type=Path,default=Path("results/native_R2_sbc_risk_cp_20261020_v1"))
    p.add_argument("--repaired",type=Path,default=Path("results/native_R2_sbc_risk_cp_20261020_v2"))
    p.add_argument("--legacy-barrier-dir",type=Path,default=Path("results/native_R2_sbc_stage2_legacy_20261019_v1"))
    a=p.parse_args(); torch.set_num_threads(1)
    m0=json.loads((a.baseline/"metrics.json").read_text()); m1=json.loads((a.repaired/"metrics.json").read_text())
    figures=a.repaired/"figures"; figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":160})
    names=["Cumulative positive change", "Proxy + mismatch budgets", "400 x largest increment", "Largest barrier value"]
    fig,ax=plt.subplots(figsize=(10,4.5)); y=np.arange(len(names))
    q0=[m0["trajectory_score_calibration"][n]["tolerance"]["q_hat"] for n in SCORE_NAMES]
    q1=[m1["trajectory_score_calibration"][n]["tolerance"]["q_hat"] for n in SCORE_NAMES]
    ax.barh(y-.17,q0,height=.3,color="#929aa5",label="Frozen legacy SBC as monitor")
    ax.barh(y+.17,q1,height=.3,color="#236a9b",label="Development-trained risk monitor")
    ax.set_xscale("symlog",linthresh=.05); ax.axvline(0,color="#aa3333",lw=1)
    ax.set_xticks([-1,-.1,0,.1,1,10]); ax.set_xticklabels(["-1","-0.1","0","0.1","1","10"])
    ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xlabel("Calibrated trajectory score q (q < 0 required for safety inference)")
    for yy,x in zip(y,q0): ax.annotate("%.5f"%x,(x,yy-.17),xytext=(4,0),textcoords="offset points",va="center",fontsize=8)
    for yy,x in zip(y,q1): ax.annotate("%.5f"%x,(x,yy+.17),xytext=(4,0),textcoords="offset points",va="center",fontsize=8)
    ax.legend(loc="lower right",fontsize=8); ax.set_title("Same 99% coverage / 99% confidence targets; different safety scores")
    fig.tight_layout(); fig.savefig(figures/"01_score_comparison.png"); plt.close(fig)
    data=np.load(a.repaired/"evaluation_rollouts.npz")
    i=int(np.argmax(data["positive_variation"])); length=int(data["valid"][i].sum())
    s=data["states_m_mps"][i,:length+1]; b=data["stopped_barrier"][i,:length+1]
    bm=json.loads((a.legacy_barrier_dir/"metrics.json").read_text())
    old=FrozenBarrier(a.legacy_barrier_dir/"sbc.pt",bm,m1["config"]["barrier_normalization"]["distance_scale"])
    bold=old.monitor(s)
    fig,axes=plt.subplots(2,2,figsize=(10,6.5)); t=np.arange(length+1)
    axes[0,0].plot(t,s[:,0],color="#236a9b"); axes[0,0].axhline(6,ls="--",color="#aa3333",label="6 m boundary")
    axes[0,0].set_ylabel("Distance (m)"); axes[0,0].legend(fontsize=8)
    axes[0,1].plot(t,s[:,1],color="#236a9b"); axes[0,1].axhline(.5,ls="--",color="#aa3333",label="0.5 m/s boundary")
    axes[0,1].set_ylabel("Speed (m/s)"); axes[0,1].legend(fontsize=8)
    axes[1,0].plot(t,bold,label="Legacy SBC monitor",color="#929aa5")
    axes[1,0].plot(t,b,label="New risk monitor",color="#236a9b")
    axes[1,0].set_ylabel("Stopped risk value [0,1]"); axes[1,0].legend(fontsize=8)
    axes[1,1].plot(t,b,color="#236a9b"); axes[1,1].set_ylabel("New risk value (zoomed)")
    for ax in axes.flat: ax.set_xlabel("Step"); ax.grid(alpha=.15)
    fig.suptitle("Identical successful image-library trajectory; largest new evaluation risk score")
    fig.tight_layout(); fig.savefig(figures/"02_same_trajectory_monitors.png"); plt.close(fig)
    print("figures:",figures,flush=True)


if __name__=="__main__": main()
