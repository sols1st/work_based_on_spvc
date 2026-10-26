"""One missing architecture/objective control; OLD development data only.

Same initialization/budget/split/objective as the saved benchmark, but the
parameter-matched MLP gets the task auxiliaries previously used by Transformer.
No PPO/SBC/representation update, CP pool access, or deployment claim.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from Aebs.dino_latent.models import SafetyProjection,SafetyDecoder
from Aebs.dino_latent.spvc_saved_actor import load_spvc_pair
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.latent_method_models import make_model,predict,temporal_offsets
from Aebs.dino_latent.temporal_world_model import advance_speed


def queried_speeds(context_speed,controls,context,horizon):
    if controls.shape[1]!=context+horizon-1:
        raise ValueError("Wrong recorded control query length")
    value=context_speed[:,-1]; values=[]
    for h in range(horizon):
        value=advance_speed(value,controls[:,context-1+h]); values.append(value)
    return torch.stack(values,1)


def task_auxiliary(predicted,target,distance,velocities,decoder,actor,distance_scale):
    def actions(z):
        return actor(torch.cat((z,velocities[...,None]),-1).flatten(0,1)).reshape_as(velocities).clamp(-3,3)
    with torch.no_grad(): target_actions=actions(target)
    return (F.mse_loss(decoder(predicted).squeeze(-1),distance/distance_scale)
            +.1*F.mse_loss(actions(predicted)/3,target_actions/3))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark",type=Path,default=Path("results/latent_methods_benchmark_20261018_run1"))
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    a=p.parse_args()
    if a.output_dir.exists(): p.error("New output directory required")
    cfg=json.loads((a.benchmark/"config.json").read_text())
    if cfg["fresh_CP_used"] is not False or cfg["test_used"] is not False:
        raise ValueError("Only old development benchmark accepted")
    files={str(f):digest for f,digest in cfg["source_sha256"].items()}
    for name in ("config.json","coordinates.pt","windows.npz","split.json","mlp32_parameter_matched.pt","transformer32_task_aware.pt"):
        path=a.benchmark/name; files[str(path)]=sha256(path)
    if not all(sha256(Path(f))==h for f,h in files.items()): raise ValueError("Frozen source changed")
    trajectories=next(Path(f) for f in files if "13_spvc_calibration_trajectories_v1/trajectories.h5" in f)
    representation=next(Path(f) for f in files if f.endswith("08_R2_group_alignment/dino_safety_latent.pt"))
    controller=next(Path(f) for f in files if f.endswith("09_R2_ppo/latent_ppo.zip"))
    spvc=next(Path(f).parent for f in files if f.endswith("latent_ppo_spvc.pt"))
    recheck=next(Path(f) for f in files if "final_recheck/metrics.json" in f)
    device=torch.device(a.device); torch.set_num_threads(1)
    rep=torch.load(representation,map_location="cpu")
    projection,decoder=SafetyProjection().to(device),SafetyDecoder().to(device)
    projection.load_state_dict(rep["projection_state_dict"]);decoder.load_state_dict(rep["auxiliary_decoder_state_dict"])
    _,student,_,_=load_spvc_pair(representation,controller,spvc,recheck)
    actor=student.policy.controller_net.to(device)
    for net in (projection,decoder,actor):
        net.eval()
        for par in net.parameters(): par.requires_grad_(False)
    with h5py.File(trajectories) as f:
        valid=f["valid"][:].astype(bool); raw=f["dino_features"][:]
        vs,ac,ds=(f[k][:] for k in ("speed_mps","image_action","distance_m"))
    with torch.no_grad():
        features=(torch.from_numpy(np.nan_to_num(raw)).to(device)-rep["feature_mean"].to(device))/rep["feature_std"].to(device)
        latent=projection(features.flatten(0,1)).reshape(*valid.shape,32)
    speed,control,distance=[torch.from_numpy(np.nan_to_num(x)).to(device) for x in (vs,ac,ds)]
    coords=torch.load(a.benchmark/"coordinates.pt",map_location="cpu")
    mean,std=[x.to(device) for x in coords["normalization"]["r2"]]
    windows=np.load(a.benchmark/"windows.npz")
    context,horizon=cfg["context"],cfg["horizon"]
    co,to,ao=[torch.as_tensor(x,device=device) for x in temporal_offsets(context,horizon)]
    def batch(rows):
        w=torch.as_tensor(rows,device=device); ei=w[:,0,None]; ti=w[:,1,None]
        return ((latent[ei,ti+co]-mean)/std,speed[ei,ti+co],control[ei,ti+ao],
                (latent[ei,ti+to]-mean)/std,distance[ei,ti+to],latent[ei,ti+to])
    @torch.no_grad()
    def selection(model):
        error,count=0.,0
        for start in range(0,len(windows["selection"]),512):
            z,v,u,target,_,_=batch(windows["selection"][start:start+512])
            pred,_=predict(model,z,v,u,horizon)
            e=(pred-target)*std; error+=float(e.square().sum());count+=e.numel()
        return (error/count)**.5
    torch.manual_seed(cfg["seed"])
    model=make_model("mlp_matched").to(device)
    rng=np.random.default_rng(cfg["seed"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    best_score=selection(model); best=copy.deepcopy(model.state_dict());best_step=0
    a.output_dir.mkdir(parents=True)
    registration=dict(name="mlp32_parameter_matched_task_aware",kind="mlp_matched",space="r2",
        parent_benchmark=str(a.benchmark),source_sha256=files,steps=cfg["steps"],batch_size=cfg["batch_size"],
        horizon=horizon,context=context,seed=cfg["seed"],task_weights=cfg["task_aux_weights"],
        selection=cfg["selection"],test_used=False,CP_pool_used=False,old_development_only=True,
        original_PPO_SBC_unchanged=True,code_sha256=sha256(Path(__file__)))
    (a.output_dir/"config.json").write_text(json.dumps(registration,indent=2))
    history=[];tic=time.monotonic()
    for step in range(1,cfg["steps"]+1):
        model.train(); ix=rng.integers(len(windows["train"]),size=cfg["batch_size"])
        hist,v,u,target,dd,real=batch(windows["train"][ix])
        pred,aux=predict(model,hist,v,u,horizon,target)
        future_v=queried_speeds(v,u,context,horizon)
        loss=F.mse_loss(pred,target)+aux+task_auxiliary(pred*std+mean,real,dd,future_v,decoder,actor,float(rep["distance_scale_m"]))
        if not torch.isfinite(loss): raise ValueError("Nonfinite candidate loss")
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step==1 or step%200==0 or step==cfg["steps"]:
            model.eval(); score=selection(model)
            if score<best_score: best_score,best,best_step=score,copy.deepcopy(model.state_dict()),step
            history.append(dict(step=step,loss=float(loss),selection_rmse=score))
            print("matched task MLP",history[-1],flush=True)
    torch.save(dict(state_dict=best,kind="mlp_matched",space="r2",config=registration,best_step=best_step),a.output_dir/"model.pt")
    if not all(sha256(Path(f))==h for f,h in files.items()): raise ValueError("Frozen source changed during training")
    result=dict(status="single_candidate_development_training_complete",config=registration,history=history,
        best_step=best_step,selection_rmse=best_score,parameters=sum(x.numel() for x in model.parameters()),
        training_seconds=time.monotonic()-tic,checkpoint_sha256=sha256(a.output_dir/"model.pt"),source_files_unchanged=True)
    (a.output_dir/"metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print("metrics:",a.output_dir/"metrics.json",flush=True)


if __name__=="__main__":main()
