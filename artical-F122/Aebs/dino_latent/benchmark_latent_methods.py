"""Frozen-encoder, 9-arm development benchmark; no PPO/SBC/test/CP updates.

Uses only old development trajectories, with a new train/selection/internal-audit
split. Internal audit is NOT globally unseen test data: earlier experiments used
this old pool. Core-mechanism adaptations, not official paper reproductions.
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

from Aebs.dino_latent.models import SafetyProjection, SafetyDecoder, PhysicalToLatent
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.spvc_saved_actor import load_spvc_pair
from Aebs.dino_latent.temporal_world_model import window_indices, advance_speed
from Aebs.dino_latent.run_transformer_comparison import gather, summary
from Aebs.dino_latent.latent_method_models import make_model, predict, split_three, paired_cluster_bootstrap, temporal_offsets


ARMS = (
    ("mlp32", "mlp", "r2"), ("transformer32", "transformer", "r2"),
    ("mlp32_parameter_matched", "mlp_matched", "r2"),
    ("transformer32_no_speed", "no_speed", "r2"),
    ("gru32", "gru", "r2"), ("rssm32", "rssm", "r2"),
    ("lifted_linear32", "lifted_linear", "r2"),
    ("transformer32_task_aware", "task_aware", "r2"),
    ("pca32_transformer", "transformer", "pca"),
)
FIELDS = ("latent_rms", "distance_error_m", "distance_reference_gap_m", "action_gap",
          "next_speed_gap_mps", "underbraking")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, default=Path("results/dino_expanded_v1/13_spvc_calibration_trajectories_v1/trajectories.h5"))
    parser.add_argument("--representation", type=Path, default=Path("results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt"))
    parser.add_argument("--ppo", type=Path, default=Path("results/dino_expanded_v1/09_R2_ppo/latent_ppo.zip"))
    parser.add_argument("--spvc-dir", type=Path, default=Path("results/dino_spvc_minimal_v1"))
    parser.add_argument("--final-recheck", type=Path, default=Path("results/dino_spvc_minimal_v1_final_recheck/metrics.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split-seed", type=int, default=71)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    if args.output_dir.exists() or min(args.steps,args.batch_size,args.eval_every) < 1 or args.horizon < 3:
        parser.error("Need new directory, positive settings, horizon>=3")
    if "13_spvc_calibration_trajectories_v1" not in str(args.trajectories):
        parser.error("This registered benchmark accepts only old step13 development pool, not fresh CP/test")
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    files = (args.trajectories,args.representation,args.ppo,args.final_recheck,
             args.spvc_dir/"latent_ppo_spvc.pt",args.spvc_dir/"sbc.pt",args.spvc_dir/"metrics.json")
    frozen = {str(f):sha256(f) for f in files}
    rep = torch.load(args.representation,map_location="cpu")
    projection,decoder,q = SafetyProjection(),SafetyDecoder(),PhysicalToLatent()
    for model,key in ((projection,"projection_state_dict"),(decoder,"auxiliary_decoder_state_dict"),(q,"physical_to_latent_state_dict")):
        model.load_state_dict(rep[key]); model.to(device).eval()
        for par in model.parameters(): par.requires_grad_(False)
    _,student,_,actor_identity = load_spvc_pair(args.representation,args.ppo,args.spvc_dir,args.final_recheck)
    actor = student.policy.controller_net.to(device).eval()
    for par in actor.parameters(): par.requires_grad_(False)
    scale = float(rep["distance_scale_m"])
    with h5py.File(args.trajectories,"r") as f:
        if not f.attrs.get("complete",False): raise ValueError("Incomplete data")
        raw,valid = f["dino_features"][:],f["valid"][:].astype(bool)
        ds,vs,acts = (f[k][:] for k in ("distance_m","speed_mps","image_action"))
        ids = f["trajectory_id"].asstr()[:].tolist()
        appearances = np.array([x+"|"+y for x,y in zip(f["weather"].asstr()[:],f["color"].asstr()[:])])
    if len(set(ids)) != len(ids): raise ValueError("Duplicate IDs")
    for value in (raw,ds,vs,acts):
        if not np.isfinite(value[valid]).all(): raise ValueError("Nonfinite valid data")
    if raw.shape != (*valid.shape,384): raise ValueError("Wrong feature shape")
    adjacent = valid[:,1:] & valid[:,:-1]
    if (np.abs(ds[:,1:]-(ds[:,:-1]-.05*vs[:,:-1]))[adjacent] > 5e-4).any():
        raise ValueError("Nonconsecutive distance records")
    expected_v = np.clip(vs[:,:-1]-.05*np.clip(acts[:,:-1],-3,3),0,3)
    if (np.abs(vs[:,1:]-expected_v)[adjacent] > 5e-4).any(): raise ValueError("Nonconsecutive speed records")
    partitions = dict(zip(("train","selection","internal_audit"),split_three(len(ids),args.split_seed)))
    windows = {k:window_indices(valid,v,3,args.horizon) for k,v in partitions.items()}
    if any(not len(w) for w in windows.values()): raise ValueError("Empty partition")
    feature = (torch.from_numpy(np.nan_to_num(raw)).to(device)-rep["feature_mean"].to(device))/rep["feature_std"].to(device)
    with torch.no_grad(): z = projection(feature.flatten(0,1)).reshape(*valid.shape,32)
    speed,action,distance = [torch.from_numpy(np.nan_to_num(x)).to(device) for x in (vs,acts,ds)]
    train = partitions["train"]
    train_mask = torch.as_tensor(valid[train],device=device)
    train_features = feature[train][train_mask]
    # Fixed train-only PCA via covariance eigh (no randomized validation fitting).
    fm = train_features.mean(0)
    centered = train_features-fm
    eigenvalues,eigenvectors = torch.linalg.eigh(centered.T@centered/(len(centered)-1))
    basis = eigenvectors[:,-32:].flip(-1)
    pca = (feature-fm)@basis
    data = dict(r2=z,pca=pca)
    norms = {k:(x[train][train_mask].mean(0),x[train][train_mask].std(0).clamp_min(.01)) for k,x in data.items()}
    args.output_dir.mkdir(parents=True)
    code_files = [Path(__file__),Path(__file__).with_name("latent_method_models.py"),Path(__file__).with_name("temporal_world_model.py")]
    config = dict(status="registered_before_training",arms=[dict(name=n,kind=k,space=s) for n,k,s in ARMS],
        steps=args.steps,batch_size=args.batch_size,horizon=args.horizon,context=3,seed=args.seed,
        split_seed=args.split_seed,train_trajectories=len(train),selection_trajectories=len(partitions["selection"]),
        internal_audit_trajectories=len(partitions["internal_audit"]),window_counts={k:len(w) for k,w in windows.items()},
        source_sha256=frozen,code_sha256={str(f):sha256(f) for f in code_files},actor_identity=actor_identity,
        selection="all-horizon common R2 RMSE on selection trajectories only",
        internal_audit_note="not globally unseen; old pool used in previous development experiments",
        learning_rate=.0003,weight_decay=.0001,task_aux_weights=dict(distance=1.,action=.1),
        rssm_aux_weights=dict(reconstruction=.1,gaussian_kl=.001),
        pca_variance_fraction=float(eigenvalues[-32:].sum()/eigenvalues.sum()),
        future_true_states_in_predictor=False,future_actions="same recorded open-loop sequence for all methods",
        diagnostic_actor_uses_target_true_speed=True,all_models_use_speed_except="transformer32_no_speed",
        matched_steps_not_matched_parameter_count=True,test_used=False,fresh_CP_used=False,
        original_PPO_SBC_frozen=True,official_paper_reproductions=False)
    (args.output_dir/"config.json").write_text(json.dumps(config,indent=2,allow_nan=False))
    (args.output_dir/"split.json").write_text(json.dumps({k:[ids[i] for i in x] for k,x in partitions.items()},indent=2))
    torch.save(dict(normalization=norms,pca_mean=fm,pca_basis=basis),args.output_dir/"coordinates.pt")
    np.savez_compressed(args.output_dir/"windows.npz",**windows)
    co,to,ao = [torch.as_tensor(x,device=device) for x in temporal_offsets(3,args.horizon)]

    def batch(space,rows):
        w = torch.as_tensor(rows,dtype=torch.long,device=device)
        m,s = norms[space]
        return ((gather(data[space],w,co)-m)/s,gather(speed,w,co),gather(action,w,ao),
                (gather(data[space],w,to)-m)/s,w)

    def common(space,value):
        m,s = norms[space]
        value = value*s+m
        return projection(value@basis.T+fm) if space=="pca" else value

    def policy(latents,velocities):
        return actor(torch.cat((latents,velocities[...,None]),-1).flatten(0,1)).reshape_as(velocities).clamp(-3,3)

    @torch.no_grad()
    def select(model,space):
        squares,count = 0.,0
        for start in range(0,len(windows["selection"]),512):
            hist,v,a,_,w = batch(space,windows["selection"][start:start+512])
            value,_ = predict(model,hist,v,a,args.horizon)
            err = common(space,value)-gather(z,w,to)
            squares += float(err.square().sum()); count += err.numel()
        return (squares/count)**.5

    @torch.no_grad()
    def evaluate(model,space,name):
        rows = windows["internal_audit"]
        horizons = [1,3,args.horizon]
        saved = []
        tic = time.monotonic()
        for start in range(0,len(rows),512):
            hist,v,a,_,w = batch(space,rows[start:start+512])
            targets = gather(z,w,to); d = gather(distance,w,to); velocity = gather(speed,w,to)
            if name=="current_q_known_distance": predicted = q(d[...,None]/scale)
            elif name=="persistence32": predicted = z[w[:,0],w[:,1]][:,None].expand(-1,args.horizon,-1)
            else:
                value,_ = predict(model,hist,v,a,args.horizon)
                predicted = common(space,value)
            ap,ar = policy(predicted,velocity),policy(targets,velocity)
            values = ((predicted-targets).square().mean(-1).sqrt(),decoder(predicted).squeeze(-1)*scale-d,
                      (decoder(predicted)-decoder(targets)).squeeze(-1)*scale,ap-ar,
                      advance_speed(velocity,ap)-advance_speed(velocity,ar),F.relu(ar-ap))
            saved.append(torch.stack([torch.stack([val[:,h-1] for val in values],-1) for h in horizons],1).cpu().numpy())
        errors = np.concatenate(saved)
        np.savez_compressed(args.output_dir/(name+"_audit.npz"),errors=errors,windows=rows,
                            horizons=np.array(horizons),fields=np.array(FIELDS))
        result = {}
        for j,h in enumerate(horizons):
            d = ds[rows[:,0],rows[:,1]+h]; v = vs[rows[:,0],rows[:,1]+h]
            masks = {"all":np.ones(len(rows),bool),"near_obstacle_d_le_8_v_ge_0.5":(d<=8)&(v>=.5),
                     "low_speed_v_lt_0.5":v<.5,"middle_speed_0.5_to_1.5":(v>=.5)&(v<1.5),"high_speed_ge_1.5":v>=1.5}
            masks.update({"appearance:"+g:appearances[rows[:,0]]==g for g in np.unique(appearances[rows[:,0]])})
            groups = {g:dict(count=int(mask.sum()),**{k:summary(errors[mask,j,i]) for i,k in enumerate(FIELDS)})
                      for g,mask in masks.items() if mask.any()}
            e = errors[:,j,4]
            worst = int(np.abs(e).argmax()); ep,t = rows[worst]
            groups["worst_physical_gap"] = dict(trajectory_id=ids[ep],context_end=int(t),target_step=h,
                distance_m=float(d[worst]),speed_mps=float(v[worst]),appearance=str(appearances[ep]),
                signed_next_speed_gap_mps=float(e[worst]))
            result[str(h)] = groups
        return result,errors,time.monotonic()-tic

    results,raw_errors = {},{}
    for name in ("persistence32","current_q_known_distance"):
        metrics,errors,seconds = evaluate(None,"r2",name)
        results[name] = dict(metrics=metrics,evaluation_seconds=seconds); raw_errors[name]=errors
    for name,kind,space in ARMS:
        torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
        model = make_model(kind).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
        model.eval(); best_score = select(model,space); best = copy.deepcopy(model.state_dict()); best_step = 0
        history = []; tic = time.monotonic()
        for step in range(1,args.steps+1):
            model.train()
            chosen = rng.integers(len(windows["train"]),size=args.batch_size)
            hist,v,a,target,w = batch(space,windows["train"][chosen])
            value,aux = predict(model,hist,v,a,args.horizon,target)
            native_loss = F.mse_loss(value,target)
            loss = native_loss+aux
            if kind=="task_aware":
                pred = common(space,value)
                # Labels are training-only targets; predictor still gets no future state.
                # Future diagnostic speeds are computed from context+queried controls.
                vv = v[:,-1]; velocities=[]
                for h in range(args.horizon):
                    vv=advance_speed(vv,a[:,2+h]); velocities.append(vv)
                vv = torch.stack(velocities,1)
                real_z = gather(z,w,to)
                with torch.no_grad(): real_a = policy(real_z,vv)
                loss = loss+F.mse_loss(decoder(pred).squeeze(-1),gather(distance,w,to)/scale)
                loss = loss+.1*F.mse_loss(policy(pred,vv)/3,real_a/3)
            if not torch.isfinite(loss): raise ValueError("Nonfinite loss: "+name)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
            if step==1 or step%args.eval_every==0 or step==args.steps:
                model.eval(); score = select(model,space)
                if score<best_score: best_score,best,best_step=score,copy.deepcopy(model.state_dict()),step
                history.append(dict(step=step,loss=float(loss),native_mse=float(native_loss),selection_rmse=score))
                print("%s step=%d selection_RMSE=%.6f loss=%.6f"%(name,step,score,float(loss)),flush=True)
        training_seconds=time.monotonic()-tic
        model.load_state_dict(best); model.eval()
        torch.save(dict(state_dict=best,kind=kind,space=space,config=config,best_step=best_step),args.output_dir/(name+".pt"))
        metrics,errors,seconds=evaluate(model,space,name)
        results[name]=dict(metrics=metrics,best_step=best_step,selection_rmse=best_score,history=history,
            parameters=sum(p.numel() for p in model.parameters()),training_seconds=training_seconds,evaluation_seconds=seconds)
        raw_errors[name]=errors
        (args.output_dir/"metrics.json").write_text(json.dumps(dict(status="running",config=config,results=results),indent=2,allow_nan=False))
        print("finished "+name,flush=True)
    paired={}
    audit_windows=windows["internal_audit"]
    for name in raw_errors:
        if name in ("transformer32","current_q_known_distance"): continue
        paired[name]={}
        for j,h in enumerate((1,3,args.horizon)):
            paired[name][str(h)]={k:paired_cluster_bootstrap(np.abs(raw_errors[name][:,j,i]),
                np.abs(raw_errors["transformer32"][:,j,i]),audit_windows[:,0]) for i,k in enumerate(FIELDS) if k!="underbraking"}
    # Explicitly compare the same-architecture changed objective and no-speed ablation.
    for file,digest in frozen.items():
        if sha256(Path(file))!=digest: raise ValueError("Frozen input changed: "+file)
    output=dict(status="development_internal_audit_complete_not_a_safety_certificate",config=config,results=results,
        paired_vs_transformer32=paired,source_files_unchanged=True,
        weights_sha256={name:sha256(args.output_dir/(name+".pt")) for name,_,_ in ARMS},
        total_training_seconds=sum(x.get("training_seconds",0) for x in results.values()))
    (args.output_dir/"metrics.json").write_text(json.dumps(output,indent=2,allow_nan=False))
    print("[Latent methods: old-pool internal audit, NOT formal test or certificate]",flush=True)
    for name,r in results.items():
        for h in (1,args.horizon):
            m=r["metrics"][str(h)]["all"]
            print("%s h%d distance_MAE=%.6f next_speed_MAE=%.6f max=%.6f"%(name,h,m["distance_error_m"]["mae"],
                m["next_speed_gap_mps"]["mae"],m["next_speed_gap_mps"]["max"]),flush=True)
    print("metrics:",args.output_dir/"metrics.json",flush=True)


if __name__=="__main__": main()
