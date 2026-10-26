"""Frozen all-path residual calibration on step20 reference-policy trajectories.

NOT autonomous rollouts of each new model. CP statements, when independence
assumptions hold, concern the registered reference-trajectory distribution only.
No model training/selection, no final RGB test, no new SBC certificate.
"""
import argparse
import copy
import json
from pathlib import Path
import h5py
import numpy as np
import torch

from Aebs.conformal.calibrate_controller import conformal_summary
from Aebs.dino_latent.models import SafetyProjection,PhysicalToLatent
from Aebs.dino_latent.spvc_saved_actor import load_spvc_pair
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.error_tail_metrics import tail_summary,trajectory_tail_summary
from Aebs.dino_latent.audit_cgan_cp_tails import paired_state_residual
from Aebs.dino_latent.benchmark_static_latents import FeatureAutoencoder
from Aebs.dino_latent.benchmark_latent_methods import ARMS
from Aebs.dino_latent.latent_method_models import make_model,predict,temporal_offsets
from Aebs.dino_latent.temporal_world_model import TemporalPredictor,window_indices,rollout
from Aebs.dino_latent.backbone import FrozenDinoV2
from cGAN.taxi_models_and_data import AebsMLPGenerator


def residual_calibration(residual,groups,n,epsilon=.01,beta=.01):
    thresholds=[.05,.1,.15,.1506854683160781]
    ts,scores=trajectory_tail_summary(residual,groups,n,thresholds)
    if not np.isfinite(scores).all(): raise ValueError("Every trajectory needs an eligible scored transition")
    cp=conformal_summary(scores,epsilon,beta)
    return dict(step=tail_summary(residual,thresholds),trajectory_score=ts,
        cp_residual_radius_mps=cp["q_hat"],rank_from_largest=cp["rank_from_largest_l"],
        reference_trajectory_coverage=1-epsilon,per_fixed_path_confidence=cp["achieved_confidence"],
        calibration_interpretation="positive residual radius is normal, NOT a failed safety score",
        coverage_scope="reference-policy trajectories and specified eligible times; NOT each candidate's autonomous closed loop",
        safety_certificate=False),scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration-h5",type=Path,default=Path("results/dino_expanded_v1/20_trajectory_sbc_cp_calibration_v1/trajectories.h5"))
    p.add_argument("--representation",type=Path,default=Path("results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt"))
    p.add_argument("--ppo",type=Path,default=Path("results/dino_expanded_v1/09_R2_ppo/latent_ppo.zip"))
    p.add_argument("--spvc-dir",type=Path,default=Path("results/dino_spvc_minimal_v1"))
    p.add_argument("--final-recheck",type=Path,default=Path("results/dino_spvc_minimal_v1_final_recheck/metrics.json"))
    p.add_argument("--static-dir",type=Path,default=Path("results/static_latent_benchmark_20261018_qcontrol"))
    p.add_argument("--temporal-dir",type=Path,default=Path("results/latent_methods_benchmark_20261018_run1"))
    p.add_argument("--cgan-draws",type=int,default=8)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists() or not 1<=a.cgan_draws<=32: p.error("New output and draws1..32 required")
    if "20_trajectory_sbc_cp_calibration_v1" not in str(a.calibration_h5): p.error("Use retained step20 pool, not development step13")
    torch.set_num_threads(1); torch.manual_seed(7); device=torch.device(a.device)
    temporal=[]
    for folder in (Path("results/transformer_pilot_20261018"),Path("results/transformer_speed_ablation_20261018")):
        config=json.loads((folder/"config.json").read_text())
        if config.get("fresh_CP_data_used") is not False or config.get("test_used") is not False:
            raise ValueError("Earlier model provenance does not exclude calibration/test use")
        for path in sorted(folder.glob("*.pt")):
            temporal.append(dict(name=folder.name+"/"+path.stem,path=path,legacy=True,trained_horizon=3))
    latest_config=json.loads((a.temporal_dir/"config.json").read_text())
    if latest_config.get("fresh_CP_used") is not False: raise ValueError("Latest model used fresh CP pool")
    temporal += [dict(name="latest/"+n,path=a.temporal_dir/(n+".pt"),legacy=False,kind=k,space=s,trained_horizon=10) for n,k,s in ARMS]
    static_names=("r2_frozen_refit_q","pca32","feature_ae32","feature_vae32")
    models=[a.representation,a.ppo,a.spvc_dir/"latent_ppo_spvc.pt",a.spvc_dir/"sbc.pt"]
    models += [a.static_dir/(n+".pt") for n in static_names]+[m["path"] for m in temporal]
    common_meta=json.loads(Path("results/dino_spvc_minimal_v1_common_latent/metrics.json").read_text())
    models += [Path(common_meta["args"][key]) for key in ("legacy_generator","dino_weights")]
    hashes={str(f):sha256(f) for f in models}
    # Freeze the exact candidates BEFORE opening calibration arrays. This file
    # is a pre-inference registration, not falsely claimed pre-collection.
    a.output_dir.mkdir(parents=True)
    config=dict(registered_before_calibration_inference=True,registered_before_collection=False,
        model_sha256=hashes,temporal_paths=[{k:str(v) if isinstance(v,Path) else v for k,v in m.items()} for m in temporal],
        reference_collector_policy="frozen PPO09; NOT these new models",calibration_file=str(a.calibration_h5),
        training=False,model_selection=False,final_RGB_test_used=False,cgan_draws=a.cgan_draws,cgan_seed=7,
        static_paths=list(static_names),epsilon=.01,beta=.01,
        dataset_note="step20 is independent from model fitting, but already used historically for a different frozen SBC calibration; not a newly collected blind test",
        per_path_confidence_is_not_joint_confidence=True,
        future_use="Do not tune candidates using this pool then reuse these CP radii as independent calibration")
    config["script_sha256"]=sha256(Path(__file__))
    config["nonlearned_baselines"]=["persistence/h1","persistence/h3","persistence/h10"]
    (a.output_dir/"frozen_methods.json").write_text(json.dumps(config,indent=2))
    with h5py.File(a.calibration_h5,"r") as f:
        attrs=dict(f.attrs); provenance=json.loads(attrs["provenance_json"])
        if (not attrs.get("complete") or not attrs.get("independent_from_training") or attrs.get("used_for_model_selection",True)
                or attrs.get("split_role")!="calibration" or provenance.get("model_updates") is not False
                or provenance.get("test_used") is not False): raise ValueError("Calibration metadata invalid")
        if provenance["representation_sha256"]!=sha256(a.representation) or provenance["controller_sha256"]!=sha256(a.ppo):
            raise ValueError("Wrong reference collection models")
        raw=f["dino_features"][:]; mask=f["valid"][:].astype(bool)
        ds,vs,aa=(f[k][:] for k in ("distance_m","speed_mps","image_action"))
        ids=f["trajectory_id"].asstr()[:].tolist()
    if len(ids)!=459 or len(set(ids))!=459 or raw.shape!=(*mask.shape,384) or not mask.any(1).all():
        raise ValueError("Need459 distinct complete reference trajectories")
    for arr in (raw,ds,vs,aa):
        if not np.isfinite(arr[mask]).all(): raise ValueError("Nonfinite data")
    adj=mask[:,1:]&mask[:,:-1]
    if (np.abs(ds[:,1:]-(ds[:,:-1]-.05*vs[:,:-1]))[adj]>5e-4).any(): raise ValueError("Nonconsecutive distance")
    expected=np.clip(vs[:,:-1]-.05*np.clip(aa[:,:-1],-3,3),0,3)
    if (np.abs(vs[:,1:]-expected)[adj]>5e-4).any(): raise ValueError("Nonconsecutive speed")
    calibration_hash=sha256(a.calibration_h5)
    config["calibration_sha256"]=calibration_hash;config["collection_provenance"]=provenance
    rep=torch.load(a.representation,map_location="cpu");scale=float(rep["distance_scale_m"])
    projection,q=SafetyProjection().eval(),PhysicalToLatent().eval()
    projection.load_state_dict(rep["projection_state_dict"]);q.load_state_dict(rep["physical_to_latent_state_dict"])
    base,saved,_,identity=load_spvc_pair(a.representation,a.ppo,a.spvc_dir,a.final_recheck)
    actor=saved.policy.controller_net.eval();config["final_actor_identity"]=identity
    @torch.no_grad()
    def actions(z,v):
        z=torch.as_tensor(z,dtype=torch.float32);v=torch.as_tensor(np.array(v,copy=True),dtype=torch.float32)
        shape=v.shape;obs=torch.cat((z,v[...,None]),-1).reshape(-1,33)
        return torch.cat([actor(obs[i:i+8192]).clamp(-3,3).reshape(-1) for i in range(0,len(obs),8192)]).numpy().reshape(shape)
    with torch.no_grad():
        f=(torch.from_numpy(np.nan_to_num(raw))-rep["feature_mean"])/rep["feature_std"]
        z=projection(f.flatten(0,1)).reshape(*mask.shape,32)
    ep,tt=np.where(mask);d,v=ds[ep,tt],vs[ep,tt]
    real=z[ep,tt]; x=f[ep,tt]
    reference_actions=actions(real,v)
    output=dict(status="running",config=config,paths={})
    scores_all={}; residual_all={}
    def add(name,residual,groups,scope,eligible_times=None):
        r,s=residual_calibration(residual,groups,len(mask));r["scope"]=scope
        if eligible_times is not None: r["eligible_times"]=eligible_times
        output["paths"][name]=r;scores_all[name]=s
        residual_all[name]=np.asarray(residual)
        np.savez_compressed(a.output_dir/(name.replace('/','__')+".npz"),residual=residual,trajectory_index=groups,trajectory_scores=s)
        (a.output_dir/"metrics.json").write_text(json.dumps(output,indent=2,allow_nan=False))
        print(name,"CP-radius=%.9f MAE=%.9f n=%d"%(r["cp_residual_radius_mps"],r["step"]["mae"],len(residual)),flush=True)

    with torch.no_grad(): original_proxy=q(torch.from_numpy(d[:,None])/scale)
    old_i=base.predict(np.column_stack((real.numpy(),v)),deterministic=True)[0].reshape(-1)
    old_p=base.predict(np.column_stack((original_proxy.numpy(),v)),deterministic=True)[0].reshape(-1)
    replay=float(np.max(np.abs(old_i-aa[ep,tt])))
    if replay>1e-4: raise ValueError("Reference actor does not replay recorded actions")
    config["recorded_action_replay_max_abs"]=replay
    add("PPO09_R2",paired_state_residual(d,v,old_i,old_p)[:,1],ep,"reference collector, own image versus own q, all valid current states")
    aq=actions(original_proxy,v)
    add("minimal_R2",paired_state_residual(d,v,reference_actions,aq)[:,1],ep,"final minimal actor, own image versus original q; reference-policy state distribution")
    coords=torch.load(a.static_dir/"pca_coordinates.pt",map_location="cpu")
    for name in static_names:
        ck=torch.load(a.static_dir/(name+".pt"),map_location="cpu")
        proxy=PhysicalToLatent().eval();proxy.load_state_dict(ck["q_state_dict"])
        with torch.no_grad():
            latent=proxy(torch.from_numpy(d[:,None])/scale)*ck["latent_std"]+ck["latent_mean"]
            if name=="r2_frozen_refit_q": iz,pz=real,latent
            elif name=="pca32":
                iz=projection(((x-coords["mean"])@coords["basis"])@coords["basis"].T+coords["mean"])
                pz=projection(latent@coords["basis"].T+coords["mean"])
            else:
                codec=FeatureAutoencoder(name=="feature_vae32").eval();codec.load_state_dict(ck["codec_state_dict"])
                iz=projection(codec.decoder(codec.encode(x)[0]));pz=projection(codec.decoder(latent))
        ai,ap=actions(iz,v),actions(pz,v)
        add(name+"/own_image",paired_state_residual(d,v,ai,ap)[:,1],ep,"same codec image versus q; no new PPO")
        if name in ("pca32","feature_ae32","feature_vae32"):
            add(name+"/R2_anchor",paired_state_residual(d,v,reference_actions,ap)[:,1],ep,"same final actor; original R2 image anchor protects against common-mode codec distortion")

    # Every saved temporal arm: one-step with complete ends plus its trained
    # maximum horizon (3 for earlier models,10 for the latest suite).
    for horizon in (1,3,10):
        windows=window_indices(mask,np.arange(len(mask)),3,horizon)
        target_z=z[windows[:,0],windows[:,1]+horizon]
        target_v=vs[windows[:,0],windows[:,1]+horizon];target_d=ds[windows[:,0],windows[:,1]+horizon]
        r=paired_state_residual(target_d,target_v,actions(target_z,target_v),actions(z[windows[:,0],windows[:,1]],target_v))[:,1]
        add("persistence/h"+str(horizon),r,windows[:,0],"repeat last observed image latent; same frozen final actor",
            "context3 and target offset%d"%horizon)
    normalized_feature=f.to(device);normalized_z=z.to(device)
    speed=torch.from_numpy(np.nan_to_num(vs)).to(device);control=torch.from_numpy(np.nan_to_num(aa)).to(device)
    latest_coords=torch.load(a.temporal_dir/"coordinates.pt",map_location="cpu")
    pca=((f-latest_coords["pca_mean"])@latest_coords["pca_basis"]).to(device)
    gp=copy.deepcopy(projection).to(device).eval()
    for spec in temporal:
        checkpoint=torch.load(spec["path"],map_location="cpu")
        if spec["legacy"]:
            native=checkpoint["native"];arm=checkpoint.get("arm",{})
            dimension=32 if native=="latent32" else 384
            model=TemporalPredictor(dimension,checkpoint["kind"],speed_mode=arm.get("speed_mode","concat"),use_action=arm.get("use_action",True)).to(device)
            source=normalized_z if dimension==32 else normalized_feature
            m,s=[y.to(device) for y in checkpoint["normalization"][native]]
        else:
            native=spec["space"];model=make_model(spec["kind"]).to(device)
            source=normalized_z if native=="r2" else pca
            m,s=[y.to(device) for y in latest_coords["normalization"][native]]
        model.load_state_dict(checkpoint["state_dict"]);model.eval()
        for horizon in (1,spec["trained_horizon"]):
            w=window_indices(mask,np.arange(len(mask)),3,horizon)
            co,_,ao=[torch.as_tensor(o,device=device) for o in temporal_offsets(3,horizon)]
            values=[]
            for start in range(0,len(w),1024):
                ww=torch.as_tensor(w[start:start+1024],device=device)
                ix=(ww[:,0,None],ww[:,1,None]+co)
                with torch.no_grad():
                    hist=(source[ix]-m)/s;vv=speed[ix];ac=control[ww[:,0,None],ww[:,1,None]+ao]
                    pred=rollout(model,hist,vv,ac,horizon) if spec["legacy"] else predict(model,hist,vv,ac,horizon)[0]
                    pred=pred[:,-1]*s+m
                    if spec["legacy"] and dimension==384: pred=gp(pred)
                    elif not spec["legacy"] and native=="pca":
                        pred=projection(pred.cpu()@latest_coords["pca_basis"].T+latest_coords["pca_mean"])
                    values.append(pred.cpu())
            predicted=torch.cat(values);true=z[w[:,0],w[:,1]+horizon]
            dv=vs[w[:,0],w[:,1]+horizon];dd=ds[w[:,0],w[:,1]+horizon]
            r=paired_state_residual(dd,dv,actions(true,dv),actions(predicted,dv))[:,1]
            add(spec["name"]+"/h"+str(horizon),r,w[:,0],
                "forecast representation versus target image, same final actor; recorded future action query, not autonomous rollout",
                "context3, target=context_end+%d; first %d target frames not scored"%(horizon,2+horizon))

    # Frozen cGAN baseline in the SAME R2 coordinates/actor. The original native
    # trajectory image path cannot be recovered from a feature-only H5.
    common=json.loads(Path("results/dino_spvc_minimal_v1_common_latent/metrics.json").read_text())
    generator_path=Path(common["args"]["legacy_generator"]);weights=Path(common["args"]["dino_weights"])
    if sha256(generator_path)!=common["hashes"]["legacy_generator"] or sha256(weights)!=common["hashes"]["dino_weights"]:
        raise ValueError("cGAN/DINO assets changed")
    gen=AebsMLPGenerator(4,1).to(device).eval();gen.load_state_dict(torch.load(generator_path,map_location=device))
    backbone=FrozenDinoV2(Path(common["args"]["dino_repo"]),weights,device).eval()
    legacy_meta=json.loads(Path("results/dino_spvc_minimal_v1_paired_error_retry1/metrics.json").read_text())
    legacy_scale=float(legacy_meta["legacy_scale_m"])
    draws=np.concatenate((np.zeros((1,4),dtype=np.float32),np.random.default_rng(7).uniform(-1,1,(a.cgan_draws,4)).astype(np.float32)))
    unique,inverse=np.unique(d,return_inverse=True);k=len(draws)
    generated=np.empty((len(unique)*k,32),np.float32)
    mean,std=rep["feature_mean"].to(device),rep["feature_std"].to(device)
    for start in range(0,len(generated),64):
        ix=np.arange(start,min(start+64,len(generated)))
        with torch.no_grad():
            y=torch.from_numpy((unique[ix//k,None]/legacy_scale).astype(np.float32)).to(device)
            zz=torch.from_numpy(draws[ix%k]).to(device)
            emb=gp((backbone((gen(zz,y)+1)*.5).float()-mean)/std)
            generated[ix]=emb.cpu().numpy()
        if start==0 or (start+64)%8192==0 or ix[-1]+1==len(generated):
            print("cGAN held-pool encodings",int(ix[-1]+1),"/",len(generated),flush=True)
    gz=generated.reshape(len(unique),k,32)[inverse]
    ga=actions(gz,np.broadcast_to(v[:,None],gz.shape[:2]))
    r=paired_state_residual(d[:,None],v[:,None],reference_actions[:,None],ga)[...,1]
    add("cGAN_shared_R2_actor/z0",r[:,0],ep,"cGAN generation->same frozen R2 encoder/final actor; not legacy native actor")
    add("cGAN_shared_R2_actor/fixed_mc%d"%a.cgan_draws,r[:,1:].reshape(-1),np.repeat(ep,a.cgan_draws),
        "maximum over eligible states AND fixed generator draws; covers this registered finite draw set, not all z")
    output["unavailable_paths"]={"cGAN_native_image_trajectory":"No raw per-frame RGB saved in retained trajectory H5. Native cGAN/state_net/PPO checked on static RGB in prior tail audit; not fabricated as a trajectory CP result."}
    # Simultaneous scalar bound: calibrate per-trajectory maximum over ALL
    # registered path/horizon/draw cases, rather than falsely giving every
    # separate99%-confidence statement a joint99%-confidence interpretation.
    joint_scores=np.max(np.stack(list(scores_all.values())),axis=0)
    joint_cp=conformal_summary(joint_scores,.01,.01)
    output["joint_reference_distribution_radius"]=dict(q_hat_mps=joint_cp["q_hat"],
        achieved_confidence=joint_cp["achieved_confidence"],target_joint_trajectory_coverage=.99,
        calibrated_score="per-reference-trajectory maximum across all registered paths, horizons and fixed cGAN draws",
        safety_certificate=False)
    np.savez_compressed(a.output_dir/"all_trajectory_scores.npz",trajectory_ids=np.array(ids),
        path_names=np.array(list(scores_all)),scores=np.stack(list(scores_all.values())),joint_scores=joint_scores)
    for path,digest in hashes.items():
        if sha256(Path(path))!=digest: raise ValueError("Frozen candidate changed")
    if sha256(a.calibration_h5)!=calibration_hash: raise ValueError("Calibration file changed")
    output.update(status="all_frozen_path_reference_residual_calibration_complete_not_closed_loop_safety",models_unchanged=True,
        path_horizon_cases=len(scores_all),calibration_trajectories=len(mask),original_RGB_test_used=False)
    (a.output_dir/"frozen_methods.json").write_text(json.dumps(config,indent=2))
    (a.output_dir/"metrics.json").write_text(json.dumps(output,indent=2,allow_nan=False))
    print("[All frozen paths reference-distribution CP residuals]",len(scores_all),"cases; joint q=",joint_cp["q_hat"],flush=True)
    print("metrics:",a.output_dir/"metrics.json",flush=True)


if __name__=="__main__": main()
