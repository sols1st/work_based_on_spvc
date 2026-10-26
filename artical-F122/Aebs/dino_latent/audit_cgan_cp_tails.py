"""Frozen cGAN/latent absolute-error and historical CP-radius development audit.

No training. Old step13 trajectories only; NOT fresh CP calibration or test.
Different controllers, references and temporal scopes are reported separately.
"""
import argparse
import copy
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from stable_baselines3 import PPO

from Aebs.dino_latent.models import SafetyProjection,SafetyDecoder,PhysicalToLatent
from Aebs.dino_latent.spvc_saved_actor import load_spvc_pair
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.calibrate_transition_contract import next_state_residual
from Aebs.dino_latent.error_tail_metrics import tail_summary,trajectory_tail_summary,align_rows
from Aebs.dino_latent.benchmark_static_latents import FeatureAutoencoder
from Aebs.dino_latent.benchmark_latent_methods import ARMS
from Aebs.dino_latent.latent_method_models import make_model,predict,temporal_offsets
from Aebs.dino_latent.temporal_world_model import window_indices
from Aebs.dino_latent.train_cp_robust_sbc import successor_scenarios,safe_terminal_mask,unsafe_mask
from Aebs.dino_latent.backbone import FrozenDinoV2
from cGAN.taxi_models_and_data import AebsMLPGenerator


def json_write(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False))


def paired_state_residual(distance,speed,image_action,proxy_action):
    # All coordinates must share the MC/speed dimensions before the registered
    # residual routine stacks the exactly-zero distance and speed components.
    return next_state_residual(*np.broadcast_arrays(distance,speed,image_action,proxy_action))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--static-dir",type=Path,default=Path("results/static_latent_benchmark_20261018_qcontrol"))
    p.add_argument("--temporal-dir",type=Path,default=Path("results/latent_methods_benchmark_20261018_run1"))
    p.add_argument("--paired-dir",type=Path,default=Path("results/dino_spvc_minimal_v1_paired_error_retry1"))
    p.add_argument("--common-dir",type=Path,default=Path("results/dino_spvc_minimal_v1_common_latent"))
    p.add_argument("--contract",type=Path,default=Path("results/dino_expanded_v1/14_transition_contract_spvc_v1/contract.json"))
    p.add_argument("--trajectories",type=Path,default=Path("results/dino_expanded_v1/13_spvc_calibration_trajectories_v1/trajectories.h5"))
    p.add_argument("--representation",type=Path,default=Path("results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt"))
    p.add_argument("--ppo",type=Path,default=Path("results/dino_expanded_v1/09_R2_ppo/latent_ppo.zip"))
    p.add_argument("--spvc-dir",type=Path,default=Path("results/dino_spvc_minimal_v1"))
    p.add_argument("--final-recheck",type=Path,default=Path("results/dino_spvc_minimal_v1_final_recheck/metrics.json"))
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--cgan-audit-draws",type=int,default=8)
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists() or not 1<=a.cgan_audit_draws<=32: p.error("New output and draws1..32 required")
    if "13_spvc_calibration_trajectories_v1" not in str(a.trajectories): p.error("Only old development trajectories allowed")
    torch.set_num_threads(1); torch.manual_seed(7); device=torch.device(a.device)
    cp=json.loads(a.contract.read_text()); qhat=float(cp["conformal"]["q_hat"])
    thresholds=[.01,.05,.1,.15,qhat]
    static_config=json.loads((a.static_dir/"config.json").read_text())
    temporal_config=json.loads((a.temporal_dir/"config.json").read_text())
    paired_meta=json.loads((a.paired_dir/"metrics.json").read_text())
    common_meta=json.loads((a.common_dir/"metrics.json").read_text())
    if paired_meta.get("status")!="development_comparison_complete" or common_meta.get("status")!="development_common_latent_comparison_complete":
        raise ValueError("Prior paired data incomplete")
    if not all(sha256(Path(path))==digest for config in (static_config,temporal_config) for path,digest in config["source_sha256"].items()):
        raise ValueError("Original benchmark inputs changed")
    if cp["controller_sha256"]!=sha256(a.ppo) or cp["representation_sha256"]!=sha256(a.representation) or cp["calibration_h5_sha256"]!=sha256(a.trajectories):
        raise ValueError("Historical CP is for different files")
    if cp["state_scales"]!={"distance_m":1.,"speed_mps":1.}: raise ValueError("Unexpected CP units")
    # Verify reused inference caches against the current immutable checkpoints.
    for name,key in (("representation","representation"),("saved_actor","dino_saved_actor"),
                     ("saved_barrier","dino_saved_barrier"),("base_controller","dino_controller"),
                     ("legacy_generator","legacy_generator"),("legacy_data","legacy_data")):
        if common_meta["hashes"][name]!=paired_meta["hashes"][key]: raise ValueError("Paired caches do not match")
    for key in ("legacy_generator","legacy_controller","legacy_state_net","legacy_data"):
        if sha256(Path(paired_meta["args"][key]))!=paired_meta["hashes"][key]: raise ValueError("Cached legacy checkpoint changed: "+key)
    files=[a.contract,a.trajectories,a.representation,a.ppo,a.final_recheck,a.spvc_dir/"latent_ppo_spvc.pt",a.spvc_dir/"sbc.pt",
           a.static_dir/"metrics.json",a.temporal_dir/"metrics.json",a.common_dir/"metrics.json",a.common_dir/"samples.npz",
           a.paired_dir/"metrics.json",a.paired_dir/"samples.npz",a.static_dir/"coordinates.pt" if (a.static_dir/"coordinates.pt").exists() else a.static_dir/"pca_coordinates.pt"]
    files += [a.static_dir/(n+".pt") for n in ("r2_frozen_refit_q","pca32","feature_ae32","feature_vae32")]
    files += [a.temporal_dir/(n+".pt") for n,_,_ in ARMS]
    frozen={str(f):sha256(f) for f in files}
    rep=torch.load(a.representation,map_location="cpu"); scale=float(rep["distance_scale_m"])
    projection,decoder,q=SafetyProjection(),SafetyDecoder(),PhysicalToLatent()
    for net,key in ((projection,"projection_state_dict"),(decoder,"auxiliary_decoder_state_dict"),(q,"physical_to_latent_state_dict")):
        net.load_state_dict(rep[key]); net.eval()
        for par in net.parameters(): par.requires_grad_(False)
    base,saved,_,identity=load_spvc_pair(a.representation,a.ppo,a.spvc_dir,a.final_recheck)
    actor=saved.policy.controller_net.eval()
    for par in actor.parameters(): par.requires_grad_(False)

    @torch.no_grad()
    def actions(latents,velocities,network=actor):
        latents=torch.as_tensor(latents,dtype=torch.float32); velocities=torch.as_tensor(velocities,dtype=torch.float32)
        shape=velocities.shape
        obs=torch.cat((latents,velocities[...,None]),-1).reshape(-1,33)
        chunks=[network(obs[i:i+8192]).clamp(-3,3).reshape(-1) for i in range(0,len(obs),8192)]
        return torch.cat(chunks).numpy().reshape(shape)

    a.output_dir.mkdir(parents=True)
    config=dict(registered_before_audit=True,training=False,test_used=False,fresh_CP_used=False,
        historical_q_hat_mps=qhat,thresholds_mps=thresholds,source_sha256=frozen,actor_identity=identity,
        residual_sign="image-path next speed MINUS proxy/predicted-path next speed (historical CP convention)",
        cgan_zero_draw_scope="all old459 trajectories; shared R2 encoder and final minimal actor",
        cgan_uniform_scope="only 69 internal audit trajectories, fixed draws; no best-of-K",
        cgan_audit_draws=a.cgan_audit_draws,cgan_seed=7,
        claim="old development diagnostics, not fresh conformal recalibration or an end-to-end safety certificate")
    json_write(a.output_dir/"config.json",config)
    out=dict(status="running",config=config)
    def save(): json_write(a.output_dir/"metrics.json",out)
    save()

    # A. Same 2460 static images: add original cGAN caches, preserving interfaces.
    audit_rows=np.asarray(json.loads((a.static_dir/"split.json").read_text())["internal_audit"],dtype=np.int64)
    cs=np.load(a.common_dir/"samples.npz"); ps=np.load(a.paired_dir/"samples.npz")
    ci=align_rows(audit_rows,cs["feature_row"]); pi=align_rows(audit_rows,ps["feature_row"])
    if not np.array_equal(cs["filename"][ci],ps["filename"][pi]): raise ValueError("Static RGB identities differ")
    if not np.allclose(cs["distance_m"][ci],ps["distance_m"][pi],atol=1e-9,rtol=0): raise ValueError("Static distances differ")
    velocities=ps["speeds_mps"]; valid=ps["nonterminal"][pi]
    real_z=cs["z_real"][ci]; true_d=cs["distance_m"][ci]
    ar=actions(np.repeat(real_z[:,None,:],len(velocities),1),np.broadcast_to(velocities,valid.shape))
    if not np.allclose(ar,ps["action_dino_image"][pi],atol=1e-5,rtol=1e-5): raise ValueError("Final actor/cache replay mismatch")
    # Encode each unique generated latent/speed only once; no new DINO inference here.
    generated=cs["z_generated_unique"]; nv=len(velocities)
    ga=actions(np.repeat(generated[:,:,None,:],nv,2),np.broadcast_to(velocities,(*generated.shape[:2],nv)))
    image_to_unique=cs["image_to_unique_distance"][ci]
    shared_ga=ga[image_to_unique]
    static=dict(images=len(ci),nonterminal_state_pairs=int(valid.sum()),legacy_policy_kind=paired_meta["legacy_policy_kind"],
        definitions=dict(native_legacy="original cGAN/state_net/PPO proxy vs its own downsampled real-image path",
            shared_controller="cGAN->fixed DINO/R2 vs real RGB->DINO/R2, same final minimal actor",
            latent="same frozen task-trained R2 coordinates; favors the R2 method; not neutral image quality"),
        common_latent={},physical={},distance={},native_pixel_RMS={})
    for label,sl in (("z_zero",slice(0,1)),("uniform_mc32",slice(1,None))):
        static["common_latent"]["cgan_"+label]=tail_summary(cs["cgan_gap"][ci,sl],unit="common R2 vector RMS")
        static["native_pixel_RMS"][label]=tail_summary(cs["cgan_pixel_gap"][ci,sl],unit="normalized grayscale pixel RMS; NOT latent units")
        cg=shared_ga[:,sl]
        r=paired_state_residual(true_d[:,None,None],velocities[None,None,:],ar[:,None,:],cg)[...,1]
        selected=r.transpose(0,2,1)[valid]
        static["physical"]["cgan_shared_"+label]=tail_summary(selected,thresholds)
        for mode in ("01","signed"):
            oldr=paired_state_residual(true_d[:,None,None],velocities[None,None,:],
                ps["action_legacy_image_"+mode][pi,None,:],ps["action_legacy_proxy"][pi,sl,:])[...,1]
            static["physical"]["cgan_native_"+mode+"_"+label]=tail_summary(oldr.transpose(0,2,1)[valid],thresholds)
    static["common_latent"]["original_R2"]=tail_summary(cs["dino_gap"][ci],unit="common R2 vector RMS")
    for name,values in (("legacy_real_image_01",ps["distance_legacy_image_01"][pi]),
                        ("legacy_real_image_signed",ps["distance_legacy_image_signed"][pi]),
                        ("R2_real_image",ps["distance_dino_image"][pi]),
                        ("legacy_proxy_mc32",ps["distance_legacy_proxy"][pi,1:]),
                        ("R2_proxy",ps["distance_dino_proxy"][pi])):
        target=true_d[:,None] if values.ndim==2 else true_d
        static["distance"][name]=tail_summary(values-target,[.1,.5,1.],unit="m")
    # Re-evaluate new codecs and all speed-grid pairs, not only per-image averages.
    manifest_path=next(Path(x) for x in static_config["source_sha256"] if Path(x).name=="cache_manifest.json")
    records=json.loads(manifest_path.read_text())["records"]
    feature_path=manifest_path.parent/"features.h5"
    with h5py.File(feature_path,"r") as f: static_raw=torch.from_numpy(f["features"][:][audit_rows])
    static_x=(static_raw-rep["feature_mean"])/rep["feature_std"]
    coord=torch.load(a.static_dir/"pca_coordinates.pt",map_location="cpu")
    codec_states={}; q_models={}; codecs={}
    for name in ("r2_frozen_refit_q","pca32","feature_ae32","feature_vae32"):
        state=torch.load(a.static_dir/(name+".pt"),map_location="cpu"); codec_states[name]=state
        net=PhysicalToLatent().eval(); net.load_state_dict(state["q_state_dict"]); q_models[name]=net
        if name.startswith("feature_"):
            net=FeatureAutoencoder(name=="feature_vae32").eval();net.load_state_dict(state["codec_state_dict"]);codecs[name]=net

    @torch.no_grad()
    def codec_paths(name,x,d):
        if name=="original_R2": return projection(x),q(d[:,None]/scale)
        state=codec_states[name]
        proxy=q_models[name](d[:,None]/scale)*state["latent_std"]+state["latent_mean"]
        if name=="r2_frozen_refit_q": return projection(x),proxy
        if name=="pca32":
            image=(x-coord["mean"])@coord["basis"]
            return projection(image@coord["basis"].T+coord["mean"]),projection(proxy@coord["basis"].T+coord["mean"])
        codec=codecs[name]
        return projection(codec.decoder(codec.encode(x)[0])),projection(codec.decoder(proxy))

    for name in ("original_R2","r2_frozen_refit_q","pca32","feature_ae32","feature_vae32"):
        zi,zq=codec_paths(name,static_x,torch.tensor(true_d,dtype=torch.float32))
        ai=actions(zi[:,None,:].expand(-1,nv,-1),np.broadcast_to(velocities,valid.shape))
        aq=actions(zq[:,None,:].expand(-1,nv,-1),np.broadcast_to(velocities,valid.shape))
        for reference,reference_actions in (("own_image",ai),("original_R2_image",ar)):
            r=paired_state_residual(true_d[:,None],velocities[None,:],reference_actions,aq)[...,1]
            static["physical"][name+"_"+reference]=tail_summary(r[valid],thresholds)
    out["static_same_images"]=static;save();print("static cGAN/latent tails complete",flush=True)

    # B. Existing horizon1/3/10 temporal audit, all methods and exact tail counts.
    temporal={}
    for file in sorted(a.temporal_dir.glob("*_audit.npz")):
        sample=np.load(file); name=file.name[:-len("_audit.npz")]; row={}
        for j,h in enumerate(sample["horizons"]):
            signed=-sample["errors"][:,j,4]  # previous benchmark was proxy-minus-image
            traj,maxima=trajectory_tail_summary(signed,sample["windows"][:,0],459,thresholds)
            row[str(int(h))]=dict(step=tail_summary(signed,thresholds),trajectory_max=traj)
        temporal[name]=row
    out["temporal_saved_internal_audit"]=temporal;save()

    # C. Same old459 physical states: reproduce old CP and test final/current codecs.
    with h5py.File(a.trajectories,"r") as f:
        if not f.attrs.get("complete",False): raise ValueError("Incomplete trajectories")
        raw=f["dino_features"][:]; mask=f["valid"][:].astype(bool)
        ds,vs,recorded_actions=(f[k][:] for k in ("distance_m","speed_mps","image_action"))
        trajectory_ids=f["trajectory_id"].asstr()[:].tolist()
    flat=np.flatnonzero(mask); ep,tt=np.unravel_index(flat,mask.shape)
    d=ds.reshape(-1)[flat];v=vs.reshape(-1)[flat]
    x=(torch.from_numpy(raw.reshape(-1,384)[flat])-rep["feature_mean"])/rep["feature_std"]
    with torch.no_grad(): z=projection(x); nominal=q(torch.from_numpy(d[:,None])/scale)
    old_observation=np.column_stack((z.numpy(),v)).astype(np.float32)
    old_nominal=np.column_stack((nominal.numpy(),v)).astype(np.float32)
    old_ai=base.predict(old_observation,deterministic=True)[0].reshape(-1)
    old_aq=base.predict(old_nominal,deterministic=True)[0].reshape(-1)
    replay=float(np.max(np.abs(old_ai-recorded_actions.reshape(-1)[flat])))
    old_r=paired_state_residual(d,v,old_ai,old_aq)[:,1]
    old_traj,old_scores=trajectory_tail_summary(old_r,ep,len(mask),thresholds)
    difference=abs(float(np.nanmax(old_scores))-qhat)
    if difference>1e-6 or replay>1e-4: raise ValueError("Historical CP reproduction failed: %s"%dict(q_difference=difference,replay=replay))
    actual=dict(trajectories=len(mask),steps=len(flat),reference="old image-collection state distribution; not autonomous new-model rollouts",
        legacy_CP_reproduction=dict(step=tail_summary(old_r,thresholds),trajectory_max=old_traj,
            q_hat_recorded=qhat,q_hat_reproduced=float(np.nanmax(old_scores)),absolute_difference=difference,
            recorded_image_action_replay_max_abs=replay),variants={})
    raw_residual={"legacy_PPO09":old_r}; scores_saved={"legacy_PPO09":old_scores}
    final_ar=actions(z,v)
    for name in ("original_R2","r2_frozen_refit_q","pca32","feature_ae32","feature_vae32"):
        zi,zq=codec_paths(name,x,torch.from_numpy(d))
        ai,aq=actions(zi,v),actions(zq,v)
        row={}
        for reference,reference_actions in (("own_image",ai),("original_R2_image",final_ar)):
            r=paired_state_residual(d,v,reference_actions,aq)[:,1]
            ts,score=trajectory_tail_summary(r,ep,len(mask),thresholds)
            row[reference]=dict(step=tail_summary(r,thresholds),trajectory_max=ts)
            if reference=="own_image": raw_residual[name]=r;scores_saved[name]=score
        actual["variants"][name]=row
        print("trajectory residual",name,"max=%.9f"%row["own_image"]["step"]["max_abs"],flush=True)
    np.savez_compressed(a.output_dir/"static_trajectory_residuals.npz",trajectory_index=ep,time=tt,
        distance_m=d,speed_mps=v,**raw_residual)
    out["old459_trajectory_residuals"]=actual;save()

    # D. Retest every predictor near trajectory ends: only ONE target frame is
    # required, so final nine valid target frames are no longer dropped.
    audit_ids=np.array(sorted(set(np.load(a.temporal_dir/"windows.npz")["internal_audit"][:,0])))
    w=window_indices(mask,np.arange(len(mask)),3,1)
    coordinates=torch.load(a.temporal_dir/"coordinates.pt",map_location="cpu")
    with torch.no_grad():
        f=(torch.from_numpy(np.nan_to_num(raw))-rep["feature_mean"])/rep["feature_std"]
        all_z=projection(f.flatten(0,1)).reshape(*mask.shape,32)
        all_pca=(f-coordinates["pca_mean"])@coordinates["pca_basis"]
    data={"r2":all_z.to(device),"pca":all_pca.to(device)}
    observed_v=torch.from_numpy(np.nan_to_num(vs)).to(device)
    observed_a=torch.from_numpy(np.nan_to_num(recorded_actions)).to(device)
    co,_,ao=[torch.as_tensor(o,device=device) for o in temporal_offsets(3,1)]
    audit_window_mask=np.isin(w[:,0],audit_ids)
    end_rows={}
    for name,kind,space in ARMS:
        checkpoint=torch.load(a.temporal_dir/(name+".pt"),map_location="cpu")
        model=make_model(kind).to(device);model.load_state_dict(checkpoint["state_dict"]);model.eval()
        m,s=[y.to(device) for y in coordinates["normalization"][space]]
        pred=[]
        for start in range(0,len(w),1024):
            ww=torch.as_tensor(w[start:start+1024],device=device)
            ix=(ww[:,0,None],ww[:,1,None]+co)
            with torch.no_grad():
                h=(data[space][ix]-m)/s;vv=observed_v[ix]
                aa=observed_a[ww[:,0,None],ww[:,1,None]+ao]
                pp,_=predict(model,h,vv,aa,1); pp=pp[:,0]*s+m
                if space=="pca":
                    native=pp.cpu()@coordinates["pca_basis"].T+coordinates["pca_mean"]
                    pp=projection(native)
                pred.append(pp.cpu())
        predicted=torch.cat(pred); real=all_z[w[:,0],w[:,1]+1]
        target_v=vs[w[:,0],w[:,1]+1]; target_d=ds[w[:,0],w[:,1]+1]
        ap,ar1=actions(predicted,target_v),actions(real,target_v)
        r=paired_state_residual(target_d,target_v,ar1,ap)[:,1]
        ts,score=trajectory_tail_summary(r,w[:,0],len(mask),thresholds)
        at,ascore=trajectory_tail_summary(r[audit_window_mask],w[audit_window_mask,0],len(mask),thresholds)
        end_rows[name]=dict(all_old459_including_train=dict(step=tail_summary(r,thresholds),trajectory_max=ts),
            internal_audit69=dict(step=tail_summary(r[audit_window_mask],thresholds),trajectory_max=at))
        np.savez_compressed(a.output_dir/(name+"_end_complete_h1.npz"),windows=w,residual=r,audit_mask=audit_window_mask)
        print("end-complete h1",name,"audit max=%.9f"%end_rows[name]["internal_audit69"]["step"]["max_abs"],flush=True)
    out["temporal_end_complete_one_step"]=end_rows;save()

    # E. Re-run ORIGINAL global-box stress, plus final actor and full braking.
    dd,vv=np.meshgrid(np.linspace(15,16,20,dtype=np.float32),np.linspace(2.5,3,20,dtype=np.float32))
    starts=np.column_stack((dd.reshape(-1),vv.reshape(-1)))
    @torch.no_grad()
    def proxy_policy(states,which):
        if which=="maximum_braking": return np.full(len(states),3.,dtype=np.float32)
        latent=q(torch.from_numpy(states[:,:1])/scale)
        if which=="legacy_PPO09": return base.predict(np.column_stack((latent.numpy(),states[:,1])),deterministic=True)[0].reshape(-1)
        return actions(latent,states[:,1])
    stress={}
    for which in ("legacy_PPO09","minimal_final_actor","maximum_braking"):
        stress[which]={}
        for radius in sorted(set([-qhat,0.,.05,.1,.15,qhat])):
            states=starts.copy(); labels=np.full(len(states),"running",dtype=object); worst=None
            for step in range(400):
                active=np.flatnonzero(labels=="running")
                if not len(active): break
                aa=proxy_policy(states[active],which)
                successors=successor_scenarios(states[active],aa,[radius])[:,0]
                states[active]=successors
                safe,unsafe=safe_terminal_mask(successors),unsafe_mask(successors)
                labels[active[safe]]="safe_terminal";labels[active[unsafe]]="unsafe"
                if worst is None and unsafe.any():
                    i=int(np.flatnonzero(unsafe)[0]);worst=dict(initial_state=starts[active[i]].tolist(),
                        unsafe_state=successors[i].tolist(),unsafe_step=step+1,action=float(aa[i]))
            labels[labels=="running"]="timeout"
            stress[which]["%.12g"%radius]=dict(outcomes={k:int((labels==k).sum()) for k in ("safe_terminal","unsafe","timeout")},counterexample=worst)
    out["global_box_stress"]=dict(initial_states=400,horizon=400,
        warning="synthetic adversarial residual on proxy dynamics; not a measured process disturbance or actual image rollout",results=stress)
    save();print("box stress complete",flush=True)

    # F. cGAN on the same old trajectory physical states through a SHARED actor.
    # Old trajectory H5 has DINO features, not raw RGB, so native legacy own-image
    # residual cannot be reconstructed here. It was tested on static RGB above.
    generator_path=Path(common_meta["args"]["legacy_generator"])
    weights_path=Path(common_meta["args"]["dino_weights"])
    if sha256(generator_path)!=common_meta["hashes"]["legacy_generator"] or sha256(weights_path)!=common_meta["hashes"]["dino_weights"]:
        raise ValueError("Generator/backbone assets changed")
    config["cgan_generator_sha256"]=sha256(generator_path);config["dino_weights_sha256"]=sha256(weights_path)
    gen=AebsMLPGenerator(4,1).to(device).eval();gen.load_state_dict(torch.load(generator_path,map_location=device))
    backbone=FrozenDinoV2(Path(common_meta["args"]["dino_repo"]),weights_path,device).eval()
    gp=copy.deepcopy(projection).to(device).eval()
    mean,std=rep["feature_mean"].to(device),rep["feature_std"].to(device)
    legacy_scale=float(paired_meta["legacy_scale_m"])

    @torch.no_grad()
    def cgan_latents(distances,draws,label):
        unique,inverse=np.unique(distances,return_inverse=True); k=len(draws)
        result=np.empty((len(unique)*k,32),dtype=np.float32)
        for start in range(0,len(result),64):
            idx=np.arange(start,min(start+64,len(result)))
            yy=torch.from_numpy((unique[idx//k,None]/legacy_scale).astype(np.float32)).to(device)
            zz=torch.from_numpy(draws[idx%k]).to(device)
            raw_g=gen(zz,yy); pixels=(raw_g+1.)*.5
            emb=gp((backbone(pixels).float()-mean)/std)
            result[idx]=emb.cpu().numpy()
            if start==0 or (start+64)%4096==0 or idx[-1]+1==len(result):
                print(label,"DINO encodings",int(idx[-1]+1),"/",len(result),flush=True)
        return result.reshape(len(unique),k,32)[inverse]

    cg_z=cgan_latents(d,np.zeros((1,4),dtype=np.float32),"cGAN z0 all459")[:,0]
    cga=actions(cg_z,v); cgr=paired_state_residual(d,v,final_ar,cga)[:,1]
    cgts,cgscore=trajectory_tail_summary(cgr,ep,len(mask),thresholds)
    cgan_traj=dict(reference="same R2 real-image features and same minimal-final actor; NOT old cGAN native state_net/PPO",
        native_legacy_own_image_trajectory_unavailable="old trajectory cache has no raw RGB; no fabricated inverse-DINO images",
        z_zero_all459=dict(step=tail_summary(cgr,thresholds),trajectory_max=cgts))
    cgan_matrix=np.full(mask.shape,np.nan);cgan_matrix[ep,tt]=cgr
    temporal_window=np.load(a.temporal_dir/"windows.npz")["internal_audit"]
    cgan_traj["aligned_to_temporal_internal_audit_known_future_distance"]={str(h):tail_summary(
        cgan_matrix[temporal_window[:,0],temporal_window[:,1]+h],thresholds) for h in (1,3,10)}
    np.savez_compressed(a.output_dir/"cgan_common_z0_trajectory.npz",trajectory_index=ep,time=tt,residual=cgr)
    keep=np.isin(ep,audit_ids)
    draws=np.random.default_rng(7).uniform(-1,1,(a.cgan_audit_draws,4)).astype(np.float32)
    cg_z=cgan_latents(d[keep],draws,"cGAN uniform audit69")
    cga=actions(cg_z,np.broadcast_to(v[keep,None],cg_z.shape[:2]))
    r=paired_state_residual(d[keep,None],v[keep,None],final_ar[keep,None],cga)[...,1]
    flat_group=np.repeat(ep[keep],len(draws))
    ts,score=trajectory_tail_summary(r.reshape(-1),flat_group,len(mask),thresholds)
    cgan_traj["uniform_mc_audit69"]=dict(draws=len(draws),step=tail_summary(r,thresholds),trajectory_max_over_states_and_draws=ts,
        score_note="extra maximum over random generator draws, not directly a recalibrated CP score")
    np.savez_compressed(a.output_dir/"cgan_common_mc_trajectory.npz",trajectory_index=ep[keep],time=tt[keep],residual=r,draws=draws)
    out["cgan_shared_actor_old_trajectory_diagnostic"]=cgan_traj
    for path,digest in frozen.items():
        if sha256(Path(path))!=digest: raise ValueError("Frozen source changed: "+path)
    out.update(status="absolute_error_and_historical_CP_radius_audit_complete_not_new_certificate",source_files_unchanged=True)
    json_write(a.output_dir/"config.json",config); save()
    print("[Absolute error and 0.15 m/s residual audit]",flush=True)
    print("old CP reproduced %.9f; current minimal maximum %.9f"%(np.nanmax(old_scores),actual["variants"]["original_R2"]["own_image"]["step"]["max_abs"]),flush=True)
    print("metrics:",a.output_dir/"metrics.json",flush=True)


if __name__=="__main__": main()
