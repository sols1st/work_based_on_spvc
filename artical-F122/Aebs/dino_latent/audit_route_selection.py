"""Role-separated, matched-target development audit plus CPU predictor cost.

Static q proxies know the target physical distance; forecasters do NOT.
They are complementary roles, never a single same-information leaderboard.
No PPO/SBC/CP update. Earlier selection-set use is reported, not hidden.
"""
import argparse
import copy
import hashlib
import json
import platform
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from Aebs.dino_latent.models import SafetyProjection,SafetyDecoder,PhysicalToLatent
from Aebs.dino_latent.spvc_saved_actor import load_spvc_pair
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.benchmark_latent_methods import ARMS
from Aebs.dino_latent.latent_method_models import make_model,predict,temporal_offsets,paired_cluster_bootstrap
from Aebs.dino_latent.temporal_world_model import TemporalPredictor,window_indices,rollout
from Aebs.dino_latent.error_tail_metrics import tail_summary,trajectory_tail_summary
from Aebs.dino_latent.calibrate_transition_contract import next_state_residual


def matched_indices(episodes,times,windows,horizon):
    keys=list(zip(np.asarray(episodes).tolist(),np.asarray(times).tolist()))
    if len(set(keys))!=len(keys): raise ValueError("Duplicate physical state keys")
    lookup={k:i for i,k in enumerate(keys)}
    try:return np.array([lookup[(int(e),int(t)+horizon)] for e,t in windows],dtype=np.int64)
    except KeyError:raise ValueError("Target physical state missing")


def state_digest(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode());h.update(str(value.dtype).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def common_nontraining_ids(latest_audit,legacy_splits):
    common=set(latest_audit)
    for split in legacy_splits:
        if set(split["train"]) & set(split["validation"]):raise ValueError("Legacy split overlap")
        common &= set(split["validation"])
    if len(common)<2:raise ValueError("Too few common nontraining trajectories")
    return common


def predictor_timing(model,legacy,dimension,horizon):
    """Warm CPU predictor-only inference, single torch thread, no DINO/actor."""
    net=copy.deepcopy(model).cpu().eval();result={}
    for batch in (1,64):
        z=torch.zeros(batch,3,dimension);v=torch.ones(batch,3);u=torch.zeros(batch,3+horizon-1)
        def call():return rollout(net,z,v,u,horizon) if legacy else predict(net,z,v,u,horizon)[0]
        with torch.no_grad():
            for _ in range(10):call()
            values=[]
            for _ in range(40):
                tic=time.perf_counter();call();values.append((time.perf_counter()-tic)*1000)
        result[str(batch)]=dict(median_ms=float(np.median(values)),p95_ms=float(np.quantile(values,.95)),
            repeats=40,warmups=10,scope="predictor-only CPU; excludes DINO, projection, actor, capture and transfer")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark",type=Path,default=Path("results/latent_methods_benchmark_20261018_run1"))
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    a=p.parse_args()
    if a.output_dir.exists():p.error("New output required")
    cfg=json.loads((a.benchmark/"config.json").read_text())
    candidate_meta=json.loads((a.candidate/"metrics.json").read_text())
    if not candidate_meta.get("source_files_unchanged") or candidate_meta["config"]["CP_pool_used"]:
        raise ValueError("Candidate provenance invalid")
    sources=dict(cfg["source_sha256"])
    static_metrics_path=Path("results/static_latent_benchmark_20261018_qcontrol/metrics.json")
    sources[str(static_metrics_path)]=sha256(static_metrics_path)
    for path in (a.benchmark/"coordinates.pt",a.benchmark/"split.json",a.benchmark/"windows.npz",a.candidate/"model.pt",a.candidate/"metrics.json"):
        sources[str(path)]=sha256(path)
    if sources[str(a.candidate/"model.pt")]!=candidate_meta["checkpoint_sha256"]:raise ValueError("Candidate changed")
    trajectories=next(Path(f) for f in sources if "13_spvc_calibration_trajectories_v1/trajectories.h5" in f)
    representation=next(Path(f) for f in sources if f.endswith("08_R2_group_alignment/dino_safety_latent.pt"))
    ppo=next(Path(f) for f in sources if f.endswith("09_R2_ppo/latent_ppo.zip"))
    spvc=next(Path(f).parent for f in sources if f.endswith("latent_ppo_spvc.pt"))
    recheck=next(Path(f) for f in sources if "final_recheck/metrics.json" in f)
    specs=[];legacy_splits=[]
    for folder in (Path("results/transformer_pilot_20261018"),Path("results/transformer_speed_ablation_20261018")):
        split=json.loads((folder/"split.json").read_text());legacy_splits.append(split)
        sources[str(folder/"split.json")]=sha256(folder/"split.json")
        for path in sorted(folder.glob("*.pt")):
            specs.append(dict(name=folder.name+"/"+path.stem,path=path,legacy=True,horizons=(1,3)))
    specs += [dict(name="latest/"+n,path=a.benchmark/(n+".pt"),legacy=False,kind=k,space=s,horizons=(1,3,10)) for n,k,s in ARMS]
    specs += [dict(name="new/mlp32_parameter_matched_task_aware",path=a.candidate/"model.pt",legacy=False,kind="mlp_matched",space="r2",horizons=(1,3,10))]
    for spec in specs:sources[str(spec["path"])]=sha256(spec["path"])
    if not all(sha256(Path(f))==h for f,h in sources.items()):raise ValueError("Source mismatch")
    torch.set_num_threads(1);device=torch.device(a.device)
    rep=torch.load(representation,map_location="cpu")
    projection,decoder,q=SafetyProjection().eval(),SafetyDecoder().eval(),PhysicalToLatent().eval()
    for net,key in ((projection,"projection_state_dict"),(decoder,"auxiliary_decoder_state_dict"),(q,"physical_to_latent_state_dict")):
        net.load_state_dict(rep[key]);net.to(device)
    _,saved,_,identity=load_spvc_pair(representation,ppo,spvc,recheck)
    actor=saved.policy.controller_net.to(device).eval()
    for net in (projection,decoder,q,actor):
        for par in net.parameters():par.requires_grad_(False)
    with h5py.File(trajectories) as f:
        valid=f["valid"][:].astype(bool);raw=f["dino_features"][:]
        ds,vs,ac=(f[k][:] for k in ("distance_m","speed_mps","image_action"))
        ids=f["trajectory_id"].asstr()[:].tolist()
        appearance=np.array([x+"|"+y for x,y in zip(f["weather"].asstr()[:],f["color"].asstr()[:])])
    latest_split=json.loads((a.benchmark/"split.json").read_text())
    audit_ids=set(latest_split["internal_audit"])
    if audit_ids & (set(latest_split["train"])|set(latest_split["selection"])):
        raise ValueError("Latest fitting/selection/audit overlap")
    common=common_nontraining_ids(audit_ids,legacy_splits)
    id_map={s:i for i,s in enumerate(ids)}
    cohorts={"latest_same_split69":np.array(sorted(id_map[x] for x in audit_ids)),
        "common_nontraining":np.array(sorted(id_map[x] for x in common))}
    with torch.no_grad():
        features=(torch.from_numpy(np.nan_to_num(raw)).to(device)-rep["feature_mean"].to(device))/rep["feature_std"].to(device)
        z=projection(features.flatten(0,1)).reshape(*valid.shape,32)
    coords=torch.load(a.benchmark/"coordinates.pt",map_location=device)
    pca=(features-coords["pca_mean"])@coords["pca_basis"]
    speed,control=[torch.from_numpy(np.nan_to_num(x)).to(device) for x in (vs,ac)]
    def actions(latents,velocity):
        v=torch.as_tensor(velocity,device=device,dtype=torch.float32)
        with torch.no_grad():return actor(torch.cat((latents,v[:,None]),-1)).clamp(-3,3).flatten().cpu().numpy()
    a.output_dir.mkdir(parents=True)
    config=dict(source_sha256=sources,actor_identity=identity,training=False,new_CP_calibration=False,
        step20_H5_used=False,RGB_test_used=False,device=str(device),torch_version=torch.__version__,platform=platform.platform(),
        cohort_counts={k:len(v) for k,v in cohorts.items()},
        cohort_trajectory_ids={k:[ids[int(i)] for i in v] for k,v in cohorts.items()},
        common_nontraining_note="not used to fit these predictors, but legacy methods used their validation pool for checkpoint selection; NOT an independent final test",
        latest69_note="same train/selection/internal-audit split for latest9 plus new candidate; old development pool, NOT new final test",
        no_universal_ranking="verification proxies know target distance; temporal predictors use history/control queries; different roles",
        training_budget_caveat="legacy suites:1200 updates/batch256; latest/new:800 updates/batch128; cross-suite differences do not isolate architecture",
        frozen_specifications=[dict(name=s["name"],path=str(s["path"]),horizons=s["horizons"]) for s in specs],
        timing="CPU predictor only, single thread, warmup10, repetitions40; batch1 and64; not camera-to-action latency",
        code_sha256=sha256(Path(__file__)))
    (a.output_dir/"config.json").write_text(json.dumps(config,indent=2))
    output=dict(status="running",config=config,forecasters={},reference_proxy={})
    raw_results={}; windows={h:window_indices(valid,cohorts["latest_same_split69"],3,h) for h in (1,3,10)}
    def summarize(r,w,h,ai=None,ap=None):
        d,v=ds[w[:,0],w[:,1]+h],vs[w[:,0],w[:,1]+h]
        groups={"all":np.ones(len(w),bool),"near_d6_7":(d>=6)&(d<=7),
                "braking_switch_d6_7_v0p5_1":(d>=6)&(d<=7)&(v>=.5)&(v<=1),
                "near_d_le8_v_ge0p5":(d<=8)&(v>=.5)}
        groups.update({"appearance:"+key:appearance[w[:,0]]==key for key in np.unique(appearance[w[:,0]])})
        result={k:tail_summary(r[keep],[.05,.1,.15]) for k,keep in groups.items() if keep.any()}
        ts,_=trajectory_tail_summary(r,w[:,0],len(ids),[.15]);result["trajectory_maxima"]=ts
        if ai is not None:
            ni=np.clip(v-.05*ai,0,3);npred=np.clip(v-.05*ap,0,3);nd=d-.05*v
            ui=(nd<=6)&(ni>.5);up=(nd<=6)&(npred>.5)
            result["matched_next_unsafe"]=dict(image_only=int((ui&~up).sum()),proxy_only=int((up&~ui).sum()),both=int((ui&up).sum()),
                scope="one-step counterfactual on reference states; NOT autonomous episodes")
        return result
    for h,w in windows.items():
        with torch.no_grad(): zp=q(torch.from_numpy(ds[w[:,0],w[:,1]+h,None]).to(device)/float(rep["distance_scale_m"]))
        truth=z[w[:,0],w[:,1]+h];vv=vs[w[:,0],w[:,1]+h];dd=ds[w[:,0],w[:,1]+h]
        ar,ap=actions(truth,vv),actions(zp,vv)
        rr=next_state_residual(dd,vv,ar,ap)[:,1]
        result={}
        for cohort,eps in cohorts.items():
            keep=np.isin(w[:,0],eps);result[cohort]=summarize(rr[keep],w[keep],h,ar[keep],ap[keep])
        output["reference_proxy"][str(h)]=result
    for spec in specs:
        ck=torch.load(spec["path"],map_location="cpu")
        if spec["legacy"]:
            dimension=32 if ck["native"]=="latent32" else 384
            arm=ck.get("arm",{})
            model=TemporalPredictor(dimension,ck["kind"],speed_mode=arm.get("speed_mode","concat"),use_action=arm.get("use_action",True)).to(device)
            native=ck["native"];source=z if dimension==32 else features
            mean,std=[x.to(device) for x in ck["normalization"][native]]
        else:
            dimension=32;model=make_model(spec["kind"]).to(device);source=z if spec["space"]=="r2" else pca
            mean,std=coords["normalization"][spec["space"]]
        model.load_state_dict(ck["state_dict"]);model.eval()
        result=dict(parameters=sum(x.numel() for x in model.parameters()),weight_tensor_digest=state_digest(ck["state_dict"]),
            legacy=spec["legacy"],horizons={},CPU_predictor_timing={})
        for h in spec["horizons"]:
            w=windows[h];co,_,ao=[torch.as_tensor(x,device=device) for x in temporal_offsets(3,h)]
            preds=[]
            with torch.no_grad():
                for start in range(0,len(w),512):
                    ww=torch.as_tensor(w[start:start+512],device=device);ei=ww[:,0,None];ti=ww[:,1,None]
                    hist=(source[ei,ti+co]-mean)/std;vv=speed[ei,ti+co];u=control[ei,ti+ao]
                    pred=rollout(model,hist,vv,u,h) if spec["legacy"] else predict(model,hist,vv,u,h)[0]
                    pred=pred[:,-1]*std+mean
                    if spec["legacy"] and dimension==384:pred=projection(pred)
                    elif not spec["legacy"] and spec["space"]=="pca":pred=projection(pred@coords["pca_basis"].T+coords["pca_mean"])
                    preds.append(pred)
            pred=torch.cat(preds);truth=z[w[:,0],w[:,1]+h]
            vv=vs[w[:,0],w[:,1]+h];dd=ds[w[:,0],w[:,1]+h]
            ar,ap=actions(truth,vv),actions(pred,vv)
            rr=next_state_residual(dd,vv,ar,ap)[:,1]
            with torch.no_grad():latent_rms=(pred-truth).square().mean(-1).sqrt().cpu().numpy()
            rrsets={}
            for cohort,eps in cohorts.items():
                if cohort=="latest_same_split69" and spec["legacy"]:continue
                keep=np.isin(w[:,0],eps)
                rrsets[cohort]=summarize(rr[keep],w[keep],h,ar[keep],ap[keep])
                rrsets[cohort]["common_R2_latent_RMS"]=tail_summary(latent_rms[keep],unit="R2 vector RMS; task-trained readout")
            result["horizons"][str(h)]=rrsets
            result["CPU_predictor_timing"][str(h)]=predictor_timing(model,spec["legacy"],dimension,h)
            raw_results[(spec["name"],h)]=rr
            np.savez_compressed(a.output_dir/(spec["name"].replace('/','__')+"__h%d.npz"%h),windows=w,residual=rr,
                image_action=ar,predicted_action=ap,common_latent_rms=latent_rms)
        output["forecasters"][spec["name"]]=result
        (a.output_dir/"metrics.json").write_text(json.dumps(output,indent=2,allow_nan=False))
        print(spec["name"],"finished",flush=True)
    paired={}
    for spec in specs:
        if spec["legacy"]:continue
        paired[spec["name"]]={}
        for h in spec["horizons"]:
            paired[spec["name"]][str(h)]=paired_cluster_bootstrap(np.abs(raw_results[(spec["name"],h)]),
                np.abs(raw_results[("latest/transformer32",h)]),windows[h][:,0])
    output["latest_paired_vs_transformer"]=paired
    output["static_representation_existing_evidence"]=json.loads(static_metrics_path.read_text())
    if not all(sha256(Path(f))==h for f,h in sources.items()):raise ValueError("Frozen source changed")
    output.update(status="role_separated_development_route_audit_complete",source_files_unchanged=True,
        forecaster_cases=len(specs),unique_predictor_weights=len(set(x["weight_tensor_digest"] for x in output["forecasters"].values())),
        no_new_closed_loop_or_safety_claim=True)
    (a.output_dir/"metrics.json").write_text(json.dumps(output,indent=2,allow_nan=False))
    print("[Route selection audit]",config["cohort_counts"],"cases",len(specs),flush=True)
    print("metrics:",a.output_dir/"metrics.json",flush=True)


if __name__=="__main__":main()
