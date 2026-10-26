"""PCA/AE/VAE image-vs-distance-proxy pilots on existing DINO feature cache.

Not full raw-image VAE, not replacement PPO/SBC training. Common R2 readout is
task-trained and favored to R2. Native latent magnitudes cannot be cross-ranked.
"""
import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from Aebs.dino_latent.models import SafetyProjection, SafetyDecoder, PhysicalToLatent
from Aebs.dino_latent.spvc_saved_actor import load_spvc_pair
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.run_transformer_comparison import summary
from Aebs.dino_latent.latent_method_models import paired_cluster_bootstrap


class FeatureAutoencoder(nn.Module):
    def __init__(self, variational=False):
        super().__init__()
        self.variational=variational
        self.encoder=nn.Sequential(nn.Linear(384,128),nn.GELU(),nn.Linear(128,64 if variational else 32))
        self.decoder=nn.Sequential(nn.Linear(32,128),nn.GELU(),nn.Linear(128,384))

    def encode(self,x):
        parameters=self.encoder(x)
        if not self.variational: return parameters,None
        mean,logvar=parameters.chunk(2,-1)
        return mean,logvar.clamp(-8,5)

    def forward(self,x):
        mean,logvar=self.encode(x)
        value=mean
        kl=x.new_zeros(())
        if self.variational:
            if self.training: value=mean+(.5*logvar).exp()*torch.randn_like(mean)
            kl=.5*(mean.square()+logvar.exp()-1-logvar).mean()
        return self.decoder(value),kl


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir",type=Path,default=Path("results/dino_expanded_v1/02_dino_h5"))
    p.add_argument("--representation",type=Path,default=Path("results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt"))
    p.add_argument("--ppo",type=Path,default=Path("results/dino_expanded_v1/09_R2_ppo/latent_ppo.zip"))
    p.add_argument("--spvc-dir",type=Path,default=Path("results/dino_spvc_minimal_v1"))
    p.add_argument("--final-recheck",type=Path,default=Path("results/dino_spvc_minimal_v1_final_recheck/metrics.json"))
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--steps",type=int,default=1200)
    p.add_argument("--q-steps",type=int,default=1500)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    a=p.parse_args()
    if a.output_dir.exists() or min(a.steps,a.q_steps)<1: p.error("New output and positive budgets required")
    torch.set_num_threads(1); device=torch.device(a.device)
    manifest_path=a.cache_dir/"cache_manifest.json"; feature_path=a.cache_dir/"features.h5"
    manifest=json.loads(manifest_path.read_text()); rows=manifest["records"]
    if manifest.get("test_encoded")!=0 or any(r["split"] not in ("train","validation") for r in rows):
        raise ValueError("No test rows allowed")
    if sha256(feature_path)!=manifest["feature_sha256"]: raise ValueError("Cache hash mismatch")
    sources={str(f):sha256(f) for f in (manifest_path,feature_path,a.representation,a.ppo,a.final_recheck,
        a.spvc_dir/"latent_ppo_spvc.pt",a.spvc_dir/"sbc.pt",a.spvc_dir/"metrics.json")}
    with h5py.File(feature_path,"r") as f:
        if not f.attrs.get("complete",False): raise ValueError("Incomplete cache")
        raw=f["features"][:]
        if f["filename"].asstr()[:].tolist()!=[r["filename"] for r in rows]: raise ValueError("Row order mismatch")
        if not np.allclose(f["distance_m"][:],[r["distance_m"] for r in rows],rtol=0,atol=1e-9):
            raise ValueError("Distance label mismatch")
    if not np.isfinite(raw).all() or raw.shape!=(len(rows),384): raise ValueError("Invalid features")
    train=np.array([i for i,r in enumerate(rows) if r["split"]=="train"])
    val=np.array([i for i,r in enumerate(rows) if r["split"]=="validation"])
    distances=np.array([r["distance_m"] for r in rows],dtype=np.float32)
    groups=np.array([str(r["group_id"]) for r in rows])
    unique=np.unique(groups[val]); order=np.random.default_rng(71).permutation(unique)
    selection=val[np.isin(groups[val],order[:len(order)//2])]
    audit=val[~np.isin(groups[val],order[:len(order)//2])]
    if set(groups[train])&set(groups[val]): raise ValueError("Cross-split exact-distance groups")
    if min(len(selection),len(audit),len(train))==0: raise ValueError("Empty split")
    rep=torch.load(a.representation,map_location="cpu")
    projection,decoder,original_q=SafetyProjection(),SafetyDecoder(),PhysicalToLatent()
    for model,key in ((projection,"projection_state_dict"),(decoder,"auxiliary_decoder_state_dict"),(original_q,"physical_to_latent_state_dict")):
        model.load_state_dict(rep[key]); model.to(device).eval()
        for par in model.parameters(): par.requires_grad_(False)
    _,student,_,actor_identity=load_spvc_pair(a.representation,a.ppo,a.spvc_dir,a.final_recheck)
    actor=student.policy.controller_net.to(device).eval()
    for par in actor.parameters(): par.requires_grad_(False)
    scale=float(rep["distance_scale_m"])
    raw=torch.from_numpy(raw).to(device)
    x=(raw-rep["feature_mean"].to(device))/rep["feature_std"].to(device)
    d=torch.as_tensor(distances,device=device)
    with torch.no_grad(): reference=projection(x)
    # train-only centering/PCA. Coordinates remain unchanged across subsequent fits.
    xm=x[train].mean(0); xx=x[train]-xm
    eigenvalues,basis=torch.linalg.eigh(xx.T@xx/(len(xx)-1)); basis=basis[:,-32:].flip(-1)
    config=dict(steps=a.steps,q_steps=a.q_steps,seed=7,split_seed=71,source_sha256=sources,
        train_rows=len(train),selection_rows=len(selection),internal_audit_rows=len(audit),
        selection_exact_distance_groups=len(np.unique(groups[selection])),audit_exact_distance_groups=len(np.unique(groups[audit])),
        native_dimension=32,vae_kl_weight=.001,test_used=False,ppo_sbc_frozen=True,
        internal_audit="old validation pool previously used for R2 selection; NOT independent final test",
        evaluator="fixed task-trained R2 projection/decoder/actor; favors R2 coordinates",
        comparison="new codecs reconstruct DINO features before common-R2 readout; not a drop-in new PPO",
        q_fit="same 1500-step conditional mean fitting for each NEW frozen codec; original q remains frozen",
        actor_identity=actor_identity,code_sha256=sha256(Path(__file__)))
    a.output_dir.mkdir(parents=True)
    (a.output_dir/"config.json").write_text(json.dumps(config,indent=2,allow_nan=False))
    (a.output_dir/"split.json").write_text(json.dumps({"train":train.tolist(),"selection":selection.tolist(),"internal_audit":audit.tolist()},indent=2))
    torch.save(dict(mean=xm,basis=basis),a.output_dir/"pca_coordinates.pt")
    results={}; errors={}

    @torch.no_grad()
    def evaluate(name,zi,zq,image_common,proxy_common):
        native=zi-zq
        centered=float((zi-zi.mean(0)).square().mean().sqrt())
        pair=(image_common-proxy_common).square().mean(-1).sqrt()
        oracle=(proxy_common-reference[audit]).square().mean(-1).sqrt()
        image_gap=(image_common-reference[audit]).square().mean(-1).sqrt()
        distance_error=decoder(proxy_common).squeeze(-1)*scale-d[audit]
        image_distance_error=decoder(image_common).squeeze(-1)*scale-d[audit]
        proxies=[]; native_pairs=[]; flat_physical=[]; flat_actions=[]
        for start in range(0,len(audit),256):
            vel=torch.linspace(0,3,31,device=device)[None].expand(len(audit[start:start+256]),-1)
            def action_for(values):
                values=values[start:start+256,None].expand(-1,31,-1)
                return actor(torch.cat((values,vel[...,None]),-1).flatten(0,1)).reshape_as(vel).clamp(-3,3)
            ap,ar,ai=action_for(proxy_common),action_for(reference[audit]),action_for(image_common)
            # Exclude already-terminal states just as in the matched diagnostic.
            dd=d[audit[start:start+256],None]
            keep=(dd>6)&(vel>1e-6)
            gap=(vel-.05*ap).clamp(0,3)-(vel-.05*ar).clamp(0,3)
            pair_gap=(vel-.05*ap).clamp(0,3)-(vel-.05*ai).clamp(0,3)
            # Every audit image lies in original validation distances (>6).
            if not bool(keep.any(1).all()): raise ValueError("Audit image without nonterminal speed pair")
            proxies.append(torch.stack([gap[i][keep[i]].abs().mean() for i in range(len(gap))]))
            native_pairs.append(torch.stack([pair_gap[i][keep[i]].abs().mean() for i in range(len(gap))]))
            flat_physical.append(gap[keep].cpu().numpy()); flat_actions.append((ap-ar)[keep].cpu().numpy())
        physical=torch.cat(proxies).cpu().numpy(); pair_physical=torch.cat(native_pairs).cpu().numpy()
        arr=np.stack((pair.cpu().numpy(),oracle.cpu().numpy(),physical,image_gap.cpu().numpy(),pair_physical),-1)
        np.savez_compressed(a.output_dir/(name+"_audit.npz"),values=arr,rows=audit,groups=groups[audit])
        result=dict(native_latent_rmse=float(native.square().mean().sqrt()),native_centered_rms=centered,
            relative_native_rmse=float(native.square().mean().sqrt())/centered if centered>1e-8 else None,
            collapsed=centered<=1e-8,common_pair_gap=summary(arr[:,0]),common_proxy_vs_original_image_gap=summary(arr[:,1]),
            common_image_codec_gap=summary(arr[:,3]),proxy_distance_error_m=summary(distance_error.cpu().numpy()),
            image_distance_error_m=summary(image_distance_error.cpu().numpy()),
            all_nonterminal_speed_pairs=int(sum(len(v) for v in flat_physical)),
            all_pair_proxy_vs_original_speed_gap_mps=summary(np.concatenate(flat_physical)),
            all_pair_proxy_vs_original_action_gap=summary(np.concatenate(flat_actions)),
            per_image_mean_proxy_vs_original_speed_gap_mps=summary(physical),
            per_image_mean_dual_path_speed_gap_mps=summary(pair_physical))
        return result,arr

    with torch.no_grad():
        original_proxy=original_q(d[audit,None]/scale)
        results["original_R2"],errors["original_R2"]=evaluate("original_R2",reference[audit],original_proxy,reference[audit],original_proxy)
    for name in ("r2_frozen_refit_q","pca32","feature_ae32","feature_vae32"):
        torch.manual_seed(7); rng=np.random.default_rng(7); codec=None; best_step=0
        if name not in ("pca32","r2_frozen_refit_q"):
            codec=FeatureAutoencoder(name=="feature_vae32").to(device)
            optimizer=torch.optim.AdamW(codec.parameters(),lr=.001,weight_decay=.0001)
            best_score=float("inf"); best=None
            for step in range(1,a.steps+1):
                codec.train(); indices=rng.choice(train,256)
                rec,kl=codec(x[indices]); loss=F.mse_loss(rec,x[indices])+.001*kl
                if not torch.isfinite(loss): raise ValueError("Nonfinite codec loss")
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(codec.parameters(),5.); optimizer.step()
                if step==1 or step%200==0 or step==a.steps:
                    codec.eval()
                    with torch.no_grad():
                        score=float(F.mse_loss(codec(x[selection])[0],x[selection]))
                    if score<best_score: best_score,best,best_step=score,copy.deepcopy(codec.state_dict()),step
                    print(name,"codec step",step,"selection_reconstruction_mse",score,flush=True)
            codec.load_state_dict(best); codec.eval()
            for par in codec.parameters(): par.requires_grad_(False)
        with torch.no_grad():
            latent=reference if name=="r2_frozen_refit_q" else (x-xm)@basis if codec is None else codec.encode(x)[0]
            lm,ls=latent[train].mean(0),latent[train].std(0).clamp_min(.01)
            target=(latent-lm)/ls
        torch.manual_seed(7); rng=np.random.default_rng(7)
        newq=PhysicalToLatent().to(device)
        optimizer=torch.optim.AdamW(newq.parameters(),lr=.001,weight_decay=.00001)
        best_q,best_q_score,best_q_step=None,float("inf"),0
        for step in range(1,a.q_steps+1):
            indices=rng.choice(train,256)
            loss=F.mse_loss(newq(d[indices,None]/scale),target[indices])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if step==1 or step%100==0 or step==a.q_steps:
                with torch.no_grad(): score=float(F.mse_loss(newq(d[selection,None]/scale),target[selection]))
                if score<best_q_score: best_q_score,best_q,best_q_step=score,copy.deepcopy(newq.state_dict()),step
        newq.load_state_dict(best_q); newq.eval()
        with torch.no_grad():
            zi=latent[audit]; zq=newq(d[audit,None]/scale)*ls+lm
            def readout(value):
                if name=="r2_frozen_refit_q": return value
                return projection(value@basis.T+xm if codec is None else codec.decoder(value))
            result,err=evaluate(name,zi,zq,readout(zi),readout(zq))
        result.update(best_codec_step=best_step,best_q_step=best_q_step,q_selection_mse=best_q_score)
        results[name]=result; errors[name]=err
        torch.save(dict(codec_state_dict=None if codec is None else codec.state_dict(),q_state_dict=best_q,
            latent_mean=lm,latent_std=ls,config=config),a.output_dir/(name+".pt"))
        print(name,"audit",result,flush=True)
    # Cluster nearby distances together, including all their appearances.
    cluster=np.array([str(rows[i]["distance_bin_025m"]) for i in audit])
    paired={name:{field:paired_cluster_bootstrap(arr[:,j],errors["original_R2"][:,j],cluster)
        for j,field in enumerate(("common_pair_gap","common_proxy_vs_original_image_gap","physical_gap","image_codec_gap","dual_path_physical_gap"))}
        for name,arr in errors.items() if name!="original_R2"}
    for path,digest in sources.items():
        if sha256(Path(path))!=digest: raise ValueError("Frozen source changed")
    result=dict(status="static_latent_development_comparison_complete",config=config,results=results,
                paired_vs_original_R2=paired,source_files_unchanged=True,test_evaluated=False)
    (a.output_dir/"metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print("[Static latent representation internal development audit]",flush=True)
    for name,r in results.items():
        print(name,"relative native",r["relative_native_rmse"],"common proxy gap",r["common_proxy_vs_original_image_gap"]["mean"],
              "physical MAE",r["per_image_mean_proxy_vs_original_speed_gap_mps"]["mean"],flush=True)
    print("metrics:",a.output_dir/"metrics.json",flush=True)


if __name__=="__main__": main()
