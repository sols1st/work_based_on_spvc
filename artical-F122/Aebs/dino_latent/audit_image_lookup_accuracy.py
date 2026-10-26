"""Audit distance coverage and replace lookup-based accuracy by exact pairs.

No training, new RGB capture, interpolation, split mixing for policy evaluation,
or final-test image encoding. Full-manifest TEST LABELS are read only to audit
the collection geometry. Policy accuracy uses separate cached train/validation
images at their OWN labelled physical distance.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from stable_baselines3 import PPO

from Aebs.dino_latent.benchmark_native_controllers import (
    NativeInterface, encode_native, matched_speed_audit,
)
from Aebs.dino_latent.benchmark_native_representations import validate_split
from Aebs.dino_latent.experiment_sbc_risk_cp import save_json, summary
from Aebs.dino_latent.prepare_expanded_data import sha256


def geometry(distances, queries):
    values=np.unique(np.asarray(distances,dtype=np.float64))
    x=np.asarray(queries,dtype=np.float64)
    if not len(values) or not np.isfinite(values).all(): raise ValueError("Finite nonempty distances required")
    right=np.clip(np.searchsorted(values,x),0,len(values)-1)
    left=np.maximum(right-1,0)
    errors=np.minimum(abs(values[left]-x),abs(values[right]-x))
    gaps=np.diff(values)
    top=np.argsort(gaps)[-10:][::-1]
    return dict(unique_distances=len(values), min_distance_m=float(values[0]),max_distance_m=float(values[-1]),
                dense_query_error_m=summary(errors),max_adjacent_gap_m=float(gaps.max()) if len(gaps) else 0.,
                coverage_at_error_m={str(t):float((errors<=t+1e-12).mean()) for t in (.0031,.01,.05)},
                largest_gaps=[dict(left_m=float(values[i]),right_m=float(values[i+1]),
                                  gap_m=float(gaps[i]),midpoint_error_m=float(gaps[i]/2)) for i in top])


def brute_force_check(interface, state_distances, appearance):
    """Independent exhaustive argmin, not the searchsorted implementation."""
    queries=np.asarray(state_distances,dtype=np.float64)
    _,errors,chosen=interface.nearest(queries,appearance)
    pool=interface.pools[appearance]
    expected=[]; minimum=[]
    for start in range(0,len(queries),256):
        diff=abs(queries[start:start+256,None]-interface.distances[pool][None].astype(np.float64))
        index=diff.argmin(1)
        expected.extend(pool[index].tolist()); minimum.extend(diff[np.arange(len(index)),index].tolist())
    expected=np.asarray(expected); minimum=np.asarray(minimum)
    return dict(checked_queries=len(queries),nonminimal_distance_count=int((abs(errors-minimum)>1e-10).sum()),
                selected_source_mismatch_count=int((chosen!=expected).sum()),
                max_error_disagreement_m=float(np.max(abs(errors-minimum))) if len(queries) else 0.)


def plot_coverage(full, rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    weather,color=rows[0]["weather"],rows[0]["color"]
    pools=dict(Original=[r["distance_m"] for r in full["records"] if r["weather"]==weather and r["color"]==color])
    for key,split in (("Train","train"),("Validation","validation")):
        pools[key]=[r["distance_m"] for r in rows if r["weather"]==weather and r["color"]==color and r["split"]==split]
    fig,axes=plt.subplots(2,1,figsize=(10,6.5),sharex=True,gridspec_kw=dict(height_ratios=[1,1.4]))
    colors=["#777777","#2471a3","#c4662b"]
    query=np.linspace(5,16,11001)
    for j,((name,values),color) in enumerate(zip(pools.items(),colors)):
        d=np.unique(values); y=2-j
        axes[0].scatter(d,np.full(len(d),y),marker="|",s=55,color=color,label=name)
        axes[0].text(16.08,y,"%d distances"%len(d),va="center",fontsize=9)
        right=np.clip(np.searchsorted(d,query),0,len(d)-1);left=np.maximum(right-1,0)
        error=np.minimum(abs(d[left]-query),abs(d[right]-query))
        axes[1].plot(query,error,color=color,label=name,lw=1.2)
    axes[0].set_yticks([2,1,0]);axes[0].set_yticklabels(list(pools));axes[0].set_ylim(-.5,2.5)
    axes[0].set_title("Dense acquisition does not imply dense coverage inside each distance-block split")
    axes[1].axhline(.01,color="black",ls="--",lw=.8,label="1 cm alignment gate")
    axes[1].set_xlabel("Actual queried distance (m)");axes[1].set_ylabel("Nearest image label mismatch (m)")
    axes[1].legend(ncol=2,fontsize=9);axes[1].set_xlim(5,17.6);axes[1].set_xticks(range(5,17))
    for ax in axes:
        ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False);ax.grid(alpha=.12)
    fig.tight_layout();fig.savefig(output/"distance_split_coverage.png",dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir",type=Path,default=Path("results/dino_expanded_v1/02_dino_h5"))
    p.add_argument("--full-manifest",type=Path,default=Path("results/dino_expanded_v1/01_data_prepare/split_manifest.json"))
    p.add_argument("--representation",type=Path,default=Path("results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt"))
    p.add_argument("--native-dir",type=Path,default=Path("results/native_representation_dimensions_20261019_v1"))
    p.add_argument("--controller-dir",type=Path,default=Path("results/native_controllers_speed_residual_20261019_v1/historical_R2"))
    p.add_argument("--shift-dir",type=Path,default=Path("results/native_R2_risk_cp_library_shift_20261020_v2"))
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--plot-only",action="store_true",help="Render label geometry into an existing completed audit; no model inference")
    a=p.parse_args()
    if a.plot_only:
        if not (a.output_dir/"metrics.json").is_file():p.error("Completed audit required for plot-only")
        plot_coverage(json.loads(a.full_manifest.read_text()),json.loads((a.cache_dir/"cache_manifest.json").read_text())["records"],a.output_dir)
        return
    if a.output_dir.exists(): p.error("Use a new output directory")
    torch.set_num_threads(1)
    paths=[a.cache_dir/"cache_manifest.json",a.cache_dir/"features.h5",a.full_manifest,a.representation,
           a.native_dir/"historical_R2.pt",a.native_dir/"split.json",a.controller_dir/"native_ppo.zip",
           a.controller_dir/"metrics.json",a.shift_dir/"shift_rollouts.npz",Path(__file__)]
    hashes={str(path):sha256(path) for path in paths}
    cache=json.loads(paths[0].read_text()); rows=cache["records"]
    full=json.loads(a.full_manifest.read_text())
    record=json.loads((a.controller_dir/"metrics.json").read_text())
    if (cache.get("test_encoded")!=0 or hashes[str(paths[1])]!=cache["feature_sha256"]
            or record["controller_sha256"]!=hashes[str(a.controller_dir/"native_ppo.zip")]
            or record["representation_sha256"]!=hashes[str(a.native_dir/"historical_R2.pt")]):
        raise ValueError("Frozen cache/model identity mismatch")
    ck=torch.load(a.native_dir/"historical_R2.pt",map_location="cpu")
    rep=torch.load(a.representation,map_location="cpu")
    if hashes[str(a.representation)] not in ck["config"]["source_sha256"].values():
        raise ValueError("Wrong source projection")
    split=validate_split(json.loads((a.native_dir/"split.json").read_text()),rows)
    with h5py.File(paths[1],"r") as f:
        features,distances=f["features"][:],f["distance_m"][:].astype(np.float32)
        if f["filename"].asstr()[:].tolist()!=[r["filename"] for r in rows]:
            raise ValueError("Cache row order mismatch")
    label_rounding=float(np.max(abs(distances.astype(float)-np.array([r["distance_m"] for r in rows]))))
    if label_rounding>2e-6: raise ValueError("Feature cache labels disagree with manifest")
    latent=encode_native(features,ck,rep)
    train=NativeInterface(ck,latent,distances,rows,split["train"])
    validation_ids=np.concatenate((split["selection"],split["internal_audit"]))
    val=NativeInterface(ck,latent,distances,rows,validation_ids)
    # Metadata-only coverage; never feed full/test image data to a model.
    grid=np.linspace(5,16,11001)
    coverage={}
    for name in train.pools:
        weather,color=name.split("|",1)
        original=[r["distance_m"] for r in full["records"] if r["weather"]==weather and r["color"]==color]
        coverage[name]=dict(original_collection_labels_only=geometry(original,grid),
                            train_pool=geometry(train.distances[train.pools[name]],grid),
                            validation_pool=geometry(val.distances[val.pools[name]],grid))
    saved=np.load(a.shift_dir/"shift_rollouts.npz")
    ii,tt=np.nonzero(saved["valid"])
    current=saved["states_m_mps"][ii,tt,0]
    checks={}
    for j,name in enumerate(saved["appearance_labels"]):
        keep=saved["appearance_index"][ii]==j
        checks[str(name)]=brute_force_check(val,current[keep],str(name))
    if any(r["nonminimal_distance_count"] for r in checks.values()):
        raise ValueError("Lookup algorithm did not return a true nearest distance")
    a.output_dir.mkdir(parents=True)
    save_json(a.output_dir/"config.json",dict(source_sha256=hashes,grid_step_m=.001,
              exact_state_speed_grid=list(np.linspace(0,3,31)),test_labels_used_for_geometry_only=True,
              test_images_encoded_or_evaluated=False,model_training=False))
    model=PPO.load(a.controller_dir/"native_ppo.zip",device="cpu")
    model.policy.set_training_mode(False);model.policy.requires_grad_(False)
    exact={}
    for name,indices in (("train",split["train"]),("validation",validation_ids)):
        exact[name]=matched_speed_audit(model,train,distances,indices,a.output_dir/(name+"_exact_matched.npz"))
        print("[Exact image-label evaluation]",name,exact[name],flush=True)
    # One centimetre is an observation-alignment quality gate, NOT a safety
    # relaxation or an error radius fitted from observed success/failure.
    eligible=all(v["train_pool"]["dense_query_error_m"]["max"]<=.01 for v in coverage.values())
    for path,digest in hashes.items():
        if sha256(Path(path))!=digest:raise ValueError("Source changed during audit")
    result=dict(status="lookup_coverage_and_exact_state_audit_complete",coverage_by_appearance=coverage,
                brute_force_lookup_checks=checks,checked_lookup_queries=len(current),
                lookup_code_nonminimal_count=sum(r["nonminimal_distance_count"] for r in checks.values()),
                feature_label_float32_rounding_max_m=label_rounding,full_split_groups=full.get("groups"),
                exact_state_results=exact,strict_image_state_alignment_gate=dict(max_error_m=.01,
                    full_domain_train_library_eligible=eligible,
                    on_failure="do NOT call nearest-library rollout a synchronized image-path evaluation; do not discard difficult episodes"),
                image_collection_label_mismatch_proven=False,
                cause="distance-BLOCK split leaves whole intervals absent from each lookup pool; many appearances are not many distinct distances",
                test_image_model_evaluation=False, source_sha256=hashes)
    save_json(a.output_dir/"metrics.json",result)
    first=next(iter(coverage.values()))
    print("[Lookup geometry]",{k:(v["unique_distances"],v["max_adjacent_gap_m"],v["dense_query_error_m"]["max"]) for k,v in first.items()},flush=True)
    print("brute-force checked",len(current),"nonminimal",result["lookup_code_nonminimal_count"],flush=True)
    print("strict synchronized-image eligibility:",eligible,"metrics:",a.output_dir/"metrics.json",flush=True)


if __name__=="__main__":main()
