"""Whole-trajectory development A/B; original R2/PPO/SBC are never edited.

Uses the old 13_* data, already reused for scenario-SBC development. Does NOT
read fresh CP calibration (20_*) or final RGB test data. No safety claim.
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
from Aebs.dino_latent.temporal_world_model import (
    TemporalPredictor, split_trajectories, window_indices, rollout,
)


def summary(values):
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    return dict(mean=float(value.mean()), mae=float(np.abs(value).mean()), rmse=float(np.sqrt((value**2).mean())),
                p95=float(np.quantile(np.abs(value), .95)),
                max=float(np.abs(value).max()))


def gather(array, windows, offsets):
    return array[windows[:, 0, None], windows[:, 1, None] + offsets[None]]


def learning_arms(suite):
    if suite == "baseline":
        rows = [("mlp32","latent32","mlp","concat",True),
                ("transformer32","latent32","transformer","concat",True),
                ("transformer384","dino384","transformer","concat",True)]
    elif suite == "speed_ablation":
        rows = [
            ("t32_concat_speed","latent32","transformer","concat",True),
            ("t32_no_speed","latent32","transformer","none",True),
            ("t32_separate_speed","latent32","transformer","add",True),
            ("t32_images_only","latent32","transformer","none",False),
            ("mlp32_no_speed","latent32","mlp","none",True),
            ("t384_concat_speed","dino384","transformer","concat",True),
            ("t384_no_speed","dino384","transformer","none",True),
            ("t384_separate_speed","dino384","transformer","add",True),
        ]
    else:
        raise ValueError("Unknown suite")
    return [dict(name=n,native=f,kind=k,speed_mode=v,use_action=a) for n,f,k,v,a in rows]


def finalize_existing(output_dir):
    """Finish checks only; never retrain or recompute/select a checkpoint."""
    path = output_dir / "metrics.json"
    result = json.loads(path.read_text())
    names = [arm["name"] for arm in result["config"].get("learning_arms",learning_arms("baseline"))]
    required = {"persistence32", "current_q_known_distance", *names}
    if set(result["results"]) != required:
        raise ValueError("Not all arms finished; cannot finalize partial metrics")
    changed = {f: sha256(Path(f)) != digest for f, digest in result["config"]["source_sha256"].items()}
    if any(changed.values()):
        raise ValueError("Frozen source changed: %s" % changed)
    result["model_sha256"] = {name:sha256(output_dir/(name+".pt"))
        for name in names}
    result["source_files_unchanged"] = True
    result["status"] = "development_prediction_comparison_complete_not_a_safety_certificate"
    result["runtime_seconds"] = sum(result["results"][name]["runtime_seconds"]
        for name in names)
    result["runtime_scope"] = "sum of learning-arm training/selection; excludes final evaluations and initial baselines"
    result["finalization_script_sha256"] = sha256(Path(__file__))
    result.setdefault("finalization_note", "Source paths converted to Path for checksum; no weights/metrics changed by finalizer")
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def print_results(output, output_dir):
    print("[Causal Transformer vs 32D development comparison]",flush=True)
    for name,r in output["results"].items():
        h = r["metrics"]["1"]
        print("%s h1 latent_RMS=%.6f distance_MAE_m=%.6f action_gap_MAE=%.6f successor_gap_mps=%.6f" % (
            name,h["latent_rms"]["mean"],h["distance_error"]["mae"],
            h["action_gap"]["mae"],h["successor_speed_gap"]["mae"]),flush=True)
    print("metrics:",output_dir/"metrics.json",flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectories", type=Path, default=Path(
        "results/dino_expanded_v1/13_spvc_calibration_trajectories_v1/trajectories.h5"))
    p.add_argument("--representation", type=Path, default=Path(
        "results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt"))
    p.add_argument("--ppo", type=Path, default=Path("results/dino_expanded_v1/09_R2_ppo/latent_ppo.zip"))
    p.add_argument("--spvc-dir", type=Path, default=Path("results/dino_spvc_minimal_v1"))
    p.add_argument("--final-recheck", type=Path, default=Path("results/dino_spvc_minimal_v1_final_recheck/metrics.json"))
    p.add_argument("--output-dir", type=Path, default=Path("results/transformer_pilot_20261018"))
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--suite", choices=("baseline", "speed_ablation"), default="baseline")
    p.add_argument("--finalize-only", action="store_true", help="Check already completed arms without retraining")
    args = p.parse_args()
    if args.finalize_only:
        print_results(finalize_existing(args.output_dir),args.output_dir)
        return
    if args.output_dir.exists() or min(args.steps, args.batch_size, args.eval_every) < 1:
        p.error("Need a NEW output directory and positive settings")
    if "20_trajectory" in str(args.trajectories):
        p.error("Do not consume the fresh formal CP calibration set")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    sources = {str(f): sha256(f) for f in (args.trajectories, args.representation, args.ppo,
        args.spvc_dir/"latent_ppo_spvc.pt", args.spvc_dir/"sbc.pt",
        args.spvc_dir/"metrics.json", args.final_recheck)}
    rep = torch.load(args.representation, map_location="cpu")
    projection, decoder, q = SafetyProjection(), SafetyDecoder(), PhysicalToLatent()
    for model, key in ((projection, "projection_state_dict"),
                       (decoder, "auxiliary_decoder_state_dict"),
                       (q, "physical_to_latent_state_dict")):
        model.load_state_dict(rep[key]); model.to(device).eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
    _, saved_actor, _, actor_evidence = load_spvc_pair(
        args.representation,args.ppo,args.spvc_dir,args.final_recheck)
    actor = saved_actor.policy.controller_net.to(device).eval()
    for parameter in actor.parameters(): parameter.requires_grad_(False)
    scale = float(rep["distance_scale_m"])
    with h5py.File(args.trajectories, "r") as f:
        if not f.attrs.get("complete", False): raise ValueError("Incomplete trajectories")
        raw = f["dino_features"][:]
        valid = f["valid"][:].astype(bool)
        distances, speeds, actions = (f[k][:] for k in ("distance_m", "speed_mps", "image_action"))
        identities = f["trajectory_id"].asstr()[:].tolist()
        appearance = [a+"|"+b for a,b in zip(f["weather"].asstr()[:], f["color"].asstr()[:])]
        original_role = str(f.attrs.get("split_role"))
    if len(set(identities)) != len(identities): raise ValueError("Duplicate trajectory IDs")
    if raw.shape[:2] != valid.shape or raw.shape[-1] != 384:
        raise ValueError("Expected global DINO features [N,T,384]")
    if (not np.isfinite(raw[valid]).all() or not np.isfinite(speeds[valid]).all()
            or not np.isfinite(actions[valid]).all() or not np.isfinite(distances[valid]).all()):
        raise ValueError("Nonfinite features/states")
    # Windows crossing timing/state resets are inadmissible, even within one ID.
    adjacent = valid[:, :-1] & valid[:, 1:]
    d_error = np.abs(distances[:, 1:] - (distances[:, :-1] - .05*speeds[:, :-1]))
    v_expected = np.clip(speeds[:, :-1] - .05*np.clip(actions[:, :-1], -3, 3), 0, 3)
    v_error = np.abs(speeds[:, 1:] - v_expected)
    if (d_error[adjacent] > 5e-4).any() or (v_error[adjacent] > 5e-4).any():
        raise ValueError("Nonconsecutive physics records; cannot fabricate temporal windows")
    train_ids, val_ids = split_trajectories(len(valid), args.seed)
    tr, va = (window_indices(valid, ids) for ids in (train_ids, val_ids))
    if not len(tr) or not len(va): raise ValueError("No complete 3-context/3-target windows")
    mean, std = rep["feature_mean"].to(device), rep["feature_std"].to(device)
    features = (torch.from_numpy(np.nan_to_num(raw)).to(device) - mean) / std
    with torch.no_grad():
        z = projection(features.flatten(0, 1)).reshape(*valid.shape, 32)
    data = {"latent32": z, "dino384": features}
    speed = torch.from_numpy(np.nan_to_num(speeds)).to(device)
    action = torch.from_numpy(np.nan_to_num(actions)).to(device)
    distance = torch.from_numpy(np.nan_to_num(distances)).to(device)
    # Native-space normalization is fitted to TRAIN trajectories only.
    normalization = {}
    for name, arr in data.items():
        rows = arr[train_ids][torch.as_tensor(valid[train_ids], device=device)]
        normalization[name] = (rows.mean(0), rows.std(0).clamp_min(.01))
    args.output_dir.mkdir(parents=True)
    arms = learning_arms(args.suite)
    config = dict(experiment="causal_CLS_world_model_development_AB", seed=args.seed,
        suite=args.suite, learning_arms=arms, diagnostic_actor_uses_true_speed=True,
        context=3, horizon=3, steps=args.steps, batch_size=args.batch_size,
        learning_rate=.0003, transformer_width=128, transformer_layers=2,
        attention_heads=4, source_sha256=sources, script_sha256=sha256(Path(__file__)),
        frozen_actor_identity=actor_evidence,
        module_sha256=sha256(Path(__file__).with_name("temporal_world_model.py")),
        trajectory_count=len(valid), train_trajectories=len(train_ids), validation_trajectories=len(val_ids),
        train_windows=len(tr), validation_windows=len(va), original_input_role=original_role,
        input_repurposed_for_development=True, future_CP_must_not_reuse_this_data=True,
        test_used=False, fresh_CP_data_used=False, representation_frozen=True, ppo_frozen=True,
        paper_adaptation="DINO CLS 384, NOT full 16x16 patch tokens; no paper-controller/barrier reproduction",
        horizon_actions="observed/planned future controls supplied to every arm equally; open-loop prediction",
        selection="validation 3-horizon mean common frozen-R2 latent RMSE",
        limitations=["same-policy old trajectories, not random-action coverage", "no new PPO/SBC/online rollout",
                     "prediction comparison not a formal safety guarantee", "models not parameter-count matched"],
        max_record_distance_error_m=float(d_error[adjacent].max()),
        max_record_speed_error_mps=float(v_error[adjacent].max()))
    config["physical_diagnostic_scope"] = (
        "At each predicted observation time, use its true speed ONLY for the frozen-PPO diagnostic; "
        "compare predicted-latent and real-image-latent actions in the SAME state. "
        "Successor gap is one physical step AFTER that observation, not the forecast speed target. "
        "No true future speed/latent/distance is supplied to learned predictors."
    )
    split = dict(train=[identities[i] for i in train_ids], validation=[identities[i] for i in val_ids],
                 appearances={identities[i]: appearance[i] for i in range(len(valid))})
    (args.output_dir/"config.json").write_text(json.dumps(config, indent=2))
    (args.output_dir/"split.json").write_text(json.dumps(split, indent=2))
    context_offsets = torch.arange(-2, 1, device=device)
    target_offsets = torch.arange(1, 4, device=device)
    action_offsets = torch.arange(-2, 3, device=device)

    def batch(native, windows):
        w = torch.as_tensor(windows, dtype=torch.long, device=device)
        m, s = normalization[native]
        history = (gather(data[native], w, context_offsets) - m) / s
        target = (gather(data[native], w, target_offsets) - m) / s
        return history, gather(speed, w, context_offsets), gather(action, w, action_offsets), target, w

    def common(native, standardized):
        m, s = normalization[native]
        native_prediction = standardized * s + m
        return projection(native_prediction) if native == "dino384" else native_prediction

    @torch.no_grad()
    def selection(model, native):
        squares = 0.; count = 0
        for start in range(0, len(va), 1024):
            hist, v, a, _, w = batch(native, va[start:start+1024])
            pred = common(native, rollout(model, hist, v, a, 3))
            actual = gather(z, w, target_offsets)
            squares += float((pred-actual).square().sum()); count += actual.numel()
        return (squares/count)**.5

    @torch.no_grad()
    def evaluate(model, native, oracle_q=False):
        saved = {h: {k: [] for k in ("latent_rms", "distance_error", "distance_reference_gap",
                  "action_gap", "successor_speed_gap")} for h in range(1,4)}
        for start in range(0, len(va), 1024):
            hist, v, a, _, w = batch(native, va[start:start+1024])
            target_z = gather(z, w, target_offsets)
            d = gather(distance, w, target_offsets)
            v_true = gather(speed, w, target_offsets)
            pred_z = q(d[..., None]/scale) if oracle_q else common(native, rollout(model,hist,v,a,3))
            d_pred = decoder(pred_z).squeeze(-1)*scale
            d_ref = decoder(target_z).squeeze(-1)*scale
            def actions_for(latents):
                observation = torch.cat((latents,v_true[...,None]),dim=-1).flatten(0,1)
                return actor(observation).reshape_as(v_true).clamp(-3,3)
            ap, ar = actions_for(pred_z), actions_for(target_z)
            vp = (v_true-.05*ap).clamp(0,3); vr = (v_true-.05*ar).clamp(0,3)
            vals = dict(latent_rms=(pred_z-target_z).square().mean(-1).sqrt(),
                        distance_error=d_pred-d, distance_reference_gap=d_pred-d_ref,
                        action_gap=ap-ar, successor_speed_gap=vp-vr)
            for h in range(1,4):
                for key,value in vals.items(): saved[h][key].append(value[:,h-1].cpu().numpy())
        result = {str(h): {k: summary(np.concatenate(v)) for k,v in fields.items()} for h,fields in saved.items()}
        # Group only after model selection; these are not extra training targets.
        for h in range(1,4):
            velocity = speeds[va[:,0],va[:,1]+h]
            groups = {}
            for label,lo,hi in (("low_0_to_0.5",0.,.5),("middle_0.5_to_1.5",.5,1.5),("high_1.5_to_3",1.5,3.001)):
                keep = (velocity >= lo) & (velocity < hi)
                if keep.any():
                    groups[label] = dict(count=int(keep.sum()), **{
                        key:summary(np.concatenate(value)[keep]) for key,value in saved[h].items()})
            result[str(h)]["by_target_speed"] = groups
        return result

    results = {"persistence32": dict(kind="no training; last image latent repeated", metrics=evaluate(None,"latent32")),
               "current_q_known_distance": dict(kind="current proxy; true FUTURE distance supplied, not a forecast competitor",
                                                 metrics=evaluate(None,"latent32",True))}
    for arm in arms:
        name, native, kind = (arm[k] for k in ("name","native","kind"))
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        model = TemporalPredictor(data[native].shape[-1],kind,
            speed_mode=arm["speed_mode"],use_action=arm["use_action"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
        best_score = selection(model,native); best = copy.deepcopy(model.state_dict()); best_step = 0
        history = []; arm_start = time.monotonic()
        for step in range(1,args.steps+1):
            model.train()
            ids = rng.integers(len(tr),size=args.batch_size)
            hist,v,a,target,_ = batch(native,tr[ids])
            pred = rollout(model,hist,v,a,3)
            loss = F.mse_loss(pred,target)
            if not torch.isfinite(loss): raise ValueError("Nonfinite loss")
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if step==1 or step%args.eval_every==0 or step==args.steps:
                model.eval(); score = selection(model,native)
                if score < best_score:
                    best_score,best,best_step = score,copy.deepcopy(model.state_dict()),step
                history.append(dict(step=step,train_native_mse=float(loss),validation_common_rmse=score))
                print("%s step=%d common_R2_RMSE=%.7f train_native_MSE=%.7f" % (name,step,score,float(loss)),flush=True)
        model.load_state_dict(best); model.eval()
        torch.save(dict(state_dict=best,normalization=normalization,native=native,kind=kind,arm=arm,config=config),
                   args.output_dir/(name+".pt"))
        results[name] = dict(arm=arm,best_step=best_step,parameter_count=sum(p.numel() for p in model.parameters()),
                             runtime_seconds=time.monotonic()-arm_start,history=history,metrics=evaluate(model,native))
        (args.output_dir/"metrics.json").write_text(json.dumps(dict(config=config,results=results),indent=2))
    print_results(finalize_existing(args.output_dir),args.output_dir)


if __name__ == "__main__":
    main()
