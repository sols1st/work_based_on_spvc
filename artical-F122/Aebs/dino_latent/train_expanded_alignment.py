"""R0/R1 joint 32-D alignment from frozen DINO H5. No PPO/SBC/test use."""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from Aebs.dino_latent.models import SafetyProjection, SafetyDecoder, PhysicalToLatent
from Aebs.dino_latent.prepare_expanded_data import sha256


def training_ids(rows, baseline=False):
    return [i for i,r in enumerate(rows) if r["split"] == "train" and
            (not baseline or (r["weather"] == "original" and r["color"] == "255,0,0"))]


def balanced_batch(rows, ids, rng, size):
    pools = {}
    for i in ids:
        pools.setdefault(rows[i]["distance_bin_025m"], []).append(i)
    # Return reusable pools rather than rebuilding for every optimizer step.
    return [np.asarray(v, dtype=np.int64) for _,v in sorted(pools.items())]


def draw(pools, rng, size):
    return np.asarray([rng.choice(pools[g]) for g in rng.integers(len(pools), size=size)])


def metrics(z, q, pred, physical_pred, distance, rows):
    error = z-q
    rmse = float(np.sqrt(np.mean(error**2)))
    scale = float(np.sqrt(np.mean((z-z.mean(0))**2)))
    norms = np.linalg.norm(error, axis=1)
    result = dict(samples=len(z), latent_rmse=rmse, latent_centered_rms=scale,
                  relative_latent_rmse=rmse/scale if scale > 1e-8 else None,
                  collapsed=scale <= 1e-8,
                  residual_l2_p95=float(np.quantile(norms,.95)), residual_l2_max=float(norms.max()),
                  image_distance_mae_m=float(np.abs(pred-distance).mean()),
                  image_distance_rmse_m=float(np.sqrt(np.mean((pred-distance)**2))),
                  physical_distance_rmse_m=float(np.sqrt(np.mean((physical_pred-distance)**2))))
    perm = np.random.default_rng(123).permutation(len(z))
    far = np.abs(distance-distance[perm]) >= 1
    separation = np.linalg.norm(z-z[perm], axis=1)/np.sqrt(z.shape[1])
    result["far_pair_margin_violation_fraction"] = float(np.mean(separation[far]<.5)) if far.any() else None
    by_group = {}
    for i,r in enumerate(rows):
        by_group.setdefault(r["distance_bin_025m"], []).append(i)
    result["group_macro_latent_rmse"] = float(np.mean([
        np.sqrt(np.mean(error[ids]**2)) for ids in by_group.values()]))
    result["by_appearance"] = {}
    for key in sorted({r["weather"]+"|"+r["color"] for r in rows}):
        ids = [i for i,r in enumerate(rows) if r["weather"]+"|"+r["color"] == key]
        result["by_appearance"][key] = dict(count=len(ids), latent_rmse=float(np.sqrt(np.mean(error[ids]**2))),
            residual_l2_p95=float(np.quantile(norms[ids],.95)))
    result["by_distance_group"] = {str(g): dict(count=len(ids), latent_rmse=float(np.sqrt(np.mean(error[ids]**2))),
        residual_l2_max=float(norms[ids].max())) for g,ids in sorted(by_group.items())}
    worst = np.argsort(norms)[-10:][::-1]
    result["worst_samples"] = [dict(filename=rows[i]["filename"], distance_m=float(distance[i]),
                                   residual_l2=float(norms[i])) for i in worst]
    return result


def evaluate(p, decoder, q, f, distance, scale, ids, rows):
    with torch.no_grad():
        z = p(f[ids]); physical = q(distance[ids,None]/scale)
        args = [v.cpu().numpy() for v in [z,physical,decoder(z).flatten()*scale,
                                         decoder(physical).flatten()*scale,distance[ids]]]
    return metrics(*args, [rows[i] for i in ids])


def train(name, raw, rows, args, steps, provenance):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ids = training_ids(rows, name == "R0_original")
    val = [i for i,r in enumerate(rows) if r["split"] == "validation"]
    if not ids or not val:
        raise ValueError("Empty training or validation subset")
    mean = raw[ids].mean(0); std = raw[ids].std(0).clamp_min(1e-5)
    f = (raw-mean)/std
    d = torch.tensor([r["distance_m"] for r in rows], dtype=torch.float32, device=raw.device)
    scale = float(d[ids].std(unbiased=False))
    if scale <= 0:
        raise ValueError("Zero distance scale")
    p, decoder, q = SafetyProjection().to(raw.device), SafetyDecoder().to(raw.device), PhysicalToLatent().to(raw.device)
    parameters = list(p.parameters())+list(decoder.parameters())+list(q.parameters())
    opt = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-4)
    pools = balanced_batch(rows, ids, rng, args.batch_size)
    best, best_score, history = None, float("inf"), []
    start = time.monotonic()
    for step in range(1, steps+1):
        batch = draw(pools, rng, args.batch_size)
        y = d[batch]/scale
        z = p(f[batch]); physical = q(y[:,None])
        perm = torch.randperm(len(batch), device=raw.device)
        far = (y-y[perm]).abs()*scale >= 1
        separation = (z-z[perm]).norm(dim=1)/math.sqrt(32)
        sep = torch.relu(.5-separation[far]).square().mean() if bool(far.any()) else z.new_tensor(0.)
        loss = (F.mse_loss(decoder(z).flatten(),y)+F.mse_loss(decoder(physical).flatten(),y)
                +F.mse_loss(z,physical)+.1*sep+1e-4*z.square().mean())
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters,5.)
        opt.step()
        if step == 1 or step % args.eval_every == 0 or step == steps:
            m = evaluate(p,decoder,q,f,d,scale,val,rows)
            relative = m["relative_latent_rmse"]
            score = m["image_distance_rmse_m"]/scale+.2*relative if relative is not None else float("inf")
            if not math.isfinite(score):
                raise RuntimeError("Collapsed/nonfinite validation representation")
            if score < best_score:
                best_score = score
                best = tuple(copy.deepcopy(net.state_dict()) for net in (p,decoder,q))
                best_step = step
            history.append(dict(step=step, loss=float(loss), score=score,
                                latent_rmse=m["latent_rmse"], relative_latent_rmse=relative))
            print("%s step=%d/%d val_relative=%.6f p95=%.6f image_rmse_m=%.6f" %
                  (name,step,steps,relative,m["residual_l2_p95"],m["image_distance_rmse_m"]), flush=True)
    for net,state in zip((p,decoder,q),best):
        net.load_state_dict(state)
    # Equal number of q-only updates; projection remains fixed.
    with torch.no_grad():
        ztrain = p(f[ids]).detach()
        zval = p(f[val]).detach()
        qscore = float(F.mse_loss(q(d[val,None]/scale),zval))
    qbest, qstep = copy.deepcopy(q.state_dict()), 0
    optq = torch.optim.AdamW(q.parameters(),lr=1e-3,weight_decay=1e-5)
    for step in range(1,args.q_steps+1):
        loss = F.mse_loss(q(d[ids,None]/scale),ztrain)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite q fitting loss")
        optq.zero_grad(); loss.backward(); optq.step()
        with torch.no_grad():
            value = float(F.mse_loss(q(d[val,None]/scale),zval))
        if value < qscore:
            qscore,qbest,qstep = value,copy.deepcopy(q.state_dict()),step
        if step == 1 or step % 100 == 0:
            print("%s q_fit=%d/%d" % (name,step,args.q_steps),flush=True)
    q.load_state_dict(qbest)
    result = dict(variant=name, train_samples=len(ids), joint_steps=steps, q_steps=args.q_steps,
                  best_joint_step=best_step,best_q_step=qstep,history=history,
                  validation=evaluate(p,decoder,q,f,d,scale,val,rows),
                  train=evaluate(p,decoder,q,f,d,scale,ids,rows),
                  runtime_seconds=time.monotonic()-start,test_evaluated=False)
    folder = args.output_dir/name
    folder.mkdir()
    torch.save(dict(projection_state_dict=p.cpu().state_dict(),auxiliary_decoder_state_dict=decoder.cpu().state_dict(),
        physical_to_latent_state_dict=q.cpu().state_dict(),feature_mean=mean.cpu(),feature_std=std.cpu(),
        distance_scale_m=scale,latent_dimension=32,dino_frozen=True,variant=name,provenance=provenance),
        folder/"dino_safety_latent.pt")
    (folder/"metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir",type=Path,default=Path("results/dino_expanded_v1/02_dino_h5"))
    p.add_argument("--output-dir",type=Path,default=Path("results/dino_expanded_v1/03_alignment_r0_r1"))
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--steps",type=int,default=0,help="0: 250*ceil(R0_train/32); same for both arms")
    p.add_argument("--q-steps",type=int,default=400)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--eval-every",type=int,default=250)
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--visual-review-confirmed",action="store_true")
    args = p.parse_args()
    if not args.visual_review_confirmed:
        p.error("Confirm screenshot review with --visual-review-confirmed")
    if args.output_dir.exists() or args.steps < 0 or min(args.q_steps,args.batch_size,args.eval_every)<1:
        p.error("Use a new output directory and valid training parameters")
    cache = json.loads((args.cache_dir/"cache_manifest.json").read_text())
    if cache.get("schema") != "spvc_expanded_dino_h5_cache_v1" or cache.get("test_encoded") != 0:
        p.error("Expected no-test H5 cache")
    rows = cache["records"]
    if any(r["split"] not in ("train","validation") for r in rows):
        p.error("Test rows forbidden")
    path = args.cache_dir/"features.h5"
    if sha256(path) != cache["feature_sha256"]:
        p.error("H5 hash mismatch")
    with h5py.File(path,"r") as stream:
        if not stream.attrs["complete"] or stream["features"].shape != (len(rows),384):
            p.error("Invalid H5")
        for key in ("filename","split","weather","color","group_id"):
            if stream[key].asstr()[:].tolist() != [r[key] for r in rows]:
                p.error("H5 row metadata mismatch: "+key)
        for key in ("distance_m", "distance_bin_025m"):
            if not np.allclose(stream[key][:], [r[key] for r in rows], rtol=0, atol=1e-9):
                p.error("H5 numerical labels mismatch: "+key)
        raw = np.asarray(stream["features"],dtype=np.float32)
    if not np.isfinite(raw).all():
        p.error("Nonfinite features")
    torch.set_num_threads(1)
    steps = args.steps or 250*math.ceil(len(training_ids(rows,True))/args.batch_size)
    args.output_dir.mkdir(parents=True)
    config = dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        joint_steps_per_arm=steps,visual_review="user_confirmed",test_evaluated=False,
        cache_manifest_sha256=sha256(args.cache_dir/"cache_manifest.json"),
        cache_provenance=cache["provenance"],script_sha256=sha256(Path(__file__)),
        selection="validation normalized image distance RMSE + 0.2 relative latent RMSE",
        loss_note="Both arms omit historical synthetic-view invariance: cache contains original RGB features only. No R2 paired loss.")
    (args.output_dir/"config.json").write_text(json.dumps(config,indent=2))
    raw = torch.from_numpy(raw).to(args.device)
    results = {name:train(name,raw,rows,args,steps,config) for name in ["R0_original","R1_all_appearances"]}
    (args.output_dir/"comparison.json").write_text(json.dumps(results,indent=2,allow_nan=False))
    print("[Expanded latent alignment R0/R1 — no PPO/SBC/test]")
    for name,result in results.items():
        for split in ["train","validation"]:
            print(name,split,json.dumps({k:v for k,v in result[split].items() if not isinstance(v,(dict,list))}))
    a,b = [results[n]["validation"]["relative_latent_rmse"] for n in results]
    print("validation relative error reduction: %s" % ((a-b)/a if a and b is not None else None))
    print("comparison: %s" % (args.output_dir/"comparison.json"))
    print("Empirical errors only; no safety or error-coverage guarantee.")


if __name__ == "__main__":
    main()
