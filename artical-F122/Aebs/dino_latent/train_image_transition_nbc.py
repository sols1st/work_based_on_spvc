"""Offline single-step Model-Free NBC component on frozen image transitions.

This implements signed barrier learning, NOT PPO-Barrier joint optimization.
Stored successors were produced by the analytic AEBS simulator with real-image
actions, not CARLA vehicle dynamics. No dynamics gradients or extra noise box.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from Aebs.dino_latent.diagnose_environment_barrier import write_json
from Aebs.dino_latent.prepare_expanded_data import sha256


class SignedNBC(nn.Module):
    def __init__(self, scale, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.register_buffer("center", torch.tensor([10.5/scale, 1.5]))
        self.register_buffer("span", torch.tensor([5.5/scale, 1.5]))

    def forward(self, states):
        return self.net((states-self.center)/self.span).flatten()


def region_grids(scale, size, device):
    def grid(distance, velocity):
        d, v = torch.meshgrid(torch.linspace(*distance, size, device=device),
                             torch.linspace(*velocity, size, device=device), indexing="ij")
        return torch.stack((d.flatten()/scale, v.flatten()), dim=1)
    return grid((15.,16.), (2.5,3.)), grid((5.,6.), (.5001,3.))


def barrier_losses(model, states, successors, initial, unsafe, barrier_lambda, epsilon, hard_fraction=0.):
    violation = F.relu(model(successors)-(1.-barrier_lambda)*model(states)+epsilon)
    mean_inv = violation.mean()
    if hard_fraction:
        k = max(1, int(np.ceil(len(violation)*hard_fraction)))
        inv = .5*mean_inv+.5*violation.topk(k).values.mean()
    else:
        inv = mean_inv
    initial_loss = F.relu(model(initial)+epsilon).mean()
    unsafe_loss = F.relu(epsilon-model(unsafe)).mean()
    return initial_loss+unsafe_loss+inv, dict(initial=initial_loss, unsafe=unsafe_loss, invariance=mean_inv, optimized_invariance=inv)


def evaluate_pairs(model, states, successors, old_current, old_next, scale, barrier_lambda, epsilon, batch_size, device):
    current = np.empty(len(states), dtype=np.float32)
    following = np.empty_like(current)
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            end = min(start+batch_size, len(states))
            current[start:end] = model(torch.from_numpy(states[start:end]).to(device)).cpu().numpy()
            following[start:end] = model(torch.from_numpy(successors[start:end]).to(device)).cpu().numpy()
    margin = (1.-barrier_lambda)*current-following
    inside = current <= 0
    next_unsafe = (successors[:, 0]*scale <= 6.) & (successors[:, 1] > .5)
    def subset(mask):
        ids = np.flatnonzero(mask)
        if not len(ids):
            return dict(pairs=0)
        return dict(pairs=len(ids), raw_invariance_violations=int((margin[ids] < -1e-6).sum()),
                    raw_invariance_violation_rate=float((margin[ids] < -1e-6).mean()),
                    training_margin_violations=int((margin[ids] < epsilon-1e-6).sum()),
                    min_invariance_margin=float(margin[ids].min()), mean_invariance_margin=float(margin[ids].mean()),
                    sign_escapes=int(((current[ids] <= 0) & (following[ids] > 0)).sum()),
                    unsafe_successors=int(next_unsafe[ids].sum()))
    worst = np.argsort(margin, kind="stable")[:5]
    result = dict(all_pairs=subset(np.ones(len(states), bool)), candidate_nonpositive_region=subset(inside),
                  candidate_region_fraction=float(inside.mean()),
                  candidate_region_with_unsafe_successor=int((inside & next_unsafe).sum()),
                  observed_unsafe_successor_nonpositive_nbc=int((next_unsafe & (following <= 0)).sum()),
                  old_sbc_deterministic_decrease_violations=int(((old_current-old_next) <= 0).sum()),
                  old_vs_nbc_condition_warning="different signed/unsigned conditions; these counts are not an apples-to-apples certificate-strength comparison",
                  worst_cases=[dict(state=[float(states[i,0]*scale), float(states[i,1])],
                                    next_state=[float(successors[i,0]*scale), float(successors[i,1])],
                                    current_nbc=float(current[i]), next_nbc=float(following[i]),
                                    margin=float(margin[i]), next_unsafe=bool(next_unsafe[i])) for i in worst])
    return result, current, following


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, default=Path("results/environment_nbc_20261013_001/01_frozen_diagnostic"))
    parser.add_argument("--spvc-dir", type=Path, default=Path("results/dino_spvc_minimal_v1"))
    parser.add_argument("--representation", type=Path, default=Path("results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-nbc", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--barrier-lambda", type=float, default=.1)
    parser.add_argument("--epsilon", type=float, default=.01)
    parser.add_argument("--hard-fraction", type=float, default=0.)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cpu","cuda"), default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists() or args.epochs < 1 or args.batch_size < 1 or args.hidden < 2 or not 0 < args.barrier_lambda < 1 or args.epsilon <= 0 or not 0 <= args.hard_fraction <= 1:
        parser.error("new output and valid training configuration required")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.monotonic()
    diagnostic_path = args.diagnostic_dir/"metrics.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    if diagnostic.get("status") != "environment_barrier_diagnostic_complete" or diagnostic.get("source_files_unchanged") is not True or diagnostic.get("test_used") is not False:
        raise ValueError("completed development diagnostic required")
    scale = diagnostic["vt_distance_scale_m"]
    sources = dict(actor=args.spvc_dir/"latent_ppo_spvc.pt", old_sbc=args.spvc_dir/"sbc.pt", representation=args.representation,
                   diagnostic=diagnostic_path, train_pairs=args.diagnostic_dir/"train_pairs.npz", validation_pairs=args.diagnostic_dir/"validation_pairs.npz")
    hashes = {k:sha256(p) for k,p in sources.items()}
    for key, old_key in (("actor","actor"),("old_sbc","barrier"),("representation","representation")):
        if hashes[key] != diagnostic["source_hashes"][old_key]:
            raise ValueError("stored image transitions belong to different frozen models")
    device = torch.device(args.device)
    with np.load(sources["train_pairs"], allow_pickle=False) as train:
        x_np, y_np = train["state"], train["image_next_state"]
        if x_np.shape != y_np.shape or not np.isfinite(x_np).all() or not np.isfinite(y_np).all():
            raise ValueError("invalid train transitions")
        train_image_rows = np.unique(train["image_row"])
        train_distances = np.unique(train["distance_m"])
    x, y = torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)
    # No computational path from stored successor to the frozen policy/dynamics.
    x.requires_grad_(False)
    y.requires_grad_(False)
    model = SignedNBC(scale, args.hidden).to(device)
    initial, unsafe = region_grids(scale, 32, device)
    init_hash = None
    if args.initial_nbc:
        init_hash = sha256(args.initial_nbc)
        checkpoint = torch.load(args.initial_nbc, map_location=device)
        if checkpoint["source_hashes"] != hashes or checkpoint["hidden"] != args.hidden or checkpoint["barrier_lambda"] != args.barrier_lambda or checkpoint["epsilon"] != args.epsilon or checkpoint["vt_distance_scale_m"] != scale:
            raise ValueError("warm-start NBC configuration or source identity differs")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    report = dict(experiment="offline_image_transition_single_step_NBC_v1", status="running", training_run=True,
        policy_updated=False, old_sbc_updated=False, representation_updated=False, ppo_joint_training=False,
        multi_step_training=False, test_used=False, cp_used=False, extra_noise_used=False,
        source_hashes=hashes, script_sha256=sha256(Path(__file__)), initial_nbc_sha256=init_hash,
        vt_distance_scale_m=scale, configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        paper_component="Model-Free NBC IV-B single-step invariant/initial/unsafe losses; positive margins inspired by official srlnbc implementation",
        not_implemented="PPO importance-ratio safety update, adaptive multiplier, multi-step trajectories, formal validation",
        condition="C(next) <= (1-lambda)*C(current); candidate region C<=0, unsafe C>0; signed linear output",
        data_scope="exact-image offline simulator transitions at labeled distances; weather/color affect actual saved actions; no nearest-image lookup",
        simulator_scope="analytic AEBS successors recorded previously, not measured CARLA vehicle dynamics",
        label_scope="known task initial [15,16]x[2.5,3]; unsafe [5,6]x[0.5001,3]; no hand-computed recoverable/TTC labels",
        loss="mean initial relu(C+epsilon) + mean unsafe relu(epsilon-C) + invariant hinge; optional train-only hard-tail emphasis",
        train_region_grid=32, evaluation_region_grid=41,
        validation_role="reused development split, not independent safety calibration",
        source_safety_probability_reused=False, history=[], results={})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir/"metrics.json", report)
    try:
        for epoch in range(1,args.epochs+1):
            model.train()
            order = torch.randperm(len(x),device=device)
            totals = {k:0. for k in ("loss","initial","unsafe","invariance","optimized_invariance")}
            for start in range(0,len(x),args.batch_size):
                ids = order[start:start+args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss, parts = barrier_losses(model,x[ids],y[ids],initial,unsafe,args.barrier_lambda,args.epsilon,args.hard_fraction)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                totals["loss"] += float(loss.detach())*len(ids)
                for k,v in parts.items(): totals[k] += float(v.detach())*len(ids)
            if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
                row = dict(epoch=epoch,**{k:v/len(x) for k,v in totals.items()})
                report["history"].append(row)
                print("single-step NBC",row,flush=True)
                write_json(args.output_dir/"metrics.json",report)
        checkpoint_path = args.output_dir/"nbc.pt"
        torch.save(dict(state_dict=model.cpu().state_dict(),hidden=args.hidden,vt_distance_scale_m=scale,
                        barrier_lambda=args.barrier_lambda,epsilon=args.epsilon,source_hashes=hashes,
                        scope="offline signed NBC candidate; no full algorithm or safety certificate"),checkpoint_path)
        saved = torch.load(checkpoint_path,map_location=device)
        model = SignedNBC(scale,args.hidden).to(device).eval()
        model.load_state_dict(saved["state_dict"],strict=True)
        ei,eu=region_grids(scale,41,device)
        with torch.no_grad():
            ci,cu=model(ei).cpu().numpy(),model(eu).cpu().numpy()
        report["region_checks"]=dict(initial_states=len(ci),initial_max=float(ci.max()),
            initial_raw_violations=int((ci>0).sum()),initial_margin_violations=int((ci>-args.epsilon+1e-6).sum()),
            unsafe_states=len(cu),unsafe_min=float(cu.min()),unsafe_raw_violations=int((cu<=0).sum()),
            unsafe_margin_violations=int((cu<args.epsilon-1e-6).sum()))
        for split in ("train","validation"):
            with np.load(sources[split+"_pairs"],allow_pickle=False) as data:
                if split == "validation" and (np.intersect1d(train_image_rows,np.unique(data["image_row"])).size or np.intersect1d(train_distances,np.unique(data["distance_m"])).size):
                    raise ValueError("train/validation overlap")
                result,current,following=evaluate_pairs(model,data["state"],data["image_next_state"],data["current_barrier"],
                    data["image_next_barrier"],scale,args.barrier_lambda,args.epsilon,args.batch_size,device)
                report["results"][split]=result
                np.savez_compressed(args.output_dir/(split+"_nbc_values.npz"),current=current,following=following)
                print(split, "NBC checks", result,flush=True)
                write_json(args.output_dir/"metrics.json",report)
        report["source_files_unchanged"]={k:sha256(p) for k,p in sources.items()}==hashes
        if not report["source_files_unchanged"]: raise ValueError("source changed during experiment")
        passed=report["region_checks"]["initial_raw_violations"]==0 and report["region_checks"]["unsafe_raw_violations"]==0 and all(r["all_pairs"]["raw_invariance_violations"]==0 for r in report["results"].values())
        report.update(status="sampled_NBC_properties_pass" if passed else "development_nbc_constraints_violated",
                      checkpoint=str(checkpoint_path),checkpoint_sha256=sha256(checkpoint_path),safety_certificate=False)
    except BaseException as exc:
        report.update(status="failed",error=repr(exc))
        raise
    finally:
        report["runtime_seconds"]=time.monotonic()-started
        write_json(args.output_dir/"metrics.json",report)
    print("\n[Image-transition single-step NBC component: policy frozen]")
    print("status:",report["status"],"regions:",report["region_checks"])
    print("No PPO safety-gradient update, multi-step training or safety certificate.")
    print("metrics:",args.output_dir/"metrics.json",flush=True)


if __name__=="__main__": main()
