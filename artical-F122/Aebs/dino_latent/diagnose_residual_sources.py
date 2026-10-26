"""Read-only residual-source audit on OLD development trajectories.

No calibration claim, model selection, model update or closed-loop evaluation.
Oracle interpolation isolates mechanisms; it is NOT a deployable observation.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from Aebs.conformal.calibrate_controller import select_tolerance_rank
from Aebs.dino_latent.models import SafetyProjection, PhysicalToLatent
from Aebs.dino_latent.spvc_saved_actor import load_spvc_pair
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.calibrate_transition_contract import next_state_residual
from Aebs.dino_latent.error_tail_metrics import tail_summary, trajectory_tail_summary
from Aebs.dino_latent.error_tail_metrics import align_rows


def appearance_floor(distance, speeds, actions, valid):
    """Minimax lower bound for ANY one deterministic next-speed proxy per d,v.

    For observed outputs in [lo,hi], the smallest possible largest absolute
    error of a point proxy is (hi-lo)/2. This is a static observed-family
    fact, NOT a trajectory-quantile lower bound or new CP result.
    """
    distance=np.asarray(distance,dtype=np.float64)
    speeds=np.asarray(speeds,dtype=np.float64)
    actions=np.asarray(actions,dtype=np.float64)
    valid=np.asarray(valid,dtype=bool)
    if actions.shape!=(len(distance),len(speeds)) or valid.shape!=actions.shape:
        raise ValueError("Bad paired appearance arrays")
    outputs=np.clip(speeds[None,:]-.05*np.clip(actions,-3,3),0,3)
    radii=[]; states=[]; sizes=[]
    for d in np.unique(distance):
        ix=distance==d
        if ix.sum()<2: continue
        eligible=valid[ix].all(0)
        lo,hi=outputs[ix].min(0),outputs[ix].max(0)
        radii.extend(((hi-lo)*.5)[eligible].tolist())
        states.extend([[float(d),float(v)] for v in speeds[eligible]])
        sizes.extend([int(ix.sum())]*int(eligible.sum()))
    if not radii: raise ValueError("No common-state appearance pairs")
    return np.array(radii),np.array(states),np.array(sizes)


def smoothness_penalty(network, observation, create_graph):
    x = observation.detach().clone().requires_grad_(True)
    grad = torch.autograd.grad(network(x).sum(), x, create_graph=create_graph)[0]
    return torch.relu(grad.norm(dim=1) - 2.).mean()


def physical_action_bound(speed, image_action, proxy_action, dt=.05):
    # Use the SAME float64 arithmetic as the registered physical residual,
    # rather than comparing it to a float32-rounded action bound.
    speed, image_action, proxy_action = np.broadcast_arrays(*(
        np.asarray(x,dtype=np.float64) for x in (speed,image_action,proxy_action)))
    residual = next_state_residual(np.zeros_like(speed), speed, image_action, proxy_action, dt)[..., 1]
    bound = dt * np.abs(np.clip(image_action, -3, 3) - np.clip(proxy_action, -3, 3))
    if np.any(np.abs(residual) > bound + 1e-12):
        raise ValueError("Clipped dynamics violated action Lipschitz bound")
    return residual, bound


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectories", type=Path, default=Path("results/dino_expanded_v1/13_spvc_calibration_trajectories_v1/trajectories.h5"))
    p.add_argument("--representation", type=Path, default=Path("results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt"))
    p.add_argument("--ppo", type=Path, default=Path("results/dino_expanded_v1/09_R2_ppo/latent_ppo.zip"))
    p.add_argument("--spvc-dir", type=Path, default=Path("results/dino_spvc_minimal_v1"))
    p.add_argument("--final-recheck", type=Path, default=Path("results/dino_spvc_minimal_v1_final_recheck/metrics.json"))
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists() or "13_spvc_calibration_trajectories_v1" not in str(a.trajectories):
        p.error("Use a new output and the OLD development pool only")
    torch.set_num_threads(1)
    sources = [a.trajectories, a.representation, a.ppo, a.spvc_dir/"latent_ppo_spvc.pt", a.spvc_dir/"sbc.pt"]
    paired_dir=Path("results/dino_spvc_minimal_v1_paired_error_retry1")
    split_path=Path("results/static_latent_benchmark_20261018_qcontrol/split.json")
    sources += [paired_dir/"metrics.json",paired_dir/"samples.npz",split_path]
    hashes = {str(x): sha256(x) for x in sources}
    rep = torch.load(a.representation, map_location="cpu")
    projection, q = SafetyProjection().eval(), PhysicalToLatent().eval()
    projection.load_state_dict(rep["projection_state_dict"])
    q.load_state_dict(rep["physical_to_latent_state_dict"])
    _, current, old, identity = load_spvc_pair(a.representation, a.ppo, a.spvc_dir, a.final_recheck)
    with h5py.File(a.trajectories) as f:
        mask = f["valid"][:].astype(bool); ep, tt = np.where(mask)
        raw = torch.from_numpy(f["dino_features"][:][mask])
        d, v = f["distance_m"][:][mask], f["speed_mps"][:][mask]
        weather, color = (f[k].asstr()[:][ep].astype(str) for k in ("weather", "color"))
    with torch.no_grad():
        zi = projection((raw-rep["feature_mean"])/rep["feature_std"])
        zq = q(torch.from_numpy(d[:, None])/float(rep["distance_scale_m"]))
    zi, zq = zi.detach(), zq.detach()
    delta = zi-zq; l2 = delta.norm(dim=1).numpy()
    def action(net, z):
        with torch.no_grad():
            obs = torch.cat((z, torch.from_numpy(v[:, None])), 1)
            return net(obs).flatten().numpy()
    nets = {"old_PPO09": old.policy.controller_net, "minimal_final": current.policy.controller_net}
    results = {}; raw_actions = {}; residuals = {}
    for name, net in nets.items():
        ai, aq = action(net, zi), action(net, zq)
        raw_actions[name] = (ai, aq)
        r, bound = physical_action_bound(v, ai, aq)
        residuals[name] = r
        gain = np.abs(np.clip(ai,-3,3)-np.clip(aq,-3,3))/np.maximum(l2, 1e-12)
        t, _ = trajectory_tail_summary(r, ep, len(mask), [.05,.1,.15])
        x = torch.cat((zq[:4096], torch.from_numpy(v[:4096,None])),1)
        detached = smoothness_penalty(net, x, False)
        differentiable = smoothness_penalty(net, x, True)
        grads = torch.autograd.grad(differentiable, tuple(net.parameters()), allow_unused=True)
        gnorm = float(sum(g.square().sum() for g in grads if g is not None).sqrt())
        results[name] = dict(step=tail_summary(r,[.05,.1,.15]), trajectory=t,
            effective_action_gain=tail_summary(gain, unit="action per latent L2 unit; not global Lipschitz"),
            action_gap=tail_summary(np.clip(ai,-3,3)-np.clip(aq,-3,3),unit="m/s^2"),
            raw_action_outside_limit=int(((np.abs(ai)>3)|(np.abs(aq)>3)).sum()),
            next_speed_clipping_reduces_bound=int((np.abs(r)<bound-1e-7).sum()),
            gradient_penalty=dict(value=float(detached),original_create_graph_false_requires_grad=detached.requires_grad,
                create_graph_true_parameter_gradient_norm=gnorm),
            oracle_latent_interpolation={})
        for alpha in (0., .25, .5, .75, 1.):
            ar = action(net, zq+alpha*delta)
            rr, _ = physical_action_bound(v, ar, aq)
            results[name]["oracle_latent_interpolation"][str(alpha)] = tail_summary(rr,[.15])
    r = residuals["minimal_final"]
    slices = {}
    for key, labels in (("distance_m",np.digitize(d,[6,7,8,10,12])),
                        ("speed_mps",np.digitize(v,[.5,1,1.5,2,2.5])),
                        ("appearance", np.char.add(np.char.add(weather.astype(str),"|"),color.astype(str)))):
        slices[key] = {}
        for label in np.unique(labels):
            ix = labels==label
            slices[key][str(label)] = dict(residual=tail_summary(r[ix],[.15]), latent_l2=tail_summary(l2[ix],unit="latent L2"),
                distance_range=[float(d[ix].min()),float(d[ix].max())],speed_range=[float(v[ix].min()),float(v[ix].max())])
    top = np.argsort(np.abs(r))[-20:][::-1]
    records = [dict(trajectory=int(ep[i]),time=int(tt[i]),distance_m=float(d[i]),speed_mps=float(v[i]),
        residual_mps=float(r[i]),latent_l2=float(l2[i]),weather=str(weather[i]),color=str(color[i]),
        old_residual_mps=float(residuals["old_PPO09"][i]),
        image_action=float(np.clip(raw_actions["minimal_final"][0][i],-3,3)),
        proxy_action=float(np.clip(raw_actions["minimal_final"][1][i],-3,3))) for i in top]
    # Sensitivity calculation only: changing dt is NOT an improvement of the
    # fixed 20-second task or a reason to weaken its guarantee.
    dt_sensitivity={}
    ai, aq=raw_actions["minimal_final"]
    for dt in (.025,.05,.1):
        rr,_=physical_action_bound(v,ai,aq,dt)
        dt_sensitivity[str(dt)]=dict(speed_residual=tail_summary(rr),acceleration_equivalent=tail_summary(rr/dt,unit="m/s^2"))
    ranks={}
    for n in (459,1000,2000,5000):
        rank,failure=select_tolerance_rank(n,.01,.01)
        ranks[str(n)]=dict(rank_from_largest=rank,confidence=1-failure)
    paired_meta=json.loads((paired_dir/"metrics.json").read_text())
    if paired_meta["hashes"]["dino_saved_actor"]!=identity["new_controller_sha256"]:
        raise ValueError("Static image actions are from a different actor")
    ps=np.load(paired_dir/"samples.npz")
    rows=json.loads(split_path.read_text())["internal_audit"]
    pi=align_rows(rows,ps["feature_row"])
    floor,states,sizes=appearance_floor(ps["distance_m"][pi],ps["speeds_mps"],ps["action_dino_image"][pi],ps["nonterminal"][pi])
    wi=int(np.argmax(floor))
    irreducible=dict(scope="same static development d,v across observed appearances, NOT trajectory CP lower bound",
        images=len(pi),common_state_pairs=len(floor),appearances_min=int(sizes.min()),appearances_max=int(sizes.max()),
        radius_lower_bound=tail_summary(floor,[.025,.05,.1,.15]),worst_state=states[wi].tolist(),
        interpretation="Even an unrestricted deterministic proxy(d,v) needs at least this max-error radius to cover ALL observed appearances at that state; q(d) plus fixed actor is more restricted")
    output=dict(status="development_source_audit_complete_not_new_CP",model_updates=False,
        reference="old13 reference-policy trajectories; NOT current-policy closed loop",test_used=False,
        step20_used=False,source_sha256=hashes,actor_identity=identity,
        states=len(d),trajectories=len(mask),latent_l2=tail_summary(l2,unit="latent L2"),
        actors=results,slices=slices,top20=records,dt_sensitivity=dt_sensitivity,CP_sample_size_ranks=ranks,
        static_appearance_minimax_floor=irreducible,
        notes=["oracle latent interpolation uses true-state q; cannot deploy it without privileged state",
               "pointwise observed gain is not a certified global sensitivity bound",
               "no claim that smaller residual alone implies task success or SBC feasibility"])
    if any(sha256(Path(k))!=val for k,val in hashes.items()): raise ValueError("Source changed")
    a.output_dir.mkdir(parents=True)
    (a.output_dir/"metrics.json").write_text(json.dumps(output,indent=2,allow_nan=False))
    np.savez_compressed(a.output_dir/"samples.npz",distance_m=d,speed_mps=v,trajectory=ep,time=tt,
        latent_l2=l2,old_residual=residuals["old_PPO09"],current_residual=r,weather=weather,color=color)
    print("[Residual source audit: OLD development data, no training]")
    for name, result in results.items():
        print(name,"max residual=",result["step"]["max_abs"],"penalty=",result["gradient_penalty"],flush=True)
    print("worst:",records[0]); print("metrics:",a.output_dir/"metrics.json")


if __name__ == "__main__": main()
