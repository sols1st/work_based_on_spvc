"""Precision-oriented lane controller fine tuning and honest trajectory metrics.

This is model-based differentiation of an existing PPO-exported actor, NOT a
new PPO run or a native SPVC reproduction. Both arms retain the same frozen
visual observer and barrier. A changed actor requires fresh camera CP and a
fresh barrier audit. PGD is a training heuristic, never a verification proof.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

ARMS = ('nominal_precision', 'adversarial_precision')
SEEDS = (202611200701, 202611200702, 202611200703)
TOLERANCES = (.02, .05, .10, .20)
DEADLINE_STEPS = 20


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def precision_metrics(states, actions, deadline=DEADLINE_STEPS):
    """Physical [N,H+1,2] states, normalized [N,H] actions; no dropped failures.

    Strict success means safe over the WHOLE episode and inside the tolerance
    at EVERY sampled time from the deadline through H, not just the last frame.
    """
    states, actions = np.asarray(states, float), np.asarray(actions, float)
    if (states.ndim != 3 or states.shape[2] != 2 or not len(states)
            or actions.shape != (len(states), states.shape[1] - 1)
            or not 0 <= deadline < states.shape[1]
            or not np.isfinite(states).all() or not np.isfinite(actions).all()):
        raise ValueError('finite complete trajectories and valid deadline required')
    unsafe = (np.abs(states / [.85, .30]) > 1).any(axis=(1, 2))
    lateral = np.abs(states[:, :, 0])
    result = dict(unsafe=unsafe, rmse_e_m=np.sqrt((states[:, 1:, 0] ** 2).mean(1)),
                  mean_abs_e_m=lateral[:, 1:].mean(1), peak_abs_e_m=lateral.max(1),
                  tail_mean_abs_e_m=lateral[:, deadline:].mean(1),
                  tail_peak_abs_e_m=lateral[:, deadline:].max(1),
                  mean_abs_heading_rad=np.abs(states[:, 1:, 1]).mean(1),
                  mean_steering_change_rad=(np.abs(np.diff(actions * .45, axis=1)).mean(1)
                      if actions.shape[1] > 1 else np.zeros(len(states))))
    for tolerance in TOLERANCES:
        key = 'strict_%02dcm' % round(tolerance * 100)
        result[key] = (~unsafe) & (result['tail_peak_abs_e_m'] <= tolerance)
    return result


def summarize(metrics):
    return {key: dict(mean=float(np.mean(value)), p95=float(np.quantile(value, .95)),
                     count=int(np.sum(value)) if np.asarray(value).dtype == bool else None)
            for key, value in metrics.items()}


def trajectory_cost(actor, initial, kappas, errors):
    """Per-trajectory differentiable cost, with fixed physical normalizations."""
    import torch
    from .lane_controller_comparison import act, successor
    x = initial
    total = x.new_zeros(len(x))
    previous = None
    for t in range(errors.shape[1]):
        u = act(actor, x, kappas, errors[:, t])
        x = successor(x, u, kappas)
        total = total + (.85*x[:, 0]/.10).square() + .2*(.30*x[:, 1]/.05).square()
        total = total + .01*(.45*u/.10).square()
        if previous is not None:
            total = total + .05*(.45*(u-previous)/.05).square()
        total = total + 25*(x.abs()-1).relu().square().sum(1)
        previous = u
    return total / errors.shape[1] + (.85*x[:, 0]/.10).square()


def adversary(actor, initial, kappas, start, radius, iterations):
    """Row-wise projected gradient ascent; finite steps are NOT exhaustive."""
    import torch
    errors = start.detach()
    for _ in range(iterations):
        errors.requires_grad_(True)
        cost = trajectory_cost(actor, initial, kappas, errors)
        grad, = torch.autograd.grad(cost.sum(), errors)
        errors = (errors + radius/2 * grad.sign()).clamp(-radius, radius).detach()
    return errors


def train(args):
    import torch
    from .tasks import load_task
    from .lane_controller_comparison import act, safety_loss
    from Aebs.lane_keeping.policy import load_actor
    from Aebs.lane_keeping.learned_sbc import load_sbc_checkpoint, save_actor
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    root = args.root / str(args.seed)
    root.mkdir(parents=True, exist_ok=False)
    task = load_task('lane')
    actor_path = Path(task.metadata['actor']['path'])
    barrier_path = Path(task.metadata['barrier']['path'])
    anchor = load_actor(actor_path).double().requires_grad_(False)
    barrier = load_sbc_checkpoint(barrier_path)[0].double().requires_grad_(False)
    actors = {name: copy.deepcopy(anchor).requires_grad_(True) for name in ARMS}
    optimizers = {name: torch.optim.Adam(actor.parameters(), lr=args.lr) for name, actor in actors.items()}
    protocol = dict(format='lane_precision_training_v1', seed=args.seed, arms=list(ARMS),
        source_sha256=digest(__file__), anchor_sha256=digest(actor_path), barrier_sha256=digest(barrier_path),
        observer_sha256=digest(actor_path.parent/'patch_projection.pt'),
        architecture=str(anchor), parameter_count=sum(p.numel() for p in anchor.parameters()),
        algorithm='model-based differentiable fine tuning of PPO-exported actor; NOT PPO',
        common_initialization='previous robust-trained actor; comparison not native SPVC',
        updates=args.updates, lr=args.lr, batch=args.batch, rollout_horizon=args.train_horizon,
        pgd_iterations=args.pgd, training_error_radius=args.radius,
        radius_scope='fixed development perturbation support; not final-policy CP',
        loss='performance + 10*common full-box one-step barrier loss + .05*anchor_action_MSE',
        nominal_precision='nominal rollout performance',
        adversarial_precision='.5*nominal + .5*PGD-perturbed rollout performance',
        matched_compute='Both execute PGD, nominal and perturbed forward/backward cost graphs; only loss weights differ',
        selection='final iterate; no calibration or test model selection',
        safety='same frozen neural B in both losses; finite training batches do not prove safety',
        performance_units='e/.10m, heading/.05rad, steering/.10rad, steering change/.05rad',
        evaluation=dict(horizon=80, dt=.1, deadline_steps=DEADLINE_STEPS, tolerances_m=list(TOLERANCES),
            no_failure_dropping=True, independent_synthetic_seed=202611200801))
    save(root/'protocol.json', protocol)
    shutil.copyfile(__file__, root/'source.py')
    rng = np.random.default_rng(args.seed)
    candidates = rng.uniform(-1, 1, (32768, 2))
    with torch.no_grad():
        active = candidates[barrier(torch.tensor(candidates)).flatten().numpy() < 20]
    if len(active) < 4096:
        raise ValueError('insufficient active states')
    active = active[:4096]
    history, timing = [], {name: 0. for name in ARMS}
    start = time.monotonic()
    for step in range(args.updates):
        x = torch.tensor(rng.uniform(task.initial_lo, task.initial_hi, (args.batch, 2)))
        k = torch.tensor(rng.choice(task.contexts, args.batch))
        safe_x = torch.cat([x[:args.batch//2], torch.tensor(active[rng.integers(len(active), size=args.batch-args.batch//2)])])
        initial_errors = torch.tensor(rng.uniform(-args.radius, args.radius, (args.batch, args.train_horizon, 2)))
        record = dict(step=step+1)
        for name, actor in actors.items():
            begin = time.monotonic()
            worst = adversary(actor, x, k, initial_errors, args.radius, args.pgd)
            nominal = trajectory_cost(actor, x, k, torch.zeros_like(worst)).mean()
            disturbed = trajectory_cost(actor, x, k, worst).mean()
            weight = 0. if name == 'nominal_precision' else .5
            performance = (1-weight)*nominal + weight*disturbed
            safe = safety_loss('robust', actor, barrier, safe_x, k, args.radius, args.train_horizon)
            anchor_penalty = (act(actor, safe_x, k)-act(anchor, safe_x, k)).square().mean()
            loss = performance + 10*safe + .05*anchor_penalty
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss')
            optimizers[name].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
            optimizers[name].step()
            timing[name] += time.monotonic()-begin
            record[name] = dict(loss=float(loss.detach()), nominal=float(nominal.detach()),
                                adversarial=float(disturbed.detach()), safety=float(safe.detach()))
        history.append(record)
        if (step+1) % 50 == 0:
            print(json.dumps(dict(seed=args.seed, update=step+1, elapsed=time.monotonic()-start, losses=record)), flush=True)
    for name, actor in actors.items():
        package = root/name/'package'
        package.mkdir(parents=True)
        save_actor(package/'actor.pt', actor.float())
        for filename in ['sbc.pt', 'patch_projection.pt']:
            shutil.copyfile(actor_path.parent/filename, package/filename)
        save(root/name/'frozen.json', dict(actor_sha256=digest(package/'actor.pt'),
            barrier_sha256=digest(package/'sbc.pt'), observer_sha256=digest(package/'patch_projection.pt'),
            training_seconds=timing[name], cp_recalibrated=False, certificate_transferred=False))
    save(root/'history.json', history)
    print(json.dumps(dict(seed=args.seed, event='training_frozen', seconds=time.monotonic()-start)), flush=True)


def synthetic(args):
    import torch
    from Aebs.lane_keeping import core
    from Aebs.lane_keeping.policy import load_actor
    from Aebs.lane_keeping.robust_trajectory_cp import initial_draws
    torch.set_num_threads(1)
    actor = load_actor(args.package/'actor.pt').requires_grad_(False)
    initial, kappas, ids = initial_draws(202611200801, args.episodes)
    rng = np.random.default_rng(202611200802)
    disturbances = dict(nominal=np.zeros((len(initial), 80, 2)),
        iid=rng.uniform(-args.radius, args.radius, (len(initial), 80, 2)),
        persistent=np.repeat(rng.uniform(-args.radius, args.radius, (len(initial), 1, 2)), 80, axis=1))
    result = {}
    args.output.mkdir(parents=True, exist_ok=False)
    for mode, errors in disturbances.items():
        states = [initial.copy()]
        actions = []
        for t in range(80):
            obs = np.c_[states[-1]/[.85, .30]+errors[:, t], kappas/.01]
            with torch.no_grad():
                u = actor(torch.tensor(obs, dtype=torch.float32)).numpy().flatten()
            states.append(core.dynamics_batch(states[-1], u*.45, kappas))
            actions.append(u)
        states, actions = np.stack(states, axis=1), np.stack(actions, axis=1)
        metrics = precision_metrics(states, actions)
        np.savez_compressed(args.output/(mode+'.npz'), states=states, actions=actions,
                            kappas=kappas, episode_seeds=ids, **metrics)
        result[mode] = summarize(metrics)
    save(args.output/'summary.json', dict(n=args.episodes, support=args.radius, metrics=result,
         actor_sha256=digest(args.package/'actor.pt'), scope='synthetic errors; not image feedback or CP proof'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('train')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--seed', type=int, default=SEEDS[0])
    p.add_argument('--updates', type=int, default=600)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--train-horizon', type=int, default=24)
    p.add_argument('--pgd', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--radius', type=float, default=.10)
    p = sub.add_parser('synthetic')
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=1000)
    p.add_argument('--radius', type=float, default=.10)
    args = parser.parse_args()
    if args.radius <= 0:
        parser.error('positive radius required')
    if args.stage == 'train':
        if min(args.updates, args.batch, args.train_horizon, args.pgd) < 1 or args.lr <= 0:
            parser.error('positive training budgets required')
        train(args)
    else:
        synthetic(args)


if __name__ == '__main__':
    main()
