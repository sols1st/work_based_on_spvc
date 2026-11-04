"""Offline trajectory demonstrations to warm-start the existing parking actor.

The optimizer uses simulator states offline only. The exported actor still
accepts exactly the existing 16 observation/history inputs and no state oracle.
Demonstrations are rechecked in the original environment before imitation.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
import torch
from .core import DT, HORIZON, INITIAL_STATES, ACTION_HIGH, STATE_LOW, STATE_HIGH, GOAL_LOW, GOAL_HIGH, OBSTACLES
from .ppo_env import ParkingPPOEnv
from .train_migration import load_ppo, quality
from .train_ppo import evaluate, sha


def rollout_tensor(initial, normalized_actions):
    x = torch.as_tensor(initial, dtype=normalized_actions.dtype)
    states = [x]
    for u in normalized_actions:
        a = u*torch.as_tensor(ACTION_HIGH, dtype=u.dtype)
        def ode(q):
            return torch.stack([q[3]*torch.cos(q[2]), q[3]*torch.sin(q[2]), a[0], a[1]])
        k1 = ode(x); k2 = ode(x+DT*.5*k1); k3 = ode(x+DT*.5*k2); k4 = ode(x+DT*k3)
        x = x+DT*(k1+2*k2+2*k3+k4)/6
        states.append(x)
    return torch.stack(states)


def demonstrations(model, iterations=250):
    _, _, initial_actions = evaluate(model, quiet=True)
    all_obs, all_actions, logs, paths = [], [], [], []
    env = ParkingPPOEnv()
    for scenario in range(15):
        goal = torch.tensor((GOAL_LOW[scenario, :2]+GOAL_HIGH[scenario, :2])/2)
        low, high = torch.tensor(STATE_LOW), torch.tensor(STATE_HIGH)
        circles = torch.tensor(OBSTACLES)
        def objective(flat):
            u = torch.tensor(flat.reshape(HORIZON, 2), dtype=torch.float64, requires_grad=True)
            x = rollout_tensor(INITIAL_STATES[scenario], u)
            excess = torch.relu(low+.015-x)+torch.relu(x-high+.015)
            clearance = torch.linalg.vector_norm(x[:, None, :2]-circles[None, :, :2], dim=2)-circles[:, 2]
            loss = (5000*(x[-1, :2]-goal).square().sum()+10*x[-1, 3].square()
                    +50000*excess.square().sum()+50000*torch.relu(.04-clearance).square().sum()
                    +.2*(x[:, :2]-goal).square().sum()+.02*u.square().sum())
            loss.backward()
            return float(loss.detach()), u.grad.numpy().ravel()
        fit = minimize(objective, (initial_actions[scenario]/ACTION_HIGH).ravel(), jac=True,
                       method='L-BFGS-B', bounds=[(-1, 1)]*(HORIZON*2),
                       options={'maxiter': iterations, 'ftol': 1e-10, 'maxls': 30})
        controls = fit.x.reshape(HORIZON, 2)
        obs, _ = env.reset(options={'scenario': scenario})
        observations, states = [], [env.task.state.copy()]
        for action in controls:
            observations.append(obs.copy())
            obs, _, _, _, info = env.step(action)
            states.append(env.task.state.copy())
        row = dict(scenario=scenario, optimizer_success=bool(fit.success), iterations=int(fit.nit),
                   feasible_original_task=bool(info['episode_success']),
                   collision=info['collision_seen'], out_of_bounds=info['out_of_bounds_seen'],
                   final_distance=info['final_goal_distance'])
        print('demonstration', json.dumps(row), flush=True)
        logs.append(row); paths.append(states)
        if info['episode_success']:
            all_obs.extend(observations); all_actions.extend(controls)
    env.close()
    return np.asarray(all_obs, dtype=np.float32), np.asarray(all_actions, dtype=np.float32), logs, np.asarray(paths)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--load', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--planner-iterations', type=int, default=250)
    p.add_argument('--projection', default='results/vision_sls_parking_20261025_adaptation_v1/projection_adapted.pt')
    a = p.parse_args()
    out = Path(a.output_dir)
    if out.exists() or min(a.epochs, a.planner_iterations) < 1:
        p.error('new directory and positive budgets required')
    torch.set_num_threads(1); torch.manual_seed(17)
    started = time.monotonic()
    out.mkdir(parents=True)
    config = dict(vars(a), training='offline model-based demonstrations, actor-only behavior cloning',
                  deployed_input='unchanged 16D image/history interface',
                  source_sha256=sha(a.load), projection_sha256=sha(a.projection),
                  development_tasks=15, independent_test=False, safety_certificate=False)
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    model = load_ppo(a.load, device='cpu')
    obs, actions, logs, demo_paths = demonstrations(model, a.planner_iterations)
    np.savez_compressed(out/'demonstrations.npz', observations=obs, actions=actions, states=demo_paths)
    (out/'demonstration_metrics.json').write_text(json.dumps(logs, indent=2)+'\n')
    if not len(obs):
        raise RuntimeError('no feasible demonstrations; actor not trained')
    x, target = torch.tensor(obs), torch.tensor(actions)
    params = list(model.policy.mlp_extractor.policy_net.parameters())+list(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(params, lr=3e-4)
    baseline, _, _ = evaluate(model, quiet=True)
    best = quality(baseline); best_epoch = 0
    model.save(out/'ppo.zip')
    history = []
    for epoch in range(1, a.epochs+1):
        prediction = model.policy.action_net(model.policy.mlp_extractor.policy_net(x))
        loss = (prediction-target).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if epoch % 100 == 0 or epoch == a.epochs:
            m, _, _ = evaluate(model, quiet=True)
            if quality(m) > best:
                best = quality(m); best_epoch = epoch; model.save(out/'ppo.zip')
            row = dict(epoch=epoch, mse=float(loss.detach()), success=m['success_count'],
                       collision=m['collision_count'], bounds=m['out_of_bounds_count'])
            history.append(row); print('imitation', json.dumps(row), flush=True)
    model.save(out/'last_ppo.zip')
    selected = load_ppo(out/'ppo.zip', device='cpu')
    m, states, controls = evaluate(selected, quiet=True)
    np.savez_compressed(out/'rollouts.npz', proxy_states=states, proxy_actions=controls)
    result = dict(config=config, baseline=baseline, proxy=m, best_epoch=best_epoch,
                  checkpoint_sha256=sha(out/'ppo.zip'), feasible_demonstrations=sum(r['feasible_original_task'] for r in logs),
                  history=history, runtime_seconds=time.monotonic()-started,
                  image_evaluated=False, safety_certificate=False)
    (out/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Selected', {k:v for k,v in m.items() if k != 'per_scenario'}, flush=True)


if __name__ == '__main__':
    main()
