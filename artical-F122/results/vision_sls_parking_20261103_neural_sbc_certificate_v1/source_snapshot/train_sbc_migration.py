"""Finite-horizon parking SBC candidate with vector observation uncertainty.

Only sampled current states/histories are covered. Successor observation boxes
are enclosed numerically; this is NOT a full continuous-state certificate.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from .core import constraints
from .spvc_backend import ParkingSPVCSystem, ParkingSPVCPolicy, collect_replay, physical_from_augmented
from .perception_error import radius
from .train_migration import load_ppo
from .train_ppo import sha


class Barrier(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(20, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                         nn.Linear(64, 1), nn.ReLU())
        # Avoid initializing the final ReLU wholly inactive.
        nn.init.constant_(self[-2].bias, 1.)


def network_box(net, low, high):
    """Differentiable IBP, without outward rounding; unknown layers fail closed."""
    for layer in net:
        if isinstance(layer, nn.Linear):
            center, half = (low+high)/2, (high-low)/2
            y = center@layer.weight.T+layer.bias
            r = half@layer.weight.abs().T
            low, high = y-r, y+r
        elif isinstance(layer, (nn.Tanh, nn.ReLU)):
            low, high = layer(low), layer(high)
        else:
            raise ValueError('unsupported interval layer')
    return low, high


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ppo-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--error-model')
    p.add_argument('--epochs', type=int, default=1200)
    a = p.parse_args()
    out = Path(a.output_dir); source = Path(a.ppo_dir)/'ppo.zip'
    if out.exists() or a.epochs < 1:
        p.error('new directory and positive epochs required')
    torch.set_num_threads(1); torch.manual_seed(37)
    model = load_ppo(source, device='cpu')
    starts, replay = collect_replay(model)
    # Unsafe and terminal states belong to the stopped process; no strict
    # decrease is imposed there. The unsafe region still has B >= 20.
    active = np.array([not constraints(x)['collision'] and constraints(x)['in_bounds']
                       for x in physical_from_augmented(replay)]) & (replay[:, 19] < 1)
    states = replay[active]
    system = ParkingSPVCSystem(starts, replay)
    policy = ParkingSPVCPolicy(model).eval().requires_grad_(False)
    s, initial = torch.tensor(states), torch.tensor(starts)
    error_model = json.loads(Path(a.error_model).read_text()) if a.error_model else None
    def radii(z):
        return (radius(physical_from_augmented(z), error_model) if error_model else np.zeros((len(z), 3)))
    with torch.no_grad():
        u = policy(torch.zeros(len(s), 4), s)
        lo, hi = system.successor_observation_box(s, u, radii(states))
        init_lo, init_hi = initial.clone(), initial.clone()
        r_init = torch.tensor(radii(starts)/3, dtype=initial.dtype)
        # Same reset observation repeated three times; independent interval
        # hull is a conservative superset of the correlated reset errors.
        for j in (4, 7, 10):
            init_lo[:, j:j+3] -= r_init; init_hi[:, j:j+3] += r_init
    out.mkdir(parents=True)
    cfg = dict(vars(a), ppo_sha256=sha(source), error_model_sha256=sha(a.error_model) if a.error_model else None,
               nonnegative_output='ReLU', augmented_state=20, hidden_layers=[64, 64],
               initial_threshold=1, unsafe_threshold=20, finite_horizon=30,
               dynamic_condition='nonincrease; horizon and unsafe process stopped',
               policy_updated=False, actor_memory_preserved=True,
               current_state_coverage='nominal policy replay points only',
               observation_error_coverage='successor and initial boxes, only at sampled states',
               initial_history_set='independent box superset of repeated single measurement',
               continuous_state_verification=False, outward_rounding=False,
               certificate_probability=None)
    (out/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    barrier = Barrier()
    opt = torch.optim.Adam(barrier.parameters(), lr=.002)
    started = time.monotonic(); history = []
    for epoch in range(a.epochs):
        unsafe = torch.tensor(system.sample_unsafe(1000+epoch, 512))
        _, init_upper = network_box(barrier, init_lo, init_hi)
        _, next_upper = network_box(barrier, lo, hi)
        current = barrier(s)
        unsafe_value = barrier(unsafe)
        init_loss = torch.relu(init_upper-.5).square().mean()
        unsafe_loss = torch.relu(22-unsafe_value).square().mean()
        step_loss = torch.relu(next_upper-current).square().mean()
        loss = 10*init_loss+10*unsafe_loss+step_loss
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(barrier.parameters(), 10); opt.step()
        if (epoch+1) % 200 == 0:
            row = dict(epoch=epoch+1, initial=float(init_loss.detach()), unsafe=float(unsafe_loss.detach()),
                       successor=float(step_loss.detach()), total=float(loss.detach()))
            history.append(row); print(json.dumps(row), flush=True)
    torch.save(barrier.state_dict(), out/'sbc.pt')
    with torch.no_grad():
        _, initial_upper = network_box(barrier, init_lo, init_hi)
        unsafe_points = torch.tensor(system.sample_unsafe(900001, 4096))
        unsafe_value = barrier(unsafe_points)
        _, next_upper = network_box(barrier, lo, hi)
        gap = (next_upper-barrier(s)).flatten()
        nominal_gap = (barrier(system.v_next(s, u))-barrier(s)).flatten()
    metrics = dict(config=cfg, history=history, replay_points=len(states),
                   initial_box_upper=float(initial_upper.max()),
                   initial_violations=int((initial_upper > 1).sum()),
                   unsafe_sample_min=float(unsafe_value.min()),
                   unsafe_sample_count=len(unsafe_points), unsafe_violations=int((unsafe_value < 20).sum()),
                   successor_box_not_proven_count=int((gap > 0).sum()),
                   max_successor_box_gap=float(gap.max()),
                   nominal_point_violation_count=int((nominal_gap > 0).sum()),
                   runtime_seconds=time.monotonic()-started,
                   ppo_unchanged=sha(source) == cfg['ppo_sha256'],
                   status='sampled_current_states_with_numeric_successor_box_bounds_not_a_certificate',
                   safety_certificate=False, certificate_probability=None)
    np.savez_compressed(out/'diagnostic_points.npz', initial=starts, replay=states,
                        unsafe=unsafe_points.numpy(), successor_upper_gap=gap.numpy(),
                        nominal_gap=nominal_gap.numpy())
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps({k:v for k,v in metrics.items() if k not in ('history', 'config')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
