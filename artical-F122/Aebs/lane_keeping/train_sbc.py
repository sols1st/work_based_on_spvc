"""Train the nonnegative lane-keeping SBC against the fixed actor.

The safe region is not assumed to be a box.  The SBC is trained to satisfy
initial <= 1, unsafe boundary >= 20, and ``B(next) <= B(x) - margin`` on every
state with ``B(x) < 20``, for every observation error in the calibrated box.
On-policy states from the initial set are mixed into training so the learned
low-B region follows the actually reachable states rather than an arbitrary
box.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from . import core, sbc
from .policy import load_actor
from .sbc import LaneSBC


def rollout_states(actor, radius, episodes=200, seed=0):
    rng = np.random.default_rng(seed)
    states = []
    for episode in range(episodes):
        state = core.sample_initial(rng, 1)[0]
        kappa = float(rng.choice(core.KAPPAS))
        for step in range(core.HORIZON):
            observation = core.to_net(state, kappa)
            observation[:2] += rng.uniform(-radius, radius, 2)
            with torch.no_grad():
                action = actor(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)).item()
            state = core.dynamics(state, core.action_from_net(action), kappa)
            states.append(core.state_to_net(state))
            if core.unsafe(state):
                break
    return torch.as_tensor(np.asarray(states), dtype=torch.float32)


def violation_stats(actor, barrier, radius, device='cpu', grid=61, errors=4):
    """Coarse grid diagnostic used during training; the real audit is in verify.py."""
    axis = torch.linspace(-1., 1., grid)
    ee, tt = torch.meshgrid(axis, axis, indexing='ij')
    states = torch.stack([ee.reshape(-1), tt.reshape(-1)], dim=1).to(device)
    errors_per_side = errors
    delta = torch.linspace(-radius, radius, errors_per_side + 1)
    de, dt = torch.meshgrid(delta, delta, indexing='ij')
    offsets = torch.stack([de.reshape(-1), dt.reshape(-1)], dim=1).to(device)
    kappas = torch.tensor([k / core.KAPPA_SCALE for k in core.KAPPAS], device=device)
    worst = -float('inf')
    violations = 0
    with torch.no_grad():
        current = barrier(states).reshape(-1)
        for kappa in kappas:
            observed = states[:, None, :] + offsets[None, :, :]
            observed = torch.cat([observed, kappa.expand(len(states), len(offsets), 1)], dim=2)
            actions = actor(observed.reshape(-1, 3)).reshape(len(states), len(offsets))
            next_states = sbc.next_states_for_errors(
                actor, states, kappa.expand(len(states)), offsets[None, :, :].expand(len(states), -1, -1))
            next_b = barrier(next_states.reshape(-1, 2)).reshape(len(states), len(offsets)).max(1).values
            residual = next_b - current + sbc.DECREASE_MARGIN
            active = current < sbc.UNSAFE_LEVEL
            worst = max(worst, float(residual[active].max())) if active.any() else worst
            violations += int((active & (residual > 0)).sum())
    return dict(worst_residual=worst, grid_violations=violations,
                checked=len(states) * len(kappas))


def collect_violations(actor, barrier, radius, device='cpu', grid=61, errors=4):
    axis = torch.linspace(-1., 1., grid, device=device)
    ee, tt = torch.meshgrid(axis, axis, indexing='ij')
    states = torch.stack([ee.reshape(-1), tt.reshape(-1)], dim=1)
    delta = torch.linspace(-radius, radius, errors + 1, device=device)
    de, dt = torch.meshgrid(delta, delta, indexing='ij')
    offsets = torch.stack([de.reshape(-1), dt.reshape(-1)], dim=1)
    found = []
    with torch.no_grad():
        current = barrier(states).flatten()
        for kappa_physical in core.KAPPAS:
            kappa = torch.full((len(states),), kappa_physical / core.KAPPA_SCALE, device=device)
            next_states = sbc.next_states_for_errors(actor, states, kappa, offsets[None, :, :].expand(len(states), -1, -1))
            next_b = barrier(next_states.reshape(-1, 2)).reshape(len(states), len(offsets)).max(1).values
            bad = (current < sbc.UNSAFE_LEVEL) & (next_b - current + sbc.DECREASE_MARGIN > 0)
            found.append(states[bad])
    found = torch.cat(found) if found else torch.empty((0, 2), device=device)
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actor', type=Path, required=True)
    parser.add_argument('--cp', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--rollout-episodes', type=int, default=200)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f'choose a new output directory: {args.output_dir}')
    args.output_dir.mkdir(parents=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cp = json.loads(args.cp.read_text())
    radius = float(cp['error_model']['radius'])
    actor = load_actor(args.actor, args.device)
    barrier = LaneSBC().to(args.device)
    optimizer = torch.optim.Adam(barrier.parameters(), lr=args.lr)
    rollout = rollout_states(actor, radius, args.rollout_episodes, args.seed).to(args.device)
    print(f'reachable buffer: {rollout.shape}', flush=True)

    def tensor(points):
        return torch.as_tensor(points, dtype=torch.float32, device=args.device)

    violation_buffer = None
    history = []
    for step in range(args.steps):
        domain = tensor(sbc.sample_domain(rng, args.batch))
        initial = tensor(sbc.sample_initial(rng, args.batch // 2))
        unsafe = tensor(sbc.sample_unsafe(rng, args.batch // 2))
        guided = rollout[torch.randint(0, len(rollout), (args.batch // 2,), device=args.device)]
        decrease_states = torch.cat([domain, guided], dim=0)
        if violation_buffer is not None and len(violation_buffer):
            extra = violation_buffer[torch.randint(0, len(violation_buffer),
                                                    (args.batch // 2,), device=args.device)]
            decrease_states = torch.cat([decrease_states, extra], dim=0)
        kappas = torch.as_tensor(rng.choice(np.asarray(core.KAPPAS) / core.KAPPA_SCALE,
                                            size=len(decrease_states)), dtype=torch.float32,
                                 device=args.device)
        errors = sbc.sample_errors(rng, len(decrease_states), radius).astype(np.float32)
        errors = tensor(errors)
        current = barrier(decrease_states).flatten()
        next_states = sbc.next_states_for_errors(actor, decrease_states, kappas, errors)
        next_b = barrier(next_states.reshape(-1, 2)).reshape(len(decrease_states), -1).max(1).values
        active = current.detach() < sbc.UNSAFE_LEVEL
        dec_terms = torch.nn.functional.relu(next_b - current + sbc.DECREASE_MARGIN)[active]
        loss_dec = dec_terms.mean() if dec_terms.numel() else torch.zeros((), device=args.device)
        loss_init = torch.nn.functional.relu(barrier(initial) - sbc.INIT_TARGET).mean()
        loss_unsafe = torch.nn.functional.relu(sbc.UNSAFE_TARGET - barrier(unsafe)).mean()
        # Lipschitz-style smoothness keeps IBP tighter than pointwise fitting.
        probe = domain.clone().requires_grad_(True)
        gradient = torch.autograd.grad(barrier(probe).sum(), probe, create_graph=True)[0]
        loss_lip = gradient.square().sum(1).mean()
        loss = 5. * loss_init + 5. * loss_unsafe + 5. * loss_dec + 1e-4 * loss_lip
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(barrier.parameters(), 5.)
        optimizer.step()
        if step % 250 == 0 or step == args.steps - 1:
            fresh = collect_violations(actor, barrier, radius, args.device)
            if len(fresh):
                violation_buffer = (fresh if violation_buffer is None else
                                    torch.cat([violation_buffer, fresh])[-4000:])
            stats = violation_stats(actor, barrier, radius)
            record = dict(step=step, loss=float(loss), loss_dec=float(loss_dec),
                          loss_init=float(loss_init), loss_unsafe=float(loss_unsafe), loss_lip=float(loss_lip),
                          buffer=int(0 if violation_buffer is None else len(violation_buffer)),
                          **stats)
            history.append(record)
            print(json.dumps(record), flush=True)
    metadata = dict(actor=str(args.actor), cp=str(args.cp), radius=radius,
                    steps=args.steps, seed=args.seed, features=list(barrier.features),
                    decrease_margin=sbc.DECREASE_MARGIN, unsafe_level=sbc.UNSAFE_LEVEL,
                    unsafe_target=sbc.UNSAFE_TARGET, initial_target=sbc.INIT_TARGET,
                    initial_level=sbc.INIT_LEVEL, history=history,
                    history_note='coarse diagnostic grid, not the continuous audit')
    sbc.save_sbc(args.output_dir / 'sbc.pt', barrier.cpu(), metadata)
    (args.output_dir / 'training.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(dict(final=history[-1]), indent=2))


if __name__ == '__main__':
    main()
