"""AEBS's actual NonnegativeSBC model, with lane observation/dynamics adapters."""
import hashlib
import numpy as np
import torch
from torch import nn

from Aebs.vision_sls.sbc_cp import NonnegativeSBC, load_sbc_checkpoint, barrier_bounds
from Aebs.vision_sls.neural_bounds import tanh_affine_bounds
from . import core
from .verify import linear_bounds, actor_bounds, dynamics_bounds


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_barrier(path, barrier, metadata):
    torch.save(dict(format='vision_sls_nonnegative_sbc_v2', features=barrier.features,
                    state_dict=barrier.state_dict(), metadata={**metadata,
                    'nonnegative_activation':'relu'}), path)


def save_actor(path, actor):
    torch.save(dict(format='lane_keeping_actor_v1', policy_net=actor.policy_net.state_dict(),
                    action_net=actor.action_net.state_dict(), observation_dim=3, action_dim=1), path)


def raw(barrier, states):
    for layer in barrier.layers[:-1]:
        states = layer(states).tanh()
    return barrier.layers[-1](states).flatten()


def raw_bounds(barrier, lower, upper, method='affine'):
    if method == 'ibp':
        for i, layer in enumerate(barrier.layers):
            lower, upper = linear_bounds(lower, upper, layer)
            if i < len(barrier.layers)-1:
                lower, upper = lower.tanh(), upper.tanh()
        return lower.flatten()-1e-12, upper.flatten()+1e-12
    layers = []
    for layer in barrier.layers[:-1]:
        layers.extend([layer, nn.Tanh()])
    layers.append(barrier.layers[-1])
    lo, hi = tanh_affine_bounds(layers, lower, upper)
    return lo.flatten(), hi.flatten()


def bounds(barrier, lower, upper, method='affine'):
    lo, hi = raw_bounds(barrier, lower, upper, method)
    return lo.relu(), hi.relu()


def error_boxes(radius, splits, dtype=torch.float64):
    cuts = torch.linspace(-radius, radius, splits+1, dtype=dtype)
    return [(torch.stack([cuts[i], cuts[j]]), torch.stack([cuts[i+1], cuts[j+1]]))
            for i in range(splits) for j in range(splits)]


def transitions(actor, lower, upper, kappas, el, eh, method='affine'):
    """Kappas in physical units; observation errors and states normalized."""
    kappas = torch.as_tensor(kappas, dtype=lower.dtype, device=lower.device).expand(len(lower))
    curve = (kappas/core.KAPPA_SCALE)[:, None]
    ol, oh = torch.cat([lower+el, curve], 1), torch.cat([upper+eh, curve], 1)
    if method == 'affine':
        lo, hi = tanh_affine_bounds(list(actor.policy_net)+[actor.action_net], ol, oh)
        lo, hi = lo.clamp(-1, 1), hi.clamp(-1, 1)
    else:
        lo, hi = actor_bounds(actor, ol, oh)
    nl, nh = dynamics_bounds(lower, upper, lo, hi, kappas)
    return nl-1e-12, nh+1e-12


def grid_boxes(low, high, grid, dtype=torch.float64):
    a = torch.linspace(low[0], high[0], grid+1, dtype=dtype)
    b = torch.linspace(low[1], high[1], grid+1, dtype=dtype)
    e, t = torch.meshgrid(a, b, indexing='ij')
    return (torch.stack([e[:-1, :-1].flatten(), t[:-1, :-1].flatten()], 1),
            torch.stack([e[1:, 1:].flatten(), t[1:, 1:].flatten()], 1))


def unsafe_boxes(grid=16, dtype=torch.float64):
    # Every one-step successor of the safe rectangle lies in [-1.3,1.3]^2.
    pairs = [grid_boxes(lo, hi, grid, dtype) for lo, hi in [
        ([-1.3,-1.3],[-1,1.3]), ([1,-1.3],[1.3,1.3]),
        ([-1,-1.3],[1,-1]), ([-1,1],[1,1.3])]]
    return torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs])


@torch.no_grad()
def task_panel(actor, radius=.08, seed=710, episodes=192):
    """Fixed development panel; it is not an independent final test."""
    rng = np.random.default_rng(seed)
    initial = core.sample_initial(rng, episodes)
    initial[:4] = [[e,t] for e in [-core.INIT_E, core.INIT_E]
                          for t in [-core.INIT_THETA, core.INIT_THETA]]
    result = {}
    for mode in ['nominal','uniform','alternating']:
        state = initial.copy()
        all_states, all_actions = [state], []
        for step in range(core.HORIZON):
            k = rng.choice(core.KAPPAS, episodes)
            error = np.zeros((episodes,2)) if mode == 'nominal' else rng.uniform(-radius,radius,(episodes,2))
            if mode == 'alternating':
                error = radius*np.column_stack([(-1.)**(np.arange(episodes)+step),
                                               (-1.)**(np.arange(episodes)+step//2)])
            obs = np.c_[state/[core.E_SCALE,core.THETA_SCALE]+error,k/core.KAPPA_SCALE]
            param = next(actor.parameters())
            action = actor(torch.tensor(obs,dtype=param.dtype)).numpy().flatten()
            state = core.dynamics_batch(state,action*core.PHI_MAX,k)
            all_states.append(state); all_actions.append(action)
        states, actions = np.array(all_states), np.array(all_actions)
        result[mode] = dict(unsafe_episodes=int((np.abs(states/[core.E_SCALE,core.THETA_SCALE])>1).any(axis=(0,2)).sum()),
                            episodes=episodes, mean_abs_e=float(abs(states[1:,:,0]).mean()),
                            mean_action_change=float(abs(np.diff(actions,axis=0)).mean()))
    return result


def retention_failures(baseline, proposed):
    reasons = []
    for mode, base in baseline.items():
        value = proposed[mode]
        if value['unsafe_episodes'] > base['unsafe_episodes']:
            reasons.append(mode+': additional unsafe episodes')
        if value['mean_abs_e'] > base['mean_abs_e']*1.15+.003:
            reasons.append(mode+': tracking degradation')
        if value['mean_action_change'] > base['mean_action_change']*1.25+.02:
            reasons.append(mode+': action oscillation')
    return reasons
