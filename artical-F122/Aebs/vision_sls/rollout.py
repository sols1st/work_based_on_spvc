"""Batched offline AEBS rollouts with the existing terminal-event contract.

No reward or success criteria are changed. Nearest-image replay is explicitly
separate from nominal observations and does not emulate live CARLA rendering.
"""
from collections import Counter
import numpy as np
import torch
from Aebs.system.outcomes import classify_terminal_outcome, TIMEOUT


def initial_grid(size=5):
    if size < 1:
        raise ValueError('grid size must be positive')
    return np.array([(d,v) for d in np.linspace(15.,16.,size)
                     for v in np.linspace(2.5,3.,size)], dtype=np.float64)


def random_initial(count, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform([15.,2.5],[16.,3.],size=(count,2))


@torch.no_grad()
def rollout(policy, scale, initial=None, distances=None, observations=None,
            max_steps=400, record_trajectories=False, monitor=None):
    """States use physical metres and m/s; controller sees observation + speed."""
    initial = initial_grid() if initial is None else np.asarray(initial,dtype=np.float64)
    if initial.ndim != 2 or initial.shape[1] != 2 or not len(initial) or not np.isfinite(initial).all():
        raise ValueError('finite nonempty [N,2] initial states required')
    if not np.isfinite(scale) or scale <= 0 or max_steps < 1:
        raise ValueError('positive scale and horizon required')
    if (distances is None) != (observations is None):
        raise ValueError('paired image distances and observations required')
    use_images = distances is not None
    if use_images:
        distances = np.asarray(distances).reshape(-1)
        observations = np.asarray(observations).reshape(-1)
        if not len(distances) or distances.shape != observations.shape or not np.isfinite(distances).all() or not np.isfinite(observations).all():
            raise ValueError('finite paired image distances and observations required')
    device = policy.C.device
    # Match AebsEnv's float32 state storage, including rounding every step.
    state = np.column_stack([initial[:,0]/scale,initial[:,1]]).astype(np.float32)
    n = len(state)
    outcomes = np.full(n,'',dtype=object)
    steps = np.zeros(n,dtype=int)
    returns = np.zeros(n,dtype=np.float64)
    max_lookup = 0.
    trace = []
    was_training = policy.training
    policy.eval()
    for _ in range(max_steps):
        active = np.flatnonzero(outcomes == '')
        if not len(active):
            break
        current = state[active]
        distance = current[:,0].astype(np.float64)*scale
        observed = current[:,0].copy()
        if use_images:
            nearest = np.abs(distance[:,None]-distances[None,:]).argmin(axis=1)
            max_lookup = max(max_lookup,float(np.abs(distance-distances[nearest]).max()))
            observed = observations[nearest]
        action = policy.forward_observation(torch.tensor(observed[:,None],device=device,dtype=torch.float32),
                                             torch.tensor(current[:,1:2],device=device)).cpu().numpy().reshape(-1)
        if not np.isfinite(action).all():
            raise ValueError('nonfinite controller action')
        action = np.clip(action,-3.,3.)
        # Use float64 intermediates as scalar AebsEnv.step does before storing
        # the next state as float32. Positive acceleration is braking.
        d = current[:,0].astype(np.float64)*scale
        v = current[:,1].astype(np.float64)
        d_next = d-v*.05
        v_next = np.clip(v-action.astype(np.float64)*.05,0.,3.)
        steps[active] += 1
        rewards = 2*(d-d_next)-.001
        for j,i in enumerate(active):
            outcome = classify_terminal_outcome(d_next[j],v_next[j])
            if outcome == 'success':
                rewards[j] += 2.
            elif outcome == 'unsafe':
                rewards[j] -= (v_next[j]-.5)*3.
            if outcome is not None:
                outcomes[i] = outcome
        returns[active] += rewards
        state[active,0] = d_next/scale
        state[active,1] = v_next
        if monitor is not None:
            monitor(active,np.column_stack([d,v]),
                    np.column_stack([state[active,0].astype(np.float64)*scale,state[active,1]]),
                    action,observed,outcomes[active])
        if record_trajectories:
            trace.append(dict(indices=active.tolist(),distance_m=d.tolist(),speed_mps=v.tolist(),
                              observation=np.asarray(observed).tolist(),braking_acceleration=action.tolist()))
    outcomes[outcomes == ''] = TIMEOUT
    policy.train(was_training)
    result = dict(episodes=n,counts=dict(Counter(outcomes)),lookup_max_error_m=max_lookup,
                  mode='saved_image_nearest_replay' if use_images else 'nominal_Cx',
                  initial_states=initial.tolist(),outcomes=outcomes.tolist(),steps=steps.tolist(),
                  returns=returns.tolist(),terminal_states=np.column_stack([state[:,0]*scale,state[:,1]]).tolist())
    if record_trajectories:
        result['trajectories'] = trace
    return result
