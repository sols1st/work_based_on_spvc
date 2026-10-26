"""Parking adapters to the existing SPVC learner, not new barrier losses.

3D image/Cx observation is unchanged. A 20D augmented verifier state retains
the current physical state AND exactly the 16 inputs of the history PPO.
This adapter alone does not certify image errors or continuous-state coverage.
"""
import copy
import numpy as np
import torch
from torch import nn
from gymnasium.spaces import Box
from .core import (C, RENDER_OFFSET, STATE_LOW, STATE_HIGH, ACTION_HIGH,
                   ACTION_LOW, OBSTACLES, DT, HORIZON, constraints)

STATE_CENTER = (STATE_LOW+STATE_HIGH)/2
STATE_SCALE = (STATE_HIGH-STATE_LOW)/2


def augment(physical_state, policy_observation):
    x, o = np.asarray(physical_state), np.asarray(policy_observation)
    if x.shape != (4,) or o.shape != (16,):
        raise ValueError('4D physical state and exact 16D PPO input required')
    return np.concatenate([(x-STATE_CENTER)/STATE_SCALE, o]).astype(np.float32)


def physical_from_augmented(s):
    return np.asarray(s)[..., :4]*STATE_SCALE+STATE_CENTER


class PhysicalActor(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy_net = copy.deepcopy(policy.mlp_extractor.policy_net)
        self.action_net = copy.deepcopy(policy.action_net)
        self.register_buffer('action_scale', torch.tensor(ACTION_HIGH, dtype=torch.float32))

    def forward(self, observation):
        mu = self.action_net(self.policy_net(observation))
        return torch.clamp(mu, -1., 1.)*self.action_scale


class ParkingSPVCPolicy(nn.Module):
    def __init__(self, ppo):
        super().__init__()
        self.controller_net = PhysicalActor(ppo.policy)

    def controller_input(self, unused_z, augmented_state):
        return augmented_state[:, 4:20]

    def forward(self, z, augmented_state):
        return self.controller_net(self.controller_input(z, augmented_state))

    def export_actor(self, ppo):
        # The original SPVC phase changes the actor, not PPO critic/log_std.
        ppo.policy.mlp_extractor.policy_net.load_state_dict(self.controller_net.policy_net.state_dict())
        ppo.policy.action_net.load_state_dict(self.controller_net.action_net.state_dict())
        return ppo


class ParkingSPVCSystem:
    def __init__(self, initial_states, replay_states):
        self.initial_states = np.asarray(initial_states, dtype=np.float32)
        self.replay_states = np.asarray(replay_states, dtype=np.float32)
        self.observation_space = Box(
            low=np.array([-2.]*4+[-4.]*9+[-1.]*4+[-2.]*2+[0.], np.float32),
            high=np.array([2.]*4+[4.]*9+[1.]*4+[2.]*2+[1.], np.float32))
        self.action_space = Box(ACTION_LOW.astype(np.float32), ACTION_HIGH.astype(np.float32))
        # This first connection experiment is the SAME zero-process-noise task.
        # No independent per-step image-error law is invented here.
        self.noise = np.zeros(20, np.float32)
        self.noise_bounds = (self.noise.copy(), self.noise.copy())
        self.space_split = np.ones(20, dtype=int)  # NOT a verification grid in this pilot.

    def v_next(self, s, a):
        def const(v):
            return torch.as_tensor(v, dtype=s.dtype, device=s.device)
        x = s[:, :4]*const(STATE_SCALE)+const(STATE_CENTER)
        a = torch.maximum(torch.minimum(a, const(ACTION_HIGH)), const(ACTION_LOW))
        def ode(q):
            return torch.stack([q[:, 3]*torch.cos(q[:, 2]),
                                q[:, 3]*torch.sin(q[:, 2]), a[:, 0], a[:, 1]], 1)
        k1 = ode(x)
        k2 = ode(x+DT*.5*k1)
        k3 = ode(x+DT*.5*k2)
        k4 = ode(x+DT*k3)
        nxt = x+DT*(k1+2*k2+2*k3+k4)/6
        # At decision t+1 the newest delayed measurement is C*x_t, NOT C*x_(t+1).
        delayed_y = (x+const(RENDER_OFFSET)) @ const(C).T / 3.
        return torch.cat([(nxt-const(STATE_CENTER))/const(STATE_SCALE),
                          s[:, 7:13], delayed_y, s[:, 15:17], a/const(ACTION_HIGH),
                          s[:, 17:19], torch.clamp(s[:, 19:20]+1/HORIZON, max=1.)], 1)

    def sample_initial(self, seed, n):
        rng = np.random.default_rng(seed)
        return self.initial_states[rng.integers(len(self.initial_states), size=n)].copy()

    def sample_unsafe(self, seed, n):
        """Parking geometry sampling only; original region loss is untouched.

        Histories are a sampled superset, not claimed reachable unsafe histories.
        Half the samples lie in original collision circles, half beyond bounds.
        """
        rng = np.random.default_rng(seed)
        result = self.replay_states[rng.integers(len(self.replay_states), size=n)].copy()
        x = physical_from_augmented(result)
        for i in range(n):
            if i % 2 == 0:
                circle = OBSTACLES[int(rng.integers(3))]
                angle, radius = rng.uniform(-np.pi, np.pi), circle[2]*np.sqrt(rng.uniform(0, .95))
                x[i, :2] = circle[:2]+radius*np.array([np.cos(angle), np.sin(angle)])
            else:
                axis = int(rng.integers(4))
                x[i, axis] = (STATE_LOW[axis]-.01 if rng.random() < .5
                              else STATE_HIGH[axis]+.01)
        result[:, :4] = (x-STATE_CENTER)/STATE_SCALE
        return result


def make_original_learner(system, policy):
    from Aebs.VT.train import VTLearner

    class ParkingLearner(VTLearner):
        # Adapt only geometry sampling; all losses, optimizers, update routines
        # and network class are inherited without editing original backend files.
        def sample_init(self, rng_seed, n):
            return torch.as_tensor(self.env.sample_initial(rng_seed, n), device=self.device)

        def sample_unsafe(self, rng_seed, n):
            return torch.as_tensor(self.env.sample_unsafe(rng_seed, n), device=self.device)

    return ParkingLearner(l_model_config=[20, 16, 8, 1], env=system,
                          p_lip=2., l_lip=4., eps=.1, gamma_decrease=1.,
                          reach_prob=.95, square_l_output=True, p_net=policy)


def collect_replay(ppo):
    from .ppo_env import ParkingPPOEnv
    rows, starts = [], []
    env = ParkingPPOEnv()
    try:
        for scenario in range(15):
            obs, _ = env.reset(options={'scenario': scenario})
            starts.append(augment(env.task.state, obs))
            for _ in range(HORIZON):
                rows.append(augment(env.task.state, obs))
                u, _ = ppo.predict(obs, deterministic=True)
                obs, _, _, _, _ = env.step(u)
    finally:
        env.close()
    return np.asarray(starts), np.asarray(rows)


def sampled_barrier_diagnostic(learner, system, states):
    """Reuses ORIGINAL VTVerifier expected-value operation at point inputs.

    No fake 20D grid, no continuous-state Lipschitz/cell claim, no probability
    bound. Zero noise is a Dirac mass, represented by exactly one zero cell.
    """
    from Aebs.VT.verify import VTVerifier
    bounded = learner.create_bounded_module(copy.deepcopy(learner.l_model))
    verifier = VTVerifier(learner, system, bounded, batch_size=256,
                          reach_prob=.9, fail_check_fast=False)
    y = torch.as_tensor(states, dtype=torch.float32, device=learner.device)
    zero = torch.zeros((1, 20), device=learner.device)
    mass = torch.ones(1, device=learner.device)
    margins = []
    with torch.no_grad():
        for batch in y.split(128):
            u = learner.p_net(torch.zeros((len(batch), 4), device=learner.device), batch)
            expected = verifier.compute_expected_l(batch, u, mass, zero, zero)
            margins.extend((learner.l_model(batch).flatten()-expected).cpu().tolist())
        init = learner.l_model(torch.as_tensor(system.initial_states, device=learner.device)).flatten()
        unsafe_points = system.sample_unsafe(101, 1024)
        unsafe = learner.l_model(torch.as_tensor(unsafe_points, device=learner.device)).flatten()
    m = np.asarray(margins)
    worst = int(np.argmin(m))
    return dict(status='sampled_diagnostic_not_a_certificate', checked_states=len(m),
                nonincrease_violations=int((m <= 0).sum()),
                strict_epsilon_violations=int((m < learner.eps).sum()),
                epsilon=learner.eps, min_margin=float(m.min()), mean_margin=float(m.mean()),
                initial_max=float(init.max()), initial_violations=int((init > 1).sum()),
                initial_count=len(init), initial_target=1.,
                unsafe_min=float(unsafe.min()), unsafe_violations=int((unsafe < 20).sum()),
                unsafe_count=len(unsafe), unsafe_target=20.,
                worst_physical_state=physical_from_augmented(states[worst]).tolist(),
                continuous_state_verification=False, image_error_covered=False,
                certificate_probability=None)
