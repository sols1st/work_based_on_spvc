"""Offline parking references, time-varying LQR, and delayed visual observer.

This is an experimental nominal-planning/LQG baseline, NOT the robust SLS
solver and not a certified controller. Runtime reference selection uses the
first visual measurement and the commanded goal, never the scenario ID/state.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from .core import (C, RENDER_OFFSET, DT, HORIZON, INITIAL_STATES, ACTION_HIGH, ACTION_LOW,
                   STATE_LOW, STATE_HIGH, GOAL_LOW, GOAL_HIGH, OBSTACLES,
                   dynamics, nominal_observation, constraints)

RESIDUAL_SCALE = np.array([.3, .5])
ERROR_SCALE = np.array([1., 1., 1., 1.])


def linearize(x, u, epsilon=1e-5):
    a = np.column_stack([(dynamics(x+np.eye(4)[j]*epsilon, u)-
                          dynamics(x-np.eye(4)[j]*epsilon, u))/(2*epsilon) for j in range(4)])
    b = np.column_stack([(dynamics(x, u+np.eye(2)[j]*epsilon)-
                          dynamics(x, u-np.eye(2)[j]*epsilon))/(2*epsilon) for j in range(2)])
    return a, b


def feedback_gains(states, actions):
    ab = [linearize(x, u) for x, u in zip(states[:-1], actions)]
    q = np.diag([12., 12., 2., 1.]); r = np.diag([.4, .15])
    p = np.diag([120., 120., 10., 6.]); gains = []
    for a, b in ab[::-1]:
        k = np.linalg.solve(r+b.T@p@b, b.T@p@a)
        p = q+a.T@p@(a-b@k)
        p = (p+p.T)/2
        gains.append(k)
    # Design covariances, not a claimed stochastic law for real perception.
    measurement = np.diag(np.square([.1, .15, .15]))
    process = np.diag(np.square([.005, .005, .01, .04]))
    p = np.diag(np.square([.08, .08, .08, .03]))
    l = np.zeros((HORIZON+1, 4, 3))
    l[0] = np.linalg.solve(C@p@C.T+measurement, C@p).T
    p = (np.eye(4)-l[0]@C)@p
    for t in range(1, HORIZON+1):
        # t=1 repeats the reset frame: do not count it as new evidence.
        if t > 1:
            l[t] = np.linalg.solve(C@p@C.T+measurement, C@p).T
            p = (np.eye(4)-l[t]@C)@p
        a = ab[t-1][0]
        p = a@p@a.T+process
    return np.asarray(gains[::-1]), l


class ReferenceBank:
    def __init__(self, path):
        with np.load(path) as z:
            self.states, self.actions = z['states'].copy(), z['actions'].copy()
            self.k, self.l, self.goals = z['feedback'].copy(), z['observer'].copy(), z['goals'].copy()
        shapes = [(15, 31, 4), (15, 30, 2), (15, 30, 2, 4), (15, 31, 4, 3), (15, 2)]
        for value, expected in zip([self.states, self.actions, self.k, self.l, self.goals], shapes):
            if value.shape != expected or not np.isfinite(value).all():
                raise ValueError('invalid reference bank')

    def select(self, observation, goal):
        y, goal = np.asarray(observation), np.asarray(goal)
        if y.shape != (3,) or goal.shape != (2,) or not np.isfinite(np.r_[y, goal]).all():
            raise ValueError('finite visual observation and commanded XY goal required')
        eligible = np.linalg.norm(self.goals-goal, axis=1) < 1e-8
        if not eligible.any():
            raise ValueError('goal absent from offline development reference bank')
        expected = (self.states[:, 0]+RENDER_OFFSET)@C.T
        costs = np.sum((expected-y)**2, axis=1)
        return int(np.argmin(np.where(eligible, costs, np.inf)))


class PlannedController:
    """Runtime interface deliberately accepts NO simulator state/scenario ID."""
    def __init__(self, bank, policy=None, mode='feedback'):
        if mode not in ('open_loop', 'feedback', 'residual'):
            raise ValueError('unknown planned-controller mode')
        if mode == 'residual' and policy is None:
            raise ValueError('residual mode requires a trained actor')
        self.bank, self.policy, self.mode = bank, policy, mode

    def reset(self, observation, commanded_goal):
        self.reference = self.bank.select(observation, commanded_goal)
        self.t = 0; self.goal = np.asarray(commanded_goal).copy()
        prior = self.bank.states[self.reference, 0]
        self.estimate = prior+self.bank.l[self.reference, 0]@(observation-nominal_observation(prior))
        self.last_action = np.zeros(2)
        return self.features()

    def base_action(self):
        j, t = self.reference, min(self.t, HORIZON-1)
        feedforward = self.bank.actions[j, t]
        if self.mode == 'open_loop':
            return feedforward.copy()
        return np.clip(feedforward-self.bank.k[j, t]@(self.estimate-self.bank.states[j, t]),
                       ACTION_LOW, ACTION_HIGH)

    def features(self):
        j, t = self.reference, min(self.t, HORIZON)
        reference = self.bank.states[j, t]
        preview = np.concatenate([self.bank.states[j, min(t+d, HORIZON), :2]-self.estimate[:2]
                                  for d in (4, 8)])
        return np.r_[(self.estimate-reference)/ERROR_SCALE, preview,
                     self.base_action()/ACTION_HIGH, self.goal-self.estimate[:2], t/HORIZON].astype(np.float32)

    def action(self, residual=None):
        if self.t >= HORIZON:
            raise RuntimeError('controller horizon completed')
        if residual is None:
            residual = (self.policy.predict(self.features(), deterministic=True)[0]
                        if self.mode == 'residual' else np.zeros(2))
        r = np.asarray(residual, dtype=float)
        if r.shape != (2,) or not np.isfinite(r).all():
            raise ValueError('finite two-dimensional residual action required')
        self.last_action = np.clip(self.base_action()+RESIDUAL_SCALE*np.clip(r, -1, 1),
                                   ACTION_LOW, ACTION_HIGH)
        return self.last_action.copy()

    def receive(self, delayed_observation):
        self.t += 1
        if self.t > HORIZON:
            raise RuntimeError('received observation beyond horizon')
        y = np.asarray(delayed_observation)
        if y.shape != (3,) or not np.isfinite(y).all():
            raise ValueError('finite three-dimensional observation required')
        # Before this update estimate concerns x_(t-1); the incoming image
        # measures exactly x_(t-1), except for the duplicate reset at t=1.
        corrected = self.estimate+self.bank.l[self.reference, self.t]@(y-nominal_observation(self.estimate))
        self.estimate = dynamics(corrected, self.last_action)
        return self.features()


def build_references(source_policy, output, iterations=300, clearance=.15, speed_margin=.025):
    import torch
    from .warmstart import rollout_tensor
    from .train_migration import load_ppo
    from .train_ppo import evaluate, sha
    torch.set_num_threads(1)
    model = load_ppo(source_policy, device='cpu')
    _, _, seeds = evaluate(model, quiet=True)
    all_x, all_u, all_k, all_l, records = [], [], [], [], []
    started = time.monotonic()
    for scenario in range(15):
        goal = torch.tensor((GOAL_LOW[scenario, :2]+GOAL_HIGH[scenario, :2])/2)
        low, high, circles = torch.tensor(STATE_LOW), torch.tensor(STATE_HIGH), torch.tensor(OBSTACLES)
        def objective(flat):
            u = torch.tensor(flat.reshape(30, 2), dtype=torch.float64, requires_grad=True)
            x = rollout_tensor(INITIAL_STATES[scenario], u)
            tightening = x.new_tensor([.025, .025, .025, speed_margin])
            margin = torch.relu(low+tightening-x)+torch.relu(x-high+tightening)
            gaps = torch.linalg.vector_norm(x[:, None, :2]-circles[None, :, :2], dim=2)-circles[:, 2]
            loss = (10000*(x[-1, :2]-goal).square().sum()+30*x[-1, 3].square()
                    +100000*margin.square().sum()+100000*torch.relu(clearance-gaps).square().sum()
                    +.2*(x[:, :2]-goal).square().sum()+.1*u.square().sum()
                    +.5*(u[1:]-u[:-1]).square().sum())
            loss.backward()
            return float(loss.detach()), u.grad.numpy().ravel()
        fit = minimize(objective, (seeds[scenario]/ACTION_HIGH).ravel(), method='L-BFGS-B', jac=True,
                       bounds=[(-1, 1)]*60, options={'maxiter': iterations, 'ftol': 1e-10, 'maxls': 30})
        actions = fit.x.reshape(30, 2)*ACTION_HIGH
        states = [INITIAL_STATES[scenario].copy()]
        for action in actions:
            states.append(dynamics(states[-1], action))
        states = np.asarray(states)
        flags = [constraints(x, scenario) for x in states]
        success = flags[-1]['in_goal'] and all(f['in_bounds'] and not f['collision'] for f in flags)
        actual_clearance = np.min(np.linalg.norm(states[:, None, :2]-OBSTACLES[None, :, :2], axis=2)-OBSTACLES[:, 2])
        row = dict(scenario=scenario, success=bool(success), optimizer_converged=bool(fit.success),
                   minimum_clearance=float(actual_clearance), objective=float(fit.fun), iterations=int(fit.nit))
        print(json.dumps(row), flush=True); records.append(row)
        if not success:
            raise RuntimeError('reference is infeasible in original task: %d' % scenario)
        k, l = feedback_gains(states, actions)
        all_x.append(states); all_u.append(actions); all_k.append(k); all_l.append(l)
    output = Path(output); output.mkdir(parents=True)
    np.savez_compressed(output/'references.npz', states=all_x, actions=all_u, feedback=all_k, observer=all_l,
                        goals=(GOAL_LOW[:, :2]+GOAL_HIGH[:, :2])/2)
    report = dict(source_policy=str(source_policy), source_sha256=sha(source_policy),
                  references=records, requested_clearance=clearance, speed_margin=speed_margin,
                  runtime_seconds=time.monotonic()-started,
                  selection='initial visual measurement and goal only; no true scenario ID',
                  initial_prior='catalogue of original development initial states; not supplied true runtime state',
                  planning='nominal nonlinear trajectory optimization; not robust SLS',
                  feedback='finite-horizon LQR with delayed linearized Kalman-style observer',
                  covariance_role='design tuning parameters, not calibrated error distribution', safety_certificate=False)
    (output/'config.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-policy', required=True); p.add_argument('--output-dir', required=True)
    p.add_argument('--iterations', type=int, default=300)
    p.add_argument('--speed-margin', type=float, default=.025)
    a = p.parse_args()
    if Path(a.output_dir).exists() or a.iterations < 1 or not 0 <= a.speed_margin < 1.5:
        p.error('new output directory and positive iteration budget required')
    build_references(a.source_policy, a.output_dir, a.iterations, speed_margin=a.speed_margin)
