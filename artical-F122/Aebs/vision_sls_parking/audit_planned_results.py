"""Replay saved image experiments, including delayed observation and memory."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .core import dynamics, constraints, ACTION_HIGH, GOAL_LOW, GOAL_HIGH
from .planning_feedback import ReferenceBank, PlannedController
from .planned_transition import PlannedTransition
from .train_migration import load_ppo
from .train_ppo import sha


def audit(directory):
    root = Path(directory); cfg = json.loads((root/'config.json').read_text())
    results = json.loads((root/'metrics.json').read_text())
    assert sha(cfg['references']) == cfg['references_sha256']
    assert sha(cfg['projection']) == cfg['projection_sha256']
    bank = ReferenceBank(cfg['references']); records = {}
    for key, result in results.items():
        mode = key.split('_jitter_')[0]
        path = cfg.get(mode+'_policy')
        policy = load_ppo(path, device='cpu') if path else None
        if path:
            assert sha(path) == cfg[mode+'_sha256']
        with np.load(root/(key+'.npz')) as z:
            x, u = z['states'], z['actions']
            capture = z['capture_states'].reshape(15, 31, 4)
            y = z['capture_observations'].reshape(15, 31, 3)
            error = z['residuals'].reshape(15, 31, 3)
            expected_capture = np.concatenate([x[:, :1], x[:, :-1]], axis=1)
            np.testing.assert_array_equal(capture, expected_capture)
            assert (np.abs(u) <= ACTION_HIGH+1e-12).all()
            maximum_dynamics = 0.; maximum_action = 0.; maximum_memory = 0.; maximum_torch = 0.
            transition = PlannedTransition(bank, policy) if mode != 'legacy' else None
            for j in range(15):
                flags = [constraints(s, j) for s in x[j]]
                collision = any(f['collision'] for f in flags)
                bounds = any(not f['in_bounds'] for f in flags)
                success = flags[-1]['in_goal'] and not (collision or bounds)
                row = result['image']['per_scenario'][j]
                assert (collision, bounds, success) == (row['collision'], row['out_of_bounds'], row['success'])
                if mode != 'legacy':
                    controller = PlannedController(bank, mode='open_loop' if mode == 'open_loop' else 'feedback')
                    obs = controller.reset(y[j, 0], (GOAL_LOW[j, :2]+GOAL_HIGH[j, :2])/2)
                    assert controller.reference == row['selected_reference']
                    maximum_memory = max(maximum_memory, np.max(np.abs(controller.estimate-z['estimates'][j, 0])))
                for t in range(30):
                    maximum_dynamics = max(maximum_dynamics, np.max(np.abs(dynamics(x[j, t], u[j, t])-x[j, t+1])))
                    if mode != 'legacy':
                        action = controller.action(np.zeros(2) if policy is None else policy.predict(obs, deterministic=True)[0])
                        maximum_action = max(maximum_action, np.max(np.abs(action-u[j, t])))
                        obs = controller.receive(y[j, t+1])
                        maximum_memory = max(maximum_memory, np.max(np.abs(controller.estimate-z['estimates'][j, t+1])))
                        if mode != 'open_loop':
                            state = torch.tensor(np.r_[x[j, t], z['estimates'][j, t]][None])
                            with torch.no_grad():
                                predicted = transition(state, controller.reference, t, torch.tensor(error[j, t+1:t+2])).numpy()[0]
                            maximum_torch = max(maximum_torch, np.max(np.abs(predicted-np.r_[x[j, t+1], z['estimates'][j, t+1]])))
            assert maximum_dynamics < 1e-10 and maximum_action < 2e-6 and maximum_memory < 2e-6 and maximum_torch < 2e-6
            records[key] = dict(transitions=450, delayed_frames=465, exact_delay_pairing=True,
                                max_dynamics_error=float(maximum_dynamics), max_action_replay_error=float(maximum_action),
                                max_memory_replay_error=float(maximum_memory), max_torch_transition_error=float(maximum_torch),
                                outcome_counts_recomputed=True, archive_sha256=sha(root/(key+'.npz')))
    report = dict(passed=True, records=records, scope='numerical interface consistency, not safety verification')
    (root/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+')
    for directory in parser.parse_args().directories:
        report = audit(directory)
        print(directory, 'passed', len(report['records']), 'comparisons', flush=True)
