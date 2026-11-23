"""Optional all-support finite-horizon precision audit, separate from CP scores.

Search valid bounded-error witnesses first. If none is found, attempt a whole
initial-box enclosure. No witness is NOT a proof; enclosure failure is UNKNOWN.
This inherits float64 bounds without directed rounding from the existing code.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .lane_precision import TOLERANCES, digest, save


def witness(task, tolerance, context):
    initial = np.array(list(itertools.product(*[np.linspace(a, b, 3)
                       for a, b in zip(task.initial_lo, task.initial_hi)])))
    signs = np.array([[0., 0.]] + list(itertools.product([-1., 1.], repeat=2)))
    x = np.repeat(initial, len(signs), axis=0)
    error = np.tile(signs, (len(initial), 1))*task.error_hi
    states = [x.copy()]
    for t in range(80):
        x = task.step(x, error=error, context=context, t=t)
        states.append(x.copy())
        unsafe = (np.abs(x) > 1).any(axis=1)
        imprecise = np.abs(x[:, 0])*.85 > tolerance if t+1 >= 20 else np.zeros(len(x), bool)
        bad = np.flatnonzero(unsafe | imprecise)
        if len(bad):
            j = int(bad[0])
            return dict(step=t+1, context=context, initial_normalized=states[0][j].tolist(),
                constant_observation_error_normalized=error[j].tolist(),
                states_physical=(np.stack(states)[:, j]*[.85, .30]).tolist(),
                unsafe=bool(unsafe[j]), precision_violation=bool(imprecise[j]),
                lateral_cm=float(abs(x[j, 0])*.85*100),
                scope='valid bounded-error model trajectory, not a claim this error trace occurs in camera images')
    return None


def audit_one(camera, arm, seconds):
    import torch
    from .tasks import load_task
    from .spvt_reachability import verify_initial_box, ReachabilityConfig
    torch.set_num_threads(1)
    task = load_task('lane', root=camera/arm/'shadow')
    original_safe = task._safe
    rows = []
    for tolerance in TOLERANCES:
        found = next((hit for context in task.contexts
                      for hit in [witness(task, tolerance, context)] if hit is not None), None)
        if found is not None:
            rows.append(dict(tolerance_m=tolerance, status='counterexample', witness=found))
            continue

        def precise_safe(lo, hi, t, tol=tolerance):
            status = original_safe(lo, hi, t)
            if status != 'safe' or t < 20:
                return status
            if lo[0]*.85 >= -tol and hi[0]*.85 <= tol:
                return 'safe'
            if hi[0]*.85 < -tol or lo[0]*.85 > tol:
                return 'unsafe'
            return 'unknown'

        task._safe = precise_safe
        checks = [verify_initial_box(task, context=context, horizon=80,
                    config=ReachabilityConfig(mode='interval', timeout_seconds=seconds,
                        max_frontier=256, max_depth=16, falsification_paths=0)) for context in task.contexts]
        task._safe = original_safe
        passed = all(item['status'] == 'certified' for item in checks)
        rows.append(dict(tolerance_m=tolerance, status='numerical_enclosure_pass' if passed else 'unknown',
                         checks=checks))
    result = dict(actor_sha256=digest(camera/arm/'package/actor.pt'),
        calibration_sha256=digest(camera/arm/'calibration.json'), radius=float(task.error_hi[0]),
        grid_witness_search='3x3 initial states, zero and four constant error corners, all three curvatures',
        whole_initial_box=True, horizon=80, deadline=20, dt=.1, rows=rows,
        outward_rounded=False, precision_cp_is_different=True,
        note='A counterexample refutes the all-error-set precision specification, not observed camera accuracy or the safety-only barrier. Unknown is not a counterexample.')
    save(camera/arm/'precision_robust_audit.json', result)
    return dict(arm=arm, results={str(row['tolerance_m']): row['status'] for row in rows})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=3.)
    args = p.parse_args()
    if args.seconds <= 0:
        p.error('positive verification time budget required')
    protocol = json.loads((args.camera/'protocol.json').read_text())
    rows = []
    for arm in protocol['arms']:
        row = audit_one(args.camera, arm, args.seconds)
        rows.append(row)
        print(json.dumps(row), flush=True)
    save(args.camera/'precision_robust_audit.json', dict(rows=rows, seconds_per_context=args.seconds,
         source_sha256=digest(__file__), whole_initial_box=True, outward_rounded=False))


if __name__ == '__main__':
    main()
