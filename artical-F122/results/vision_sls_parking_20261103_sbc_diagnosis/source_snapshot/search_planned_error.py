"""Search admissible visual-box counterexamples; no invented noise law."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from .core import INITIAL_STATES, nominal_observation, dynamics, constraints
from .planning_feedback import ReferenceBank, PlannedController
from .perception_error import radius
from .train_migration import load_ppo
from .train_ppo import sha


def trial(bank, policy, error_model, scenario, direction, scale):
    controller = PlannedController(bank, policy, mode='residual')
    x = INITIAL_STATES[scenario].copy()
    controller.reset(nominal_observation(x), bank.goals[scenario])
    states = [x.copy()]; estimates = [controller.estimate.copy()]
    actions, errors, radii = [], [], []
    for t in range(30):
        try:
            r = radius(x[None], error_model)[0]
        except ValueError:
            return dict(status='perception_domain_exit', step=t), None
        # The first new input repeats the zero-error initial frame.
        error = np.zeros(3) if t == 0 else scale*r*np.asarray(direction)
        action = controller.action()
        nxt = dynamics(x, action)
        controller.receive(nominal_observation(x)+error)
        actions.append(action); errors.append(error); radii.append(r)
        states.append(nxt.copy()); estimates.append(controller.estimate.copy()); x = nxt
        flag = constraints(x, scenario)
        if flag['collision'] or not flag['in_bounds']:
            data = dict(states=np.asarray(states), estimates=np.asarray(estimates),
                        actions=np.asarray(actions), errors=np.asarray(errors), radii=np.asarray(radii))
            return dict(status='unsafe_reached', step=t+1, collision=flag['collision'],
                        out_of_bounds=not flag['in_bounds'], state=x.tolist(),
                        maximum_radius_fraction=float(np.max(np.abs(data['errors'])/data['radii'])),
                        selected_reference=controller.reference), data
    return dict(status='safe_at_horizon', step=30), None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', required=True); p.add_argument('--error-model', required=True)
    p.add_argument('--output-dir', required=True)
    a = p.parse_args(); root = Path(a.candidate); out = Path(a.output_dir)
    if out.exists():
        p.error('new output directory required')
    model = json.loads(Path(a.error_model).read_text())
    if model['provenance']['projection_sha256'] != sha(root/'projection.pt'):
        raise ValueError('CP/projection mismatch')
    bank = ReferenceBank(root/'references.npz'); policy = load_ppo(root/'ppo.zip', device='cpu')
    out.mkdir(parents=True)
    config = dict(vars(a), ppo_sha256=sha(root/'ppo.zip'), references_sha256=sha(root/'references.npz'),
                  error_model_sha256=sha(a.error_model), initial_visual_error='zero', process_noise='zero',
                  scope='robust universal-error condition, NOT a stochastic noise probability',
                  scales=[1., .5, .25, .1, .05], direction_family='eight constant normalized box corners')
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    records = []; witnesses = []
    for scale in config['scales']:
        counts = dict(unsafe_reached=0, perception_domain_exit=0, safe_at_horizon=0)
        first = None
        for scenario in range(15):
            for direction in itertools.product([-1., 1.], repeat=3):
                row, data = trial(bank, policy, model, scenario, direction, scale)
                row.update(scale=scale, scenario=scenario, direction=list(direction))
                counts[row['status']] += 1; records.append(row)
                if data is not None and first is None:
                    first = dict(row, archive='witness_scale_'+str(scale)+'.npz')
                    np.savez_compressed(out/first['archive'], **data)
                    witnesses.append(first)
        print(scale, counts, flush=True)
    summary = dict(config=config, witnesses=witnesses, trials=records,
                   robust_nonincrease_impossible_with_initial_1_unsafe_20=bool(witnesses),
                   stochastic_sbc_impossibility_claimed=False, actual_image_failure_claimed=False,
                   safety_certificate=False)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
