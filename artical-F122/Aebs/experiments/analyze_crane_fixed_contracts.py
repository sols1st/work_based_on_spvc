"""Post-hoc geometry and action-only ablation on the frozen crane confirmation batch."""
import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import compare_crane_contracts as grid
from ..curve_braking.evaluate import digest
from ..lane_keeping.cp import conformal_radius
from ..overhead_crane import certificate, plant


def read_json(path):
    return json.loads(path.read_text())


def verified_digest(path, expected):
    observed = digest(path)
    if observed != expected:
        raise ValueError(f"Input digest mismatch: {path}")
    return observed


def exact_geometry(k, q, initial_scale):
    """For fixed uniform per-axis command support solve all r_i * scale + b_i < 1."""
    base = np.array([Fraction(str(v)) for v in plant.INITIAL])
    rows = []
    for name, step, row, offset, limit in certificate.planes(k, q, plant.HORIZON, exact=True):
        axes = 2 if name in ('total_accel', 'payload_diagonal') else 1
        slope = axes * (abs(row) @ base) / limit
        intercept = axes * offset / limit
        rows.append((name, step, slope, intercept))
    if any(slope < 0 or intercept >= 1 for _, _, slope, intercept in rows):
        raise ValueError('Expected a feasible origin and monotone support bounds')
    supremum = min((1 - intercept) / slope for _, _, slope, intercept in rows if slope > 0)
    active = [{"constraint": name, "step": step} for name, step, slope, intercept in rows
              if slope * supremum + intercept == 1]
    if max(slope * supremum + intercept for _, _, slope, intercept in rows) != 1:
        raise ArithmeticError('Failed exact support-function boundary check')
    upper = max(slope * initial_scale + intercept for _, _, slope, intercept in rows)
    return dict(upper=float(upper), upper_exact=str(upper), passes=upper < 1,
                scale_supremum=float(supremum), scale_supremum_exact=str(supremum),
                active_at_supremum=active)


def split_scores(path, k, methods):
    with np.load(path) as records:
        errors = records['errors']
    command = k[0] * errors[:, :, :2] + .5 * k[2] * errors[:, :, 2:]
    row_scales = grid.direction_scales(k, np.ones(4))
    scores = {key: grid.trajectory_scores(errors, k, definition)
              for key, definition in methods.items()}
    scores['action_only'] = np.max(np.abs(command) / row_scales, axis=(1, 2))
    if (not np.all(scores['action_only'] <= scores['box_and_action'])
            or not np.all(scores['box_and_action'] <= scores['box'] + 1e-14)):
        raise ValueError('Unexpected ordering of nested trajectory scores')
    return scores


def analyze(grid_dir, confirm_dir):
    grid_protocol = read_json(grid_dir / 'protocol.json')
    protocol = read_json(confirm_dir / 'protocol.json')
    certificates = read_json(confirm_dir / 'certificates.json')
    confirmation = read_json(confirm_dir / 'summary.json')
    k = grid_protocol['k']
    methods = {name: grid_protocol['methods'][name] for name in protocol['names']}
    if set(methods) != {'box', 'box_and_action'} or any(
            d['weights'] != [1.] * 4 for d in methods.values()):
        raise ValueError('This analysis requires unit visual-coordinate weights in both arms')
    scale = Fraction(str(protocol['scale']))
    source = Path(__file__).resolve()
    expected_files = {
        grid_dir / 'protocol.json': protocol['grid_protocol'],
        confirm_dir / 'protocol.json': certificates['protocol'],
        confirm_dir / 'certificates.json': confirmation['certificates'],
        confirm_dir / 'calibration.npz': certificates['calibration'],
        confirm_dir / 'test.npz': confirmation['test'],
        Path(grid.__file__): grid_protocol['source'],
    }
    code_root = source.parents[1]
    for relative in ('overhead_crane/certificate.py', 'overhead_crane/plant.py', 'lane_keeping/cp.py'):
        expected_files[code_root / relative] = grid_protocol['bindings']['source'][relative]
    input_hashes = {}
    for path, expected in expected_files.items():
        if path.parent == grid_dir:
            label = 'grid/' + path.name
        elif path.parent == confirm_dir:
            label = 'confirmation/' + path.name
        else:
            label = str(path.resolve().relative_to(code_root.parent))
        input_hashes[label] = verified_digest(path, expected)
    input_hashes['confirmation/summary.json'] = digest(confirm_dir / 'summary.json')
    cal = split_scores(confirm_dir / 'calibration.npz', k, methods)
    held = split_scores(confirm_dir / 'test.npz', k, methods)
    halfwidth_factor = abs(Fraction(str(k[0]))) + abs(Fraction(str(k[2]))) / 2
    arms = {}
    for key, scores in cal.items():
        if len(scores) != protocol['calibration'][0] or len(held[key]) != protocol['test'][0]:
            raise ValueError('Unexpected split sizes')
        q, rank = conformal_radius(scores, protocol['alpha'])
        action_bound = halfwidth_factor * Fraction.from_float(q)
        visual_bound = None if key == 'action_only' else q
        if key == 'box_and_action':
            visual_bound = q / grid.JOINT_VISUAL_WEIGHT
        covered = int((held[key] <= q).sum())
        geometry = exact_geometry(k, q, scale)
        if key in methods:
            old = confirmation['fits'][key]
            if not (q == old['q'] and rank == old['rank'] and covered == old['covered']
                    and geometry['upper_exact'] == old['upper_exact']
                    and [float(action_bound)] * 2 == old['action_halfwidths']
                    and [visual_bound] * 4 == old['visual_halfwidths']):
                raise ArithmeticError(f"Independent replay differs from archived {key} certificate")
        arms[key] = dict(q=q, rank=rank, visual_halfwidth_per_coordinate=visual_bound,
                         command_halfwidth_per_axis=float(action_bound),
                         command_halfwidth_exact=str(action_bound), held_covered=covered, **geometry)
    box_scale = Fraction(arms['box']['scale_supremum_exact'])
    joint_scale = Fraction(arms['box_and_action']['scale_supremum_exact'])
    return dict(analysis='Fixed-contract post-hoc geometric supremum and action-only ablation',
                scope='Strict supremum of the published sufficient certificate for each frozen tube. '
                      'The calibration and held-out sampling law uses initial scale 1.5 only. '
                      'Action-only controls the command projection and leaves its visual nullspace unbounded.',
                input_hashes=input_hashes,
                analysis_source_sha256=digest(source), alpha=protocol['alpha'],
                calibration_n=len(cal['box']), test_n=len(held['box']), initial_scale=float(scale),
                arms=arms, joint_over_box_scale_gain=float(joint_scale / box_scale - 1),
                joint_over_box_scale_gain_exact=str(joint_scale / box_scale - 1))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--grid', type=Path, default=Path('results/crane_contract_benefit_20261117_v1'))
    cli.add_argument('--confirmation', type=Path, default=Path('results/crane_contract_confirmation_20261117_v1'))
    cli.add_argument('--output', type=Path, default=Path('results/crane_fixed_contract_analysis_20261117/summary.json'))
    args = cli.parse_args()
    output = json.dumps(analyze(args.grid, args.confirmation), indent=2, allow_nan=False) + '\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.read_text() != output:
            raise FileExistsError(f'Results differ; choose a new output path: {args.output}')
    else:
        with args.output.open('x') as handle:
            handle.write(output)
    print(args.output)
