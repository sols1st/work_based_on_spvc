"""Recompute the paired AEBS table from episode records, contracts and checks."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_arm(run, label):
    path = run / label
    row = read(run / 'summary.json')[label]
    contract = read(path / 'contract.json')
    audit = read(path / 'audit.json')
    assert row['contract_sha256'] == digest(path / 'contract.json') == audit['contract_sha256']
    assert row['audit_sha256'] == digest(path / 'audit.json')
    assert row['holdout_sha256'] == digest(path / 'holdout.npz')
    assert contract['calibration_npz'] == digest(path / 'calibration.npz')
    assert contract['protocol_sha256'] == digest(run / 'protocol.json')
    assert audit['status'] == row['audit']
    with np.load(path / 'calibration.npz') as records:
        scores = records['scores']
        rank = math.ceil((len(scores) + 1) * .95)
        assert rank == row['rank'] == contract['calibration']['rank'] == 190
        assert row['q'] == np.sort(scores)[rank - 1] == audit['q']
    with np.load(path / 'holdout.npz') as records:
        holdout = dict(records)
        assert row['covered'] == np.sum(holdout['scores'] <= row['q'])
        counts = dict(zip(*np.unique(holdout['outcomes'], return_counts=True)))
        assert counts == row['outcomes']
    return row, holdout


def compare(primary, alternate, lib, budget, variant):
    baseline, first = read_arm(primary, f'{lib}__gray_{budget}')
    candidate, second = read_arm(alternate, f'{lib}__{variant}_{budget}')
    for key in ('initial', 'appearance', 'group_counts'):
        np.testing.assert_array_equal(first[key], second[key])
    delta = second['scores'] - first['scores']
    rng = np.random.default_rng(202611171903)
    indices = rng.integers(len(delta), size=(10000, len(delta)))
    lower, upper = np.percentile(delta[indices].mean(axis=1), [2.5, 97.5])
    return dict(library=lib, budget=budget, candidate=variant, images=candidate['library_images'],
                gray_q=baseline['q'], candidate_q=candidate['q'],
                q_reduction_percent=100 * (1 - candidate['q'] / baseline['q']),
                gray_covered=baseline['covered'], candidate_covered=candidate['covered'],
                gray_success=baseline['outcomes'].get('success', 0), candidate_success=candidate['outcomes'].get('success', 0),
                gray_unsafe=baseline['outcomes'].get('unsafe', 0), candidate_unsafe=candidate['outcomes'].get('unsafe', 0),
                gray_audit=baseline['audit'], candidate_audit=candidate['audit'],
                gray_mean_max_error=first['scores'].mean(), candidate_mean_max_error=second['scores'].mean(),
                delta_mean_max_error=delta.mean(), delta_ci_low=lower, delta_ci_high=upper,
                candidate_lower_score_episodes=(delta < 0).sum(), total_episodes=len(delta),
                gray_lookup_max=first['max_lookup_error_m'].max(), candidate_lookup_max=second['max_lookup_error_m'].max())


def write_episode_table(root, runs):
    fields = ('study', 'arm', 'phase', 'episode', 'initial_d_m', 'initial_v_mps', 'appearance_index',
              'max_error_m', 'mean_error_m', 'max_image_lookup_m', 'steps', 'outcome', 'terminal_d_m', 'terminal_v_mps')
    with (root / 'episodes.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for run in runs:
            for arm in sorted(read(run / 'summary.json')):
                for phase in ('calibration', 'holdout'):
                    with np.load(run / arm / (phase + '.npz')) as archive:
                        rows = dict(archive)
                    for i, initial in enumerate(rows['initial']):
                        writer.writerow([run.name, arm, phase, i, *initial, rows['appearance'][i], rows['scores'][i],
                                         rows['episode_mean_abs_error_m'][i], rows['max_lookup_error_m'][i],
                                         rows['steps'][i], rows['outcomes'][i], *rows['terminal_states'][i]])


def write_geometry_and_frame_errors(root):
    library = root / 'library'
    with np.load(library / 'predictions.npz') as archive:
        table = dict(archive)
    if (root / 'gray400/predictions.npz').exists():
        with np.load(root / 'gray400/predictions.npz') as archive:
            np.testing.assert_array_equal(archive['filename'], table['filename'])
            table['dino_gray_400'] = archive['observation']
    scale = read(library / 'manifest.json')['scale']
    groups = sorted(set(zip(table['weather'].tolist(), table['color'].tolist())))
    file_index = {name: index for index, name in enumerate(table['filename'])}
    geometry, frames = [], []
    for name, files in read(library / 'libraries.json').items():
        ids = np.array([file_index[f] for f in files])
        for group, (weather, color) in enumerate(groups):
            mask = (table['weather'][ids] == weather) & (table['color'][ids] == color)
            distances = np.unique(table['distance_m'][ids[mask]])
            i = np.argmax(np.diff(distances))
            geometry.append(dict(library=name, appearance_index=group, weather=weather, color=color,
                                 images=int(mask.sum()), minimum_m=distances[0], maximum_m=distances[-1],
                                 largest_internal_gap_m=distances[i + 1] - distances[i],
                                 gap_lower_m=distances[i], gap_upper_m=distances[i + 1]))
        models = [key for key in table if key.startswith(('dino_', 'gray_'))]
        for model in models:
            if name != 'common' and name.split('_')[-1] != model.split('_')[-1]:
                continue
            errors = np.abs(table[model][ids].astype(float) * scale - table['distance_m'][ids])
            frames.append(dict(library=name, model=model, images=len(ids), mae_m=errors.mean(),
                               p95_m=np.quantile(errors, .95), max_m=errors.max()))
    for filename, records in [('library_geometry.csv', geometry), ('frame_errors.csv', frames)]:
        with (root / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def main(root):
    primary = root / 'paired'
    rows = [compare(primary, primary, library, budget, 'dino')
            for budget in (400, 10000, 24000) for library in ('common', f'strict_{budget}')]
    if (root / 'gray_paired/summary.json').exists():
        rows.extend(compare(primary, root / 'gray_paired', library, 400, 'dino_gray')
                    for library in ('common', 'strict_400'))
    with (root / 'paired_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    runs = [primary, root / 'gray_paired'] if (root / 'gray_paired/summary.json').exists() else [primary]
    write_episode_table(root, runs)
    write_geometry_and_frame_errors(root)
    for row in rows:
        print(row['library'], row['budget'], row['candidate'], 'q', round(row['gray_q'], 4),
              round(row['candidate_q'], 4), 'success', row['gray_success'], row['candidate_success'],
              'certificate', row['gray_audit'], row['candidate_audit'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
