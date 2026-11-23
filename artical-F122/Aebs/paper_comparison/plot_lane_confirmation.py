"""Figures from the frozen new-image confirmation; no policy or sample selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .tasks import RESULTS


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=RESULTS/'paper_controller_image_confirmation_20261118_v2')
    parser.add_argument('--output', type=Path, default=RESULTS.parent.parent/'docs/assets/paper_backend_comparison_20261118')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.root/'summary.json').read_text())
    arms = ['anchor', 'spvt_first_seed']
    data = {arm: np.load(args.root/arm/'test.npz', allow_pickle=False) for arm in arms}
    assert np.array_equal(data[arms[0]]['states'][:, 0], data[arms[1]]['states'][:, 0])
    colors = ['#486581', '#1a806c']
    labels = ['Frozen anchor', 'SPVT-style training + our calibration/barrier']
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

    # Whole-trajectory paired points; no smoothing or per-frame independence.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
    specifications = [
        ('mean_abs_e_m', 'Mean absolute lateral error (cm)', 100),
        ('mean_abs_heading_rad', 'Mean absolute heading error (deg)', 180/np.pi),
        ('mean_steering_change_rad', 'Mean steering change (deg / step)', 180/np.pi),
    ]
    for ax, (key, title, scale) in zip(axes, specifications):
        values = []
        for arm in arms:
            d = data[arm]
            if key == 'mean_abs_e_m':
                v = np.abs(d['states'][:, 1:, 0]).mean(1)
            elif key == 'mean_abs_heading_rad':
                v = np.abs(d['states'][:, 1:, 1]).mean(1)
            else:
                v = np.abs(np.diff(.45*d['actions'], axis=1)).mean(1)
            values.append(v*scale)
        for a, b in zip(*values):
            ax.plot([0, 1], [a, b], color='#aab7c2', alpha=.24, lw=.7, zorder=1)
        for i, v in enumerate(values):
            ax.scatter(np.full(len(v), i), v, s=9, color=colors[i], alpha=.48, zorder=2)
            ax.scatter(i, np.mean(v), marker='D', s=90, color=colors[i], edgecolors='black', zorder=3)
        assert np.isclose(values[0].mean(), summary['paired_metrics'][key]['anchor_mean']*scale)
        assert np.isclose(values[1].mean(), summary['paired_metrics'][key]['spvt_mean']*scale)
        ax.set_xticks([0, 1], ['Anchor', 'Combined'])
        ax.set_ylabel(title)
        ax.set_xlim(-.25, 1.25)
        ax.grid(axis='y', alpha=.18)
    fig.suptitle('100 paired new-image trajectories | diamonds: mean | lines: matched scenarios', fontsize=13)
    fig.savefig(args.output/'lane_image_paired_metrics.png', dpi=180)
    fig.savefig(args.output/'lane_image_paired_metrics.svg')
    plt.close(fig)

    # Scene zero was declared before the second arm completed. Show exact
    # deterministic rerenders at fixed steps, not a selected best trajectory.
    scene = 0
    steps = [0, 20, 79]
    # Rendering-library versions differ across hosts. Use server-reconstructed
    # bytes that match the complete original logged image stream, not local
    # rerenders that can differ at polygon edges.
    provenance = json.loads((args.root/'scene0_reconstruction.json').read_text())
    image_archive = args.root/'scene0_images.npz'
    assert hashlib.sha256(image_archive.read_bytes()).hexdigest() == provenance['image_archive_sha256']
    frames = np.load(image_archive, allow_pickle=False)
    assert provenance['scene_index'] == scene and provenance['selected_steps'] == steps
    for arm in arms:
        record = provenance['arms'][arm]
        assert record['full_trajectory_hash_matches']
        assert record['recorded80_frame_sha256'] == record['reconstructed80_frame_sha256'] == str(data[arm]['images_sha256'][scene])
        assert hashlib.sha256((args.root/arm/'test.npz').read_bytes()).hexdigest() == record['test_sha256']
        for frame, metadata in zip(frames[arm], record['frames']):
            assert hashlib.sha256(frame.tobytes()).hexdigest() == metadata['raw_pixel_sha256']
    (args.output/'scene0_reconstruction.json').write_text(json.dumps(provenance, indent=2))
    fig = plt.figure(figsize=(13.5, 10), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.05])
    for row, arm in enumerate(arms):
        for col, step in enumerate(steps):
            state = data[arm]['states'][scene, step]
            frame = frames[arm][col]
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(frame)
            ax.set_title(f'{"Anchor" if row == 0 else "Combined"} | t={step*.1:.1f} s | e={state[0]*100:.2f} cm')
            ax.axis('off')
    names = ['Lateral error (cm)', 'Heading error (deg)', 'Steering command (deg)']
    for col, name in enumerate(names):
        ax = fig.add_subplot(grid[2, col])
        for row, arm in enumerate(arms):
            if col < 2:
                trace = data[arm]['states'][scene, :, col]*(100 if col == 0 else 180/np.pi)
            else:
                trace = data[arm]['actions'][scene]*.45*180/np.pi
            ax.plot(np.arange(len(trace))*.1, trace, label=labels[row], color=colors[row], lw=1.6)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(name)
        ax.grid(alpha=.2)
    fig.suptitle('Prespecified holdout scene 0 | deterministic camera rerenders and logged closed-loop states', fontsize=13)
    fig.legend(labels, loc='outside lower center', ncol=1, frameon=False)
    fig.savefig(args.output/'lane_image_scene0.png', dpi=180)
    plt.close(fig)
    print(json.dumps({'figures': str(args.output), 'scene': scene, 'steps': steps}))


if __name__ == '__main__':
    main()
