"""Plot archived input pixels and paired trajectories for the comparison report."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(observations, trajectories, output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Arial Unicode MS', 'axes.unicode_minus': False, 'font.size': 11})
    with np.load(observations) as data:
        pixels = data['pixels']
        conditions = data['conditions'].tolist()
        distances = data['distance_m']
        filenames = data['filename']
    figures = [
        ('brightness_inputs.png', ['identity', 'brightness_0p7', 'brightness_1p3'],
         ['原始输入', '亮度乘 0.7', '亮度乘 1.3']),
        ('blur_inputs.png', ['identity', 'blur_0p5', 'blur_1p0'],
         ['原始输入', '高斯模糊 σ = 0.5 像素', '高斯模糊 σ = 1.0 像素']),
    ]
    for name, keys, titles in figures:
        fig, axes = plt.subplots(3, 3, figsize=(9, 8.6), layout='constrained')
        for row in range(3):
            for col, condition in enumerate(keys):
                axes[row, col].imshow(pixels[conditions.index(condition), row], cmap='gray',
                                      vmin=0, vmax=255, interpolation='nearest')
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                if row == 0:
                    axes[row, col].set_title(titles[col])
                if col == 0:
                    axes[row, col].set_ylabel(f'{distances[row]:.2f} 米\n{filenames[row]}')
        fig.suptitle('归档实际模型输入，每帧 32×32 像素，按最近邻放大展示')
        fig.savefig(output/name, dpi=180)
        plt.close(fig)
    paths = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout='constrained')
    initial = None
    for key, label, color, style in [
        ('spvt_pid', 'SPVT 发布 PID 初始化', '#777777', '-'),
        ('same_target_e2e_2026112211', '原结构同目标监督重训', '#b18320', '--'),
        ('current_lowdim_2026112211', '二维状态估计＋解析反馈', '#2678b2', '-'),
    ]:
        path = trajectories/f'iid__{key}'/'test.npz'
        with np.load(path) as data:
            state = data['states_cm_degrees'][0]
            action = data['actions'][0]*30
        if initial is None:
            initial = state[0]
        np.testing.assert_array_equal(initial, state[0])
        axes[0].plot(np.arange(len(state))*.05, state[:, 0], label=label, color=color, linestyle=style)
        axes[1].plot(np.arange(len(action))*.05, action, color=color, linestyle=style)
        paths[str(path)] = sha(path)
    axes[0].axhspan(-5, 5, color='#8cbda2', alpha=.18)
    axes[0].axhline(5, color='#66997b', linestyle=':', linewidth=1)
    axes[0].axhline(-5, color='#66997b', linestyle=':', linewidth=1)
    for ax in axes:
        ax.axvline(2, color='#9b4545', linestyle=':', linewidth=1)
        ax.set_xlabel('时间（s）')
        ax.grid(alpha=.2)
    axes[0].set(title='横向位置与 ±5 cm 目标范围', ylabel='横向偏差（cm）')
    axes[1].set(title='相同转向限幅内的实际控制指令', ylabel='转向角（度）')
    axes[0].legend(fontsize=8)
    fig.suptitle(f'同一条归档测试场景，初始横偏 {initial[0]:.2f} cm，航向 {initial[1]:.2f}°')
    fig.savefig(output/'spvt_same_start_trajectory.png', dpi=180)
    plt.close(fig)
    record = dict(observations_sha256=sha(observations), pixel_display='fixed grayscale range 0..255, nearest-neighbor display',
                  image_rows=[dict(filename=str(f),distance_m=float(d)) for f,d in zip(filenames,distances)],
                  trajectory_selection='first registered test scenario, index0; fixed training seed2026112211; no metric-based selection',
                  trajectory_sources=paths, source_sha256=sha(Path(__file__)))
    (output/'illustrations_manifest.json').write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--trajectories', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.observations, args.trajectories, args.output)
