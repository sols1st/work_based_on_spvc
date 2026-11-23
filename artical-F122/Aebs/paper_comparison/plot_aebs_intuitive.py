"""Paper figures from all registered AEBS conditions and training seeds."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

METHODS = ['gray32_mlp', 'gray32_cls', 'gray32_spatial']
LABELS = ['原 SPVC 结构 MLP', '旧 DINO 全局读出', '新 DINO 空间读出']
COLORS = ['#406580', '#ba772b', '#14816c']
CONDITIONS = ['identity', 'brightness_0p7', 'brightness_1p3', 'contrast_0p7', 'contrast_1p3', 'blur_0p5', 'blur_1p0']
CONDITION_LABELS = ['正常', '亮度\n70%', '亮度\n130%', '对比度\n70%', '对比度\n130%', '模糊\n0.5 像素', '模糊\n1.0 像素']


def read(path):
    return json.loads(Path(path).read_text())


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--static-csv', type=Path, required=True)
    parser.add_argument('--nominal', type=Path, required=True)
    parser.add_argument('--shift', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.static_csv.open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 189, 'Every one of 27 frozen models and seven conditions is required'
    plt.rcParams.update({'font.family':'Arial Unicode MS', 'axes.unicode_minus':False, 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    exported = []
    for i, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS)):
        for ax, ticks, condition_values, budget_values in [
            (axes[0], [0, 1, 2], ['identity']*3, [400, 10000, 24000]),
            (axes[1], list(range(7)), CONDITIONS, [24000]*7),
        ]:
            values = []
            for condition, budget in zip(condition_values, budget_values):
                subset = [r for r in rows if r['mode'] == method and r['condition'] == condition and int(r['budget']) == budget]
                assert len(subset) == 3
                values.append([float(r['mae_m']) for r in subset])
                exported.append(dict(method=method, condition=condition, budget=budget,
                                     mae_mean_m=float(np.mean(values[-1])), per_seed_mae_m=values[-1]))
            values = np.asarray(values)
            means = values.mean(1)
            ax.plot(ticks, means, '-o', color=color, label=label, markersize=5)
            ax.fill_between(ticks, values.min(1), values.max(1), color=color, alpha=.12)
            ax.set_yscale('log')
            ax.set_ylabel('平均测距误差（米，对数刻度）')
            ax.grid(alpha=.2)
    axes[0].set_xticks([0, 1, 2], ['400', '10k', '24k'])
    axes[0].set_xlabel('原始 RGB 图片总包量')
    axes[0].set_title('正常画面：原结构仍然更准')
    axes[0].legend(loc='best', frameon=True)
    axes[1].set_xticks(range(7), CONDITION_LABELS, fontsize=9)
    axes[1].set_title('24k 数据：变暗时 DINO 更准，其他条件有取舍')
    fig.suptitle('同像素、同四次选型机会；曲线为三个种子均值，阴影为最小至最大值', fontsize=13)
    fig.savefig(args.output/'aebs_accuracy_all_conditions.png', dpi=180)
    fig.savefig(args.output/'aebs_accuracy_all_conditions.svg')
    plt.close(fig)

    runs = list(read(args.nominal/'summary.json')['arms'].values()) + list(read(args.shift/'summary.json')['arms'].values())
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.1), constrained_layout=True)
    success = {}
    for ax, actor, title in zip(axes, ['original_ppo', 'current_robust'], ['原发布 PPO 控制器', '当前鲁棒控制器']):
        matrix = np.zeros((3, 7))
        for i, mode in enumerate(METHODS):
            for j, condition in enumerate(CONDITIONS):
                group = [r for r in runs if r['arm']['actor'] == actor and r['model']['mode'] == mode
                         and r['model']['budget'] == 24000 and r['model'].get('condition', 'identity') == condition]
                assert len(group) == 3
                matrix[i, j] = 100*np.mean([r['outcomes']['success']/r['n'] for r in group])
        success[actor] = matrix.tolist()
        im = ax.imshow(matrix, vmin=0, vmax=100, cmap='YlGnBu', aspect='auto')
        for i in range(3):
            for j in range(7):
                ax.text(j, i, f'{matrix[i,j]:.1f}', ha='center', va='center', fontsize=10,
                        color='white' if matrix[i,j] > 65 else 'black')
        ax.set_xticks(range(7), CONDITION_LABELS, fontsize=9)
        ax.set_yticks(range(3), ['原结构 MLP', '旧 DINO', '新空间 DINO'])
        ax.set_title(title)
    fig.colorbar(im, ax=axes, label='任务完成率（%）', shrink=.78)
    fig.suptitle('24k 数据；每个种子同 1000 个场景；所有失败保留在分母中', fontsize=13)
    fig.savefig(args.output/'aebs_completion_all_conditions.png', dpi=180)
    fig.savefig(args.output/'aebs_completion_all_conditions.svg')
    plt.close(fig)
    (args.output/'plotted_values.json').write_text(json.dumps(dict(static=exported, success=success,
        methods=METHODS, conditions=CONDITIONS), indent=2))
    print(json.dumps({'output': str(args.output), 'static_rows': len(rows), 'closed_loop_arms': len(runs)}))


if __name__ == '__main__':
    main()
