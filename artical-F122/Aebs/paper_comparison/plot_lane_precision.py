"""Plots all registered arms; no scene/seed selection by observed benefit."""
import argparse
import json
from pathlib import Path

import numpy as np

from .lane_precision import ARMS, SEEDS, save


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    for font in ['PingFang SC', 'Arial Unicode MS', 'Noto Sans CJK SC']:
        if any(f.name == font for f in font_manager.fontManager.ttflist):
            plt.rcParams['font.sans-serif'] = [font]
            break
    plt.rcParams['axes.unicode_minus'] = False
    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.camera/'summary.json').read_text())
    labels = ['旧控制器', '精度训练', '精度＋最坏误差训练']
    groups = [['anchor']]+[[str(seed)+'__'+arm for seed in SEEDS] for arm in ARMS]
    values = dict(rmse_cm=[], strict_5cm_percent=[], precision_cp_cm=[])
    for group in groups:
        rows = [summary['per_arm'][key] for key in group]
        values['rmse_cm'].append([r['metrics']['rmse_e_m']['mean']*100 for r in rows])
        values['strict_5cm_percent'].append([r['metrics']['strict_05cm']['mean']*100 for r in rows])
        values['precision_cp_cm'].append([r['precision_calibration']['radius_m']*100
                                         if r['precision_calibration']['finite_bound'] else None for r in rows])
    save(args.output/'plotted_values.json', values)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    colors = ['#7c899a', '#377eb8', '#cf6c38']
    for ax, key, title, ylabel in zip(axes, values,
            ['实际横向 RMSE：越低越好', '2–8 秒全程在 ±5 cm 内且安全', '轨迹 CP 校准的统计精度界'],
            ['cm', '达标率（%）', '最大横向偏差上界（cm）']):
        rows = values[key]
        for i, group in enumerate(rows):
            finite = [x for x in group if x is not None]
            if len(finite) != len(group):
                ax.text(i, 0, '含无穷界', ha='center')
                continue
            mean = np.mean(finite)
            ax.bar(i, mean, color=colors[i], alpha=.75, width=.6)
            ax.scatter(i+np.linspace(-.12, .12, len(finite)), finite, c='black', s=22, zorder=3)
            ax.annotate('%.2f'%mean, (i, max(finite)), xytext=(0, 8), textcoords='offset points', ha='center')
        ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=11); ax.set_ylabel(ylabel); ax.set_ylim(bottom=0)
        finite_values = [v for group in rows for v in group if v is not None]
        if finite_values:
            ax.set_ylim(0, max(finite_values)*1.20)
        ax.grid(axis='y', alpha=.2)
        if key == 'strict_5cm_percent':
            ax.set_ylim(0, 115)
    fig.suptitle('同一视觉前端、同一车速与安全约束；黑点为各训练种子', fontsize=13)
    status = []
    for label, group in zip(labels, groups):
        statuses = [summary['per_arm'][name]['barrier']['status'] for name in group]
        status.append('%s：%d/%d 屏障通过'%(label, statuses.count('pass'), len(group)))
    fig.text(.5, .045, '；'.join(status), ha='center', fontsize=10)
    fig.text(.5, .01, '柱为种子均值；CP 界的均值不构成新证书。统计精度界不是全扰动精度证书，也不是完整原生 SPVC 对比。', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .11, 1, .93])
    fig.savefig(args.output/'precision_comparison.png', dpi=180)
    fig.savefig(args.output/'precision_comparison.svg')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for group, label, color in zip(groups, labels, colors):
        data = np.load(args.camera/group[0]/'holdout.npz')
        for i, ax in enumerate(axes.flat):
            e = data['states'][i, :, 0]*100
            ax.plot(np.arange(len(e))*.1, e, label=label, color=color)
            ax.set_title('预先固定测试序号 %d；曲率 %.2f'%(i, data['kappas'][i]))
    for ax in axes.flat:
        ax.axvline(2, color='gray', ls=':')
        ax.axhspan(-5, 5, color='green', alpha=.08)
        ax.set_ylabel('横向偏差（cm）'); ax.set_xlabel('时间（s）'); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('前四条新图像闭环轨迹；两种精调均展示预定第一个训练种子')
    fig.tight_layout()
    fig.savefig(args.output/'paired_trajectories.png', dpi=180)
    plt.close(fig)
    examples = args.camera/'camera_examples/provenance.json'
    if examples.exists():
        from PIL import Image
        frames = json.loads(examples.read_text())['frames']
        fig, axes = plt.subplots(3, 3, figsize=(10, 9))
        for i, frame in enumerate(frames):
            ax = axes.flat[i]
            ax.imshow(Image.open(examples.parent/frame['filename']))
            e, theta = frame['physical_state']
            ax.set_title('%s，t=%.1fs\ne=%.2fcm，航向=%.2f°' %
                         (labels[i//3], frame['seconds'], e*100, np.degrees(theta)), fontsize=10)
            ax.axis('off')
        fig.suptitle('同一初始场景下的真实仿真相机输入；完整像素序列哈希已核对')
        fig.tight_layout()
        fig.savefig(args.output/'camera_observations.png', dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    main()
