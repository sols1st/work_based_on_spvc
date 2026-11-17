"""Plot paired proxy effects for three different camera-control systems."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(summary, out):
    labels = ['5D braking', '6D hover', '8D crane']
    fig, (radius, dist) = plt.subplots(1, 2, figsize=(8, 2.8), layout='constrained')
    for row, key in enumerate(['braking', 'hover', 'crane']):
        item = summary[key]
        for offset, pair in zip([-.13, 0., .13], item['seeds']):
            change = 100 * (pair['neural']['q'] / pair['linear']['q'] - 1)
            radius.scatter(change, row + offset, s=41, marker='o', color='#16698D', zorder=3)
        est = item['neural_minus_linear']['terminal_distance_m']
        mean, low, high = [1000 * est[k] for k in ('mean', 'low', 'high')]
        dist.errorbar(mean, row, xerr=[[mean - low], [high - mean]], fmt='o',
                      color='#965429', capsize=4, zorder=3)
    for ax in (radius, dist):
        ax.axvline(0, color='#555555', linewidth=.8, linestyle='--')
        ax.grid(axis='x', alpha=.14)
        ax.set_yticks([0, 1, 2], labels)
        ax.set_ylim(2.42, -.42)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    radius.set(title='(a) Calibrated trajectory radius', xlabel='Relative change from linear proxy (%)')
    dist.set(title='(b) Blind terminal distance to target', xlabel='Neural minus linear (mm), paired 95% interval')
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.png', '.svg'):
        fig.savefig(out.with_suffix(ext), dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=Path('results/high_dim_proxy_comparison_20261117/summary.json'))
    parser.add_argument('--output', type=Path, default=Path('../docs/assets/high_dim_proxies_20261117/paired_effects'))
    args = parser.parse_args()
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'svg.fonttype': 'none'})
    plot(json.loads(args.summary.read_text()), args.output)
