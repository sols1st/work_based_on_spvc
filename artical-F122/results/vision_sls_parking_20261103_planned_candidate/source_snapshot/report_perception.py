"""Paired parking perception report; never labels this as controller success."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--opengl-dir', type=Path, required=True)
    p.add_argument('--tiny-dir', type=Path, required=True)
    p.add_argument('--matched-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists(): p.error('new output required')
    dirs = [args.opengl_dir, args.tiny_dir, args.matched_dir]
    metrics = [json.loads((d/'metrics.json').read_text()) for d in dirs]
    arrays = [np.load(d/'samples.npz', allow_pickle=False) for d in dirs]
    if (any(m['status'] != 'frozen_perception_diagnostic_complete' or m['training'] for m in metrics)
            or not np.array_equal(arrays[0]['states'], arrays[1]['states'])
            or not np.array_equal(arrays[0]['target'], arrays[1]['target'])
            or not np.array_equal(arrays[2]['source_indices'], metrics[2]['same_state_source_comparison']['source_indices'])):
        raise ValueError('not matching frozen/pairwise inputs')
    config = [json.loads((d/'config.json').read_text()) for d in dirs]
    for suffix in ['dinov2_vitg14_pretrain.pth', 'unicycle_checkpoint_observability_ft_075w_3pi4_097_300000_post.pt']:
        values = [next(v for k, v in c['source_sha256'].items() if k.endswith(suffix)) for c in config]
        if len(set(values)) != 1: raise ValueError('different model hashes')
    gl, tiny = metrics[0]['new_rgb'], metrics[1]['new_rgb']
    source = metrics[2]['same_state_source_comparison']['supplied_feature_metrics']
    rerendered = metrics[2]['new_rgb']
    delta = metrics[2]['same_state_source_comparison']
    report = dict(status='perception_transfer_gap_measured_not_controller_performance',
                  rendering_comparison=dict(states=gl['samples'], OpenGL=gl, TinyRenderer=tiny),
                  exact_same_source_state_comparison=dict(states=source['samples'], supplied_features=source,
                                                         rerendered_OpenGL=rerendered),
                  feature_cosine_mean=delta['feature_cosine_mean'],
                  projection_output_gap_linf_mean=delta['projection_output_gap_linf_mean'],
                  training=False, PPO_tested=False, SBC_tested=False, formal_guarantee=False,
                  next_step='adapt projection to current OpenGL RGB; preserve observability and scale; then retrain corresponding PPO')
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    bars = axes[0].bar(['TinyRenderer', 'OpenGL'], [tiny['latent_linf_mean'], gl['latent_linf_mean']],
                       color=['#c56a5b', '#386a9d'], width=.55)
    axes[0].bar_label(bars, fmt='%.3f', padding=3)
    axes[0].set_title('Same 147 states, same frozen model')
    axes[0].set_ylabel('Mean max-coordinate observation error (unitless)')
    axes[0].set_ylim(0, tiny['latent_linf_mean']*1.2)
    vals = [100*source['pose_xy_euclidean_mae_m'], 100*rerendered['pose_xy_euclidean_mae_m']]
    bars = axes[1].bar(['Supplied features', 'New OpenGL RGB'], vals,
                       color=['#497d67', '#c56a5b'], width=.55)
    axes[1].bar_label(bars, fmt='%.2f cm', padding=3)
    axes[1].set_title('Same 128 source poses, same target Cx')
    axes[1].set_ylabel('Mean position error via inverse C (cm; diagnostic only)')
    axes[1].set_ylim(0, max(vals)*1.2)
    fig.suptitle('Frozen perception test: no PPO evaluation or safety certificate')
    fig.savefig(args.output_dir/'paired_perception_summary.png', dpi=160)
    plt.close(fig)
    print('[Parking frozen perception paired report]')
    print('147-state mean observation error: Tiny=%.6f OpenGL=%.6f' %
          (tiny['latent_linf_mean'], gl['latent_linf_mean']))
    print('128 EXACT same poses: source position MAE=%.3f cm -> new RGB=%.3f cm' % tuple(vals))
    print('old empirical bound coverage: %.2f%% -> %.2f%%' %
          (100*source['empirical_bound_coverage'], 100*rerendered['empirical_bound_coverage']))
    print('No PPO performance or SBC certificate. Adapt perception before training controller.')
    print('report:', args.output_dir/'summary.json')


if __name__ == '__main__': main()
