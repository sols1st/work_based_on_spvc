"""Plot all original parking tasks; no success-based task selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from .core import OBSTACLES, GOAL_LOW, GOAL_HIGH, dynamics, constraints, ACTION_HIGH


def audit_rollouts(metrics, data):
    """Independent recomputation from saved states/actions, no new model test."""
    audit = {}
    for name in ('proxy', 'image'):
        if name+'_states' not in data:
            continue
        states, actions = data[name+'_states'], data[name+'_actions']
        if states.shape != (15, 31, 4) or actions.shape != (15, 30, 2):
            raise ValueError('all 15 original tasks and all 30 steps required')
        gap, successes, collisions, bounds, arrival_xy = 0., 0, 0, 0, 0
        for i in range(15):
            flags = [constraints(s, i) for s in states[i]]
            collision = any(f['collision'] for f in flags)
            outside = any(not f['in_bounds'] for f in flags)
            success = not collision and not outside and flags[-1]['in_goal']
            row = metrics[name]['per_scenario'][i]
            if (success, collision, outside) != (row['success'], row['collision'], row['out_of_bounds']):
                raise ValueError('saved rollout/metric disagreement')
            for t in range(30):
                predicted = dynamics(states[i, t], actions[i, t])
                gap = max(gap, float(np.max(np.abs(predicted-states[i, t+1]))))
            successes += success
            collisions += collision
            bounds += outside
            arrival_xy += not collision and flags[-1]['in_goal_xy']
        counts = (successes, collisions, bounds)
        reported = tuple(metrics[name][k] for k in
                         ('success_count', 'collision_count', 'out_of_bounds_count'))
        excess = float(np.maximum(np.abs(actions)-ACTION_HIGH, 0).max())
        if counts != reported or gap > 1e-10 or excess > 1e-10:
            raise ValueError('rollout replay/count/action-bound audit failed')
        audit[name] = dict(replayed_transitions=450, max_dynamics_replay_error=gap,
                           max_action_bound_excess=excess, counts_match=True,
                           upstream_xy_arrival_without_collision_count=arrival_xy)
    result = dict(status='saved_rollout_replay_and_counts_pass', paths=audit)
    if 'baseline' in metrics:
        paired = {}
        for name in audit:
            old, new = metrics['baseline'][name]['per_scenario'], metrics[name]['per_scenario']
            if [r['scenario'] for r in old] != [r['scenario'] for r in new]:
                raise ValueError('baseline and candidate tasks do not match')
            old_bad = {r['scenario'] for r in old if r['collision'] or r['out_of_bounds']}
            new_bad = {r['scenario'] for r in new if r['collision'] or r['out_of_bounds']}
            paired[name] = dict(new_unsafe_task_ids=sorted(new_bad-old_bad),
                                removed_unsafe_task_ids=sorted(old_bad-new_bad),
                                old_unsafe_task_ids=sorted(old_bad), new_unsafe_task_ids_all=sorted(new_bad))
        result['baseline_paired_outcomes'] = paired
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-dir', required=True)
    args = p.parse_args()
    out = Path(args.result_dir)
    m = json.loads((out/'metrics.json').read_text())
    data = np.load(out/'rollouts.npz')
    audit = audit_rollouts(m, data)
    (out/'rollout_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    fig, axes = plt.subplots(3, 5, figsize=(17, 12.5))
    for i, ax in enumerate(axes.flat):
        for x, y, r in OBSTACLES:
            ax.add_patch(Circle((x, y), r, color='.65', alpha=.6))
        lo, hi = GOAL_LOW[i, :2], GOAL_HIGH[i, :2]
        ax.add_patch(Rectangle(lo, *(hi-lo), color='green', alpha=.25))
        ax.add_patch(Rectangle((-3.5, -2.), 4., 3.9, fill=False,
                               edgecolor='.4', linestyle='--', lw=.8))
        labels, drawn = [], []
        for name, color in [('proxy', '#0072B2'), ('image', '#D55E00')]:
            if name+'_states' not in data:
                continue
            s = data[name+'_states'][i]
            drawn.append(s)
            ax.plot(s[:, 0], s[:, 1], '-o', color=color, ms=2, lw=1.4,
                    label='Cx proxy' if name == 'proxy' else 'Actual image')
            ax.scatter(*s[-1, :2], color=color, marker='x', s=35)
            row = m[name]['per_scenario'][i]
            labels.append('%s:%s' % (name, 'OK' if row['success'] else 'FAIL'))
        ax.set_title('Task %02d  %s' % (i, ' / '.join(labels)), fontsize=9)
        ss = np.concatenate(drawn)
        # Never hide an out-of-domain trajectory by clipping the plot to task bounds.
        ax.set_xlim(min(-3.6, ss[:, 0].min()-.1), max(.6, ss[:, 0].max()+.1))
        ax.set_ylim(min(-2.1, ss[:, 1].min()-.1), max(2., ss[:, 1].max()+.1))
        ax.set_aspect('equal')
        ax.grid(alpha=.2)
        if i % 5:
            ax.tick_params(labelleft=False)
    axes[0, 0].legend(fontsize=8, loc='lower right')
    fig.suptitle('Same frozen PPO, original 15 tasks, one-frame delay, 30 steps\n'
                 'Grey: obstacle; green: goal; dashed: position bounds; x: final. Development only.')
    fig.supxlabel('World x [m]')
    fig.supylabel('World y [m]')
    fig.subplots_adjust(left=.055, right=.995, bottom=.065, top=.90,
                        hspace=.30, wspace=.10)
    fig.savefig(out/'paired_rollouts.png', dpi=150)
    plt.close(fig)
    print(out/'paired_rollouts.png')


if __name__ == '__main__':
    main()
