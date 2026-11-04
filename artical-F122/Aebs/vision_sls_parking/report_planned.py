"""Package a measured planning/feedback/residual development candidate."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from .core import OBSTACLES, GOAL_LOW, GOAL_HIGH, STATE_LOW, STATE_HIGH
from .train_ppo import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feedback-eval', required=True); p.add_argument('--residual-eval', required=True)
    p.add_argument('--legacy-eval', required=True); p.add_argument('--output-dir', required=True)
    p.add_argument('--fresh-feedback'); p.add_argument('--fresh-residual')
    p.add_argument('--test-report', required=True)
    a = p.parse_args(); out = Path(a.output_dir)
    if out.exists():
        p.error('new package directory required')
    directories = {k:Path(v) for k,v in vars(a).items() if k not in ('output_dir','test_report') and v}
    summaries = {}; sources = {}
    for name, root in directories.items():
        summary = json.loads((root/'summary.json').read_text())
        audit = json.loads((root/'audit.json').read_text())
        if not audit['passed'] or not summary['projection_unchanged']:
            raise ValueError('evaluation audit must pass')
        summaries[name] = dict(path=str(root), results=summary['results'], audit=audit)
        for filename in ('config.json', 'summary.json', 'audit.json'):
            sources[str(root/filename)] = sha(root/filename)
    cfg = json.loads((directories['residual_eval']/'config.json').read_text())
    for key in ('references', 'projection', 'residual_policy'):
        hash_key = 'residual_sha256' if key=='residual_policy' else key+'_sha256'
        if sha(cfg[key]) != cfg[hash_key]:
            raise ValueError('changed model or reference')
    out.mkdir(parents=True)
    for key, target in [('references','references.npz'),('projection','projection.pt'),('residual_policy','ppo.zip')]:
        shutil.copy2(cfg[key], out/target)
    projection_cfg = json.loads((Path(cfg['projection']).parent/'config.json').read_text())
    package_cfg = dict(status='development_candidate_not_certified', world_domain=projection_cfg.get('world_domain'),
                       reference_sha256=sha(out/'references.npz'), projection_sha256=sha(out/'projection.pt'),
                       ppo_sha256=sha(out/'ppo.zip'), dino_sha256=cfg['dino_sha256'], actor_input_dimension=13,
                       continuous_verifier_dimension=8, discrete_verifier_state='reference index and time',
                       runtime_inputs='delayed visual measurement, goal, executed actions; catalogue prior',
                       safety_certificate=False, cp_trajectory_coverage_established=False)
    (out/'config.json').write_text(json.dumps(package_cfg, indent=2)+'\n')
    for key, name in [('references','reference_design.json'),('residual_policy','policy_training.json')]:
        shutil.copy2(Path(cfg[key]).parent/'config.json', out/name)
    shutil.copy2(Path(cfg['residual_policy']).parent/'metrics.json', out/'training_metrics.json')
    snapshot = out/'source_snapshot'; snapshot.mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path, snapshot/path.name)
    manifest = {path.name:sha(path) for path in snapshot.iterdir()}
    (out/'source_snapshot_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    test_report = json.loads(Path(a.test_report).read_text())
    shutil.copy2(a.test_report, out/'test_summary.json')
    report = dict(config=package_cfg, comparisons=summaries, source_artifacts_sha256=sources,
                  tests=test_report, safety_certificate=False,
                  limitation='finite sampled point-center simulations, zero process noise; not a general safety probability')
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True, layout='constrained')
    variants = [('Legacy PPO','legacy_eval','legacy'),('Plan + feedback','feedback_eval','feedback'),
                ('Plan + feedback + PPO','residual_eval','residual')]
    for column,(title,directory,mode) in enumerate(variants):
        for row,jitter in enumerate([0., .04]):
            key = mode+'_jitter_'+str(jitter)
            stats = summaries[directory]['results'][key]['image']
            with np.load(directories[directory]/(key+'.npz')) as data:
                states = data['states']
            ax = axes[row, column]
            for x,y,r in OBSTACLES:
                ax.add_patch(Circle((x,y),r,facecolor='#e0e4e8',edgecolor='#4a5568'))
            for j,path in enumerate(states):
                color = '#167d8d' if stats['per_scenario'][j]['success'] else '#cc6633'
                ax.plot(path[:,0],path[:,1],color=color,alpha=.75,linewidth=1.2)
                ax.scatter(*path[0,:2],color=color,s=9);ax.scatter(*path[-1,:2],color=color,s=16,marker='x')
            goals = np.unique(np.column_stack([GOAL_LOW[:,:2],GOAL_HIGH[:,:2]]),axis=0)
            for xl,yl,xh,yh in goals:
                ax.add_patch(Rectangle((xl,yl),xh-xl,yh-yl,fill=False,linestyle='--',edgecolor='#333333'))
            ax.set(xlim=(STATE_LOW[0]-.05,STATE_HIGH[0]+.05),ylim=(STATE_LOW[1]-.05,STATE_HIGH[1]+.05),
                   aspect='equal',xlabel='x (m)',ylabel='y (m)',
                   title='%s\n%s: %d/15; collision %d, bounds %d' %
                   (title,'Original' if jitter==0 else 'Perturbed start',stats['success_count'],
                    stats['collision_count'],stats['out_of_bounds_count']))
            ax.grid(alpha=.15)
    fig.suptitle('Actual rendered images / frozen DINO + projection / H=30 / zero process noise',fontsize=12)
    fig.savefig(out/'trajectory_comparison.png',dpi=180);plt.close(fig)
    root = str(out)
    command = ('CUDA_VISIBLE_DEVICES=0 xvfb-run -a /opt/miniconda3/envs/vt/bin/python '
               '-m Aebs.vision_sls_parking.evaluate_planned_images --references '+root+'/references.npz '
               '--projection '+root+'/projection.pt --residual-policy '+root+'/ppo.zip '
               '--modes feedback residual --output-dir results/parking_planned_candidate_recheck')
    (out/'README.md').write_text('# 规划、视觉反馈与 PPO 修正候选\n\n'
        '冻结参考库＋延迟观测器＋时变 LQR＋有界 PPO 修正。具体对照结果见 summary.json。'
        '控制器使用首帧图像与目标选参考，不使用运行时真实状态；参考库基于原 15 个开发起点。\n\n'
        '**尚无 SBC 安全证书，也没有轨迹级 CP 覆盖保证。** 原 20 维历史策略证书不适用于此模型。'
        'source_snapshot 保存本次实现；projection.pt 的原适配域记录在 config.json。\n\n'
        '在 GPU 服务器 artical-F122 目录运行，输出目录需为新目录：\n\n```bash\n'+command+'\n```\n',encoding='utf-8')
    print(out, 'packaged; no safety certificate', flush=True)


if __name__ == '__main__':
    main()
