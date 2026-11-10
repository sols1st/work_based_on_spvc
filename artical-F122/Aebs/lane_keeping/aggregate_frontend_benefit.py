"""Aggregate 5 paired fixed-C vs joint-C lane runs (new common external seeds)."""
import argparse, json
from pathlib import Path
import numpy as np
from .compare_frontend_benefit import metrics, cert, load


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); root = a.root; out = a.output
    if out.exists(): raise SystemExit('output exists')
    out.mkdir(parents=True)
    pairs = [('20261109', root/'fixed_C', root/'joint_C')]
    train_pairs = [('20261109',
                    root.parent/'lane_vision_sls_fixed_C_20261109_ablation_v1',
                    root.parent/'lane_vision_sls_joint_C_20261109_v1')]
    for seed in ['20261111','20261112','20261113','20261114']:
        pairs.append((seed, root/f'cp_seed_{seed}_fixed', root/f'cp_seed_{seed}_joint'))
        train_pairs.append((seed, root/f'train_seed_{seed}_fixed', root/f'train_seed_{seed}_joint'))
    train_meta={seed:(json.loads(fd.joinpath('report.json').read_text()),
                       json.loads(jd.joinpath('report.json').read_text()))
                for seed,fd,jd in train_pairs}
    rows=[]; all_f=[]; all_j=[]
    for seed, fd, jd in pairs:
        f=metrics(fd); j=metrics(jd); fc=cert(fd); jc=cert(jd)
        fcal,fhold,_=load(fd); jcal,jhold,_=load(jd)
        paired = dict(
            calibration_initial=bool(np.allclose(fcal['states'][:,0], jcal['states'][:,0], rtol=0, atol=0)),
            holdout_initial=bool(np.allclose(fhold['states'][:,0], jhold['states'][:,0], rtol=0, atol=0)),
            kappas=bool(np.array_equal(fcal['kappas'], jcal['kappas'])),
            seeds=bool(np.array_equal(fcal['episode_seeds'], jcal['episode_seeds'])))
        fcov=np.asarray(fhold['scores'])<=f['q']; jcov=np.asarray(jhold['scores'])<=j['q']
        ftr,jtr=train_meta[seed]
        static=dict(fixed_test_mae=ftr['metrics']['test']['decoded_mae_physical'],
                    joint_test_mae=jtr['metrics']['test']['decoded_mae_physical'],
                    fixed_best_epoch=ftr['best_epoch'], joint_best_epoch=jtr['best_epoch'],
                    fixed_C=ftr['C_final'], joint_C=jtr['C_final'],
                    fixed_C_change=ftr['C_change_frobenius'], joint_C_change=jtr['C_change_frobenius'],
                    fixed_condition=ftr['condition'], joint_condition=jtr['condition'])
        row=dict(seed=seed, fixed=f, joint=j, fixed_certificate=fc, joint_certificate=jc,
                 static=static,
                 q_difference=float(j['q']-f['q']), q_ratio=float(j['q']/f['q']) if f['q'] else None,
                 holdout_coverage_difference=float(j['holdout_fraction']-f['holdout_fraction']),
                 paired=paired,
                 paired_coverage=dict(both=int((fcov&jcov).sum()), fixed_only=int((fcov&~jcov).sum()),
                                      joint_only=int((~fcov&jcov).sum()), neither=int((~fcov&~jcov).sum())))
        rows.append(row); all_f.append(f); all_j.append(j)
    def arr(name, rows, which):
        return np.asarray([r[which][name] for r in rows],dtype=float)
    def summarize(which):
        return dict(q_mean=float(arr('q',rows,which).mean()), q_median=float(np.median(arr('q',rows,which))),
                    q_std=float(arr('q',rows,which).std(ddof=1)),
                    coverage_mean=float(arr('holdout_fraction',rows,which).mean()),
                    coverage_values=[int(r[which]['holdout_covered']) for r in rows],
                    unsafe_total=int(sum(r[which]['unsafe_holdout'] for r in rows)),
                    cert_passed=sum(1 for r in rows if r[which]['certificate_status']=='passed_numerical'),
                    action_mean=float(arr('mean_abs_action',rows,which).mean()),
                    action_change_mean=float(arr('mean_abs_action_change',rows,which).mean()),
                    test_e_mae_mean=float(np.mean([r['static'][which+'_test_mae'][0] for r in rows])),
                    test_theta_mae_mean=float(np.mean([r['static'][which+'_test_mae'][1] for r in rows])),
                    C_change_mean=float(np.mean([r['static'][which+'_C_change'] for r in rows])),
                    condition_mean=float(np.mean([r['static'][which+'_condition'] for r in rows])))
    diffs=np.asarray([r['q_difference'] for r in rows]); ratios=np.asarray([r['q_ratio'] for r in rows])
    covdiff=np.asarray([r['holdout_coverage_difference'] for r in rows])
    summary=dict(format='lane_frontend_benefit_5seed_v1', n_pairs=len(rows),
        external_seeds=dict(calibration=202611100101, holdout=202611100102),
        fixed=summarize('fixed'), joint=summarize('joint'),
        q_difference_mean=float(diffs.mean()), q_difference_median=float(np.median(diffs)),
        q_difference_std=float(diffs.std(ddof=1)), q_ratio_mean=float(ratios.mean()),
        q_ratio_median=float(np.median(ratios)), joint_smaller_pairs=int((diffs<0).sum()),
        coverage_difference_mean=float(covdiff.mean()), coverage_difference_values=covdiff.tolist(),
        paired_coverage=dict(both=int(sum(r['paired_coverage']['both'] for r in rows)),
                             fixed_only=int(sum(r['paired_coverage']['fixed_only'] for r in rows)),
                             joint_only=int(sum(r['paired_coverage']['joint_only'] for r in rows)),
                             neither=int(sum(r['paired_coverage']['neither'] for r in rows))),
        pairs=rows,
        scope='5 training-seed pairs, one common external condition set per split; closed-loop trajectories generated independently per model. Pilot-level paired evidence, not a powered population claim.',
        notes=['q is normalized lane state; physical halfwidths are 0.85q m and 0.30q rad.',
               'Both certificate passes are numerical audits, not outward-rounded floating-point proofs.',
               'C learning is small; q differences may be sampling/training noise; do not claim architecture causality from one pair.'])
    (out/'summary_all.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    # figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True)
    x=np.arange(len(rows)); qf=arr('q',rows,'fixed'); qj=arr('q',rows,'joint')
    axs[0].plot(x,qf,'o-',label='fixed C'); axs[0].plot(x,qj,'s-',label='joint C')
    axs[0].set_xticks(x,[r['seed'] for r in rows],rotation=30); axs[0].set_ylabel('q (normalized)')
    axs[0].set_title('CP radius q'); axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)
    cf=arr('holdout_fraction',rows,'fixed')*100; cj=arr('holdout_fraction',rows,'joint')*100
    axs[1].plot(x,cf,'o-',label='fixed C'); axs[1].plot(x,cj,'s-',label='joint C')
    axs[1].axhline(95,color='gray',ls='--',lw=1); axs[1].set_xticks(x,[r['seed'] for r in rows],rotation=30)
    axs[1].set_ylabel('holdout coverage (%)'); axs[1].set_ylim(80,101); axs[1].set_title('100-trajectory holdout coverage')
    axs[1].legend(fontsize=8); axs[1].grid(alpha=.3)
    axs[2].bar(x-.18,ratios,width=.36,label='joint/fixed q ratio',color='#4c72b0')
    axs[2].axhline(1,color='gray',ls='--',lw=1); axs[2].set_xticks(x,[r['seed'] for r in rows],rotation=30)
    axs[2].set_ylim(.8,1.2); axs[2].set_title('q ratio (<1: joint smaller)'); axs[2].grid(alpha=.3,axis='y')
    fig.suptitle('Lane front-end benefit: 5 paired training seeds, common external conditions')
    fig.savefig(out/'q_coverage.png',dpi=180)
    # markdown
    L=['# 车道前端收益证明：5 对训练种子配对结果','',
       '固定 C 与联合 C 每对共享训练数据、warm start、预算和外部条件种子；但闭环轨迹由各自模型重新生成。','',
       '| 训练种子 | 固定 C q | 联合 C q | q 比 | 固定 C 留出 | 联合 C 留出 | 固定 C 证书 | 联合 C 证书 |',
       '|---|---:|---:|---:|---:|---:|---|---|']
    for r in rows:
        L.append(f"| {r['seed']} | {r['fixed']['q']:.6f} | {r['joint']['q']:.6f} | {r['q_ratio']:.4f} | {r['fixed']['holdout_covered']}/100 | {r['joint']['holdout_covered']}/100 | {r['fixed']['certificate_status']} | {r['joint']['certificate_status']} |")
    L += ['', '| 汇总 | 固定 C | 联合 C |', '|---|---:|---:|',
          f"| q 均值 | {summary['fixed']['q_mean']:.6f} | {summary['joint']['q_mean']:.6f} |",
          f"| q 中位数 | {summary['fixed']['q_median']:.6f} | {summary['joint']['q_median']:.6f} |",
          f"| 留出覆盖均值 | {summary['fixed']['coverage_mean']:.3f} | {summary['joint']['coverage_mean']:.3f} |",
          f"| SBC 通过 | {summary['fixed']['cert_passed']}/5 | {summary['joint']['cert_passed']}/5 |",
          f"| 平均 |action| | {summary['fixed']['action_mean']:.6f} | {summary['joint']['action_mean']:.6f} |",
          f"| 平均 |Δaction| | {summary['fixed']['action_change_mean']:.6f} | {summary['joint']['action_change_mean']:.6f} |",
          f"| 静态 test e MAE | {summary['fixed']['test_e_mae_mean']:.6f} m | {summary['joint']['test_e_mae_mean']:.6f} m |",
          f"| 静态 test θ MAE | {summary['fixed']['test_theta_mae_mean']:.6f} rad | {summary['joint']['test_theta_mae_mean']:.6f} rad |",
          f"| C 变化 / 条件数 | {summary['fixed']['C_change_mean']:.6f} / {summary['fixed']['condition_mean']:.6f} | {summary['joint']['C_change_mean']:.6f} / {summary['joint']['condition_mean']:.6f} |",
          '', f"q 差（joint−fixed）均值 **{summary['q_difference_mean']:.6f}**，中位数 {summary['q_difference_median']:.6f}；联合 C 更小的对数 **{summary['joint_smaller_pairs']}/5**。", '',
          f"留出覆盖差（joint−fixed）均值 **{summary['coverage_difference_mean']:.3f}**；配对覆盖总计 both={summary['paired_coverage']['both']}, fixed-only={summary['paired_coverage']['fixed_only']}, joint-only={summary['paired_coverage']['joint_only']}, neither={summary['paired_coverage']['neither']}。", '',
          '**解释边界：**5 对是训练种子层面的小样本先导；q 差异很小且受采样噪声影响；联合 C 的变化本身很小，不能把 q 差直接归因于“C 学到了更好的可观测性”。两种证书都是条件数值审计，不是实车/形式化证明。', '',
          '![q and coverage](q_coverage.png)']
    (out/'report.md').write_text('\n'.join(L)+'\n')
    print(json.dumps({k:summary[k] for k in ['q_difference_mean','q_ratio_mean','joint_smaller_pairs','coverage_difference_mean']},indent=2))

if __name__=='__main__': main()
