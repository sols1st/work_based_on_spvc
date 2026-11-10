"""Stage-1 paired fixed-C vs joint-C lane benefit report.

Reads two closed-loop CP output directories produced with the SAME external
seeds. Reports q, holdout coverage, SBC certificate status and paired initial
condition checks. It does not turn a single pair into a population claim.
"""
import argparse, json, math
from pathlib import Path
import numpy as np
from scipy.stats import beta


def interval(k, n):
    lo = 0.0 if k == 0 else float(beta.ppf(.025, k, n-k+1))
    hi = 1.0 if k == n else float(beta.ppf(.975, k+1, n-k))
    return [lo, hi]


def load(d):
    cal = dict(np.load(d/'calibration.npz', allow_pickle=False))
    hold = dict(np.load(d/'holdout.npz', allow_pickle=False))
    rep = json.loads((d/'report.json').read_text())
    return cal, hold, rep


def metrics(d):
    cal, hold, rep = load(d)
    c = rep['contract']
    q = c['radius']
    hs = np.asarray(hold['scores'], dtype=float)
    cov = int((hs <= q).sum())
    # normalized observation error, max over time/coordinates and over episodes
    e = np.abs(np.asarray(hold['errors'], dtype=float)).max(2)
    a = np.asarray(hold['actions'], dtype=float)
    return dict(
        q=q, rank=c['rank'], n_cal=c['n'], cal_covered=int(c['coverage']['covered']),
        cal_fraction=float(c['coverage']['fraction']),
        holdout_n=len(hs), holdout_covered=cov, holdout_fraction=cov/len(hs),
        holdout_interval=interval(cov, len(hs)),
        holdout_score_mean=float(hs.mean()), holdout_score_max=float(hs.max()),
        holdout_frame_error_mean=float(e.mean()), holdout_frame_error_max=float(e.max()),
        unsafe_holdout=int(np.asarray(hold['unsafe']).sum()),
        mean_abs_action=float(np.abs(a).mean()),
        mean_abs_action_change=float(np.abs(np.diff(a, axis=1)).mean()),
        certificate_status=rep.get('certificate_status'),
        certificate_sha256=rep.get('certificate_sha256'),
        report=rep,
    )


def cert(d):
    p = d/'verification/certificate.json'
    if not p.exists(): return None
    x = json.loads(p.read_text())
    checks=[]
    checked=witnesses=unresolved=0
    for c in x.get('checks', []):
        hist=c.get('history',[])
        last=hist[-1] if hist else {}
        checked_now=int(last.get('outside',0)+last.get('plateau',0)+last.get('decreasing',0))
        checked+=checked_now
        witnesses+=int(c.get('actual_witnesses',0))
        unresolved+=int(c.get('unresolved',0))
        checks.append(dict(kappa=c.get('kappa'), passed=c.get('passed'),
                           witnesses=c.get('actual_witnesses'),
                           unresolved=c.get('unresolved'), checked=checked_now))
    return dict(passed=bool(x.get('passed')), checked_cells=checked,
                actual_witnesses=witnesses, unresolved_cells=unresolved,
                initial=x.get('initial'), unsafe=x.get('unsafe'), checks=checks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fixed', type=Path, required=True)
    p.add_argument('--joint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output
    if out.exists(): raise SystemExit('output exists')
    out.mkdir(parents=True)
    f = metrics(a.fixed); j = metrics(a.joint)
    fc = cert(a.fixed); jc = cert(a.joint)
    fcal, fhold, _ = load(a.fixed); jcal, jhold, _ = load(a.joint)
    paired = {}
    for split, x, y in [('calibration', fcal, jcal), ('holdout', fhold, jhold)]:
        paired[split] = dict(
            initial_states_equal=bool(np.allclose(x['states'][:, 0], y['states'][:, 0], rtol=0, atol=0)),
            kappas_equal=bool(np.array_equal(x['kappas'], y['kappas'])),
            episode_seeds_equal=bool(np.array_equal(x['episode_seeds'], y['episode_seeds'])),
        )
    # paired trajectory coverage at each method's own frozen q
    fcov = np.asarray(fhold['scores']) <= f['q']
    jcov = np.asarray(jhold['scores']) <= j['q']
    paired['holdout_coverage_table'] = dict(
        both_covered=int((fcov & jcov).sum()),
        fixed_only=int((fcov & ~jcov).sum()),
        joint_only=int((~fcov & jcov).sum()),
        neither=int((~fcov & ~jcov).sum()),
    )
    summary = dict(
        format='lane_frontend_benefit_stage1_v1',
        fixed=f, joint=j, fixed_certificate=fc, joint_certificate=jc,
        paired_external_conditions=paired,
        q_difference=float(j['q'] - f['q']),
        q_ratio=float(j['q'] / f['q']) if f['q'] else None,
        holdout_coverage_difference=float(j['holdout_fraction'] - f['holdout_fraction']),
        scope='One paired checkpoint pair with new common external seeds; pilot evidence, not a 5-seed population claim. q units are normalized lane coordinates; physical halfwidths are 0.85*q m and 0.30*q rad.',
        note='Each method uses its own closed-loop trajectories; only the initial conditions/kappas/seeds are shared. Do not interpret q difference as proof of architecture superiority.',
    )
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    lines = [
        '# 车道前端收益证明：阶段一配对先导', '',
        '固定 C 与联合 C 使用同一组新外部条件种子（校准 202611100101，留出 202611100102），但各自生成闭环轨迹。', '',
        '| 指标 | 固定 C | 联合 C |', '|---|---:|---:|',
        f"| q（归一化） | {f['q']:.10f} | {j['q']:.10f} |",
        f"| 校准覆盖 | {f['cal_covered']}/{f['n_cal']} | {j['cal_covered']}/{j['n_cal']} |",
        f"| 旧数据回归/新留出覆盖 | {f['holdout_covered']}/{f['holdout_n']} | {j['holdout_covered']}/{j['holdout_n']} |",
        f"| 留出覆盖率 95% 区间 | [{f['holdout_interval'][0]:.3f}, {f['holdout_interval'][1]:.3f}] | [{j['holdout_interval'][0]:.3f}, {j['holdout_interval'][1]:.3f}] |",
        f"| 留出 unsafe 轨迹 | {f['unsafe_holdout']} | {j['unsafe_holdout']} |",
        f"| 留出平均帧误差 | {f['holdout_frame_error_mean']:.6f} | {j['holdout_frame_error_mean']:.6f} |",
        f"| 留出最大帧误差 | {f['holdout_frame_error_max']:.6f} | {j['holdout_frame_error_max']:.6f} |",
        f"| 平均 |action| | {f['mean_abs_action']:.6f} | {j['mean_abs_action']:.6f} |",
        f"| 平均 |Δaction| | {f['mean_abs_action_change']:.6f} | {j['mean_abs_action_change']:.6f} |",
        f"| SBC 状态 | {f['certificate_status']} | {j['certificate_status']} |",
        f"| q 差 / 比 | — | {summary['q_difference']:.10f} / {summary['q_ratio']:.6f} |",
        f"| 覆盖率差 | — | {summary['holdout_coverage_difference']:.3f} |", '',
        f"配对初始条件一致：cal={paired['calibration']}，holdout={paired['holdout']}。", '',
        f"配对留出覆盖：both={paired['holdout_coverage_table']['both_covered']}, fixed-only={paired['holdout_coverage_table']['fixed_only']}, joint-only={paired['holdout_coverage_table']['joint_only']}, neither={paired['holdout_coverage_table']['neither']}。", '',
        '证书：', f"- 固定 C: {json.dumps(fc, ensure_ascii=False)}", f"- 联合 C: {json.dumps(jc, ensure_ascii=False)}", '',
        '**边界：**这是一对检查点、一次新种子先导，不足以宣称总体显著优势；q 是归一化坐标，物理半宽是 `0.85q m` 和 `0.30q rad`。不能把 q 差直接归因于 C，也不能用留出覆盖 95/100 之外的读数替代条件 SBC。',
    ]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k: summary[k] for k in ['q_difference','q_ratio','holdout_coverage_difference']}, indent=2))


if __name__ == '__main__': main()
