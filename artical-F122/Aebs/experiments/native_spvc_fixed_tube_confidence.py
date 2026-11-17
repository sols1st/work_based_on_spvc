"""Independent exact binomial check on the already verified AEBS deployment tube."""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta

from Aebs.experiments.native_spvc_camera_cross import (
    PACKAGE, RELEASE, RGB_GRAY, Models, collect, meters, read, tensor, write,
)
from Aebs.vision_sls.finalize_robust_aebs import constant_radius, inclusion_evidence
from Aebs.vision_sls.run import sha256


def lower(k, n, alpha):
    return 0.0 if k == 0 else float(beta.ppf(alpha, k, n-k+1))


def upper(k, n, alpha):
    return 1.0 if k == n else float(beta.ppf(1-alpha, k+1, n-k))


@torch.inference_mode()
def run(bank, out):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    inputs = [bank/'estimates.npz', bank/'protocol.json', bank/'groups.json',
              Path(__file__), Path(__file__).with_name('native_spvc_camera_cross.py'),
              PACKAGE/'sbc.pt', PACKAGE/'policy.pt', PACKAGE/'observation.pt', PACKAGE/'observations.npz',
              RELEASE/'development_certificate.json', RELEASE/'development_contract.json',
              Path('Aebs/system/outcomes.py'), RGB_GRAY]
    inputs_hashes = {str(f):sha256(f) for f in inputs}
    frozen = read(bank/'protocol.json')
    for f, h in frozen['inputs'].items():
        if sha256(Path(f)) != h:
            raise ValueError('frozen deployment inputs changed: ' + f)
    certificate = read(RELEASE/'development_certificate.json')
    contract = read(RELEASE/'development_contract.json')
    if certificate['source_contract_sha256'] != sha256(RELEASE/'development_contract.json'):
        raise ValueError('tube/certificate binding')
    inclusion_evidence(certificate,contract,contract,PACKAGE/'policy.pt',PACKAGE/'sbc.pt',PACKAGE/'observation.pt')
    threshold = constant_radius(contract)
    if threshold != .45:
        raise ValueError('fixed development tube changed')
    n, seed, batch = 2000, 202611170521, 200
    methods = ('rgb_native','rgb_dino')
    write(out/'protocol.json',dict(inputs=inputs_hashes, seed=seed, n=n, batch=batch,
        methods=methods, fixed_threshold_m=threshold, max_steps=400, stop='the existing five AEBS episode outcomes',
        law='fresh iid uniform initial d[15,16] and v[2.5,3], iid uniform appearance among the 12 fixed release pairs per trajectory; all images in the fixed 24k library',
        primary='per-trajectory maximum absolute physical-distance error <= 0.45m for the entire stopped episode up to 400 steps; no threshold fitting or trial selection',
        model='fixed common robust actor and its certified tube; frozen original gray estimator or frozen RGB DINO',
        intervals='per-arm one-sided 95% exact binomial lower; additional simultaneous one-sided contrast: DINO lower at 97.5%, native upper at 97.5% by Bonferroni',
        claim='marginal over the declared episode draw for this fixed library and numerical model; distinct from calibration-marginal split conformal coverage'))
    models = Models()
    data = {}
    with np.load(bank/'estimates.npz') as cache, np.load(PACKAGE/'observations.npz') as lib, np.load(RGB_GRAY) as raw:
        groups = sorted(set(zip(lib['weather'].tolist(), lib['color'].tolist())))
        index = dict(zip(groups,range(len(groups))))
        gid = np.array([index[k] for k in zip(lib['weather'],lib['color'])])
        native = meters(np.concatenate([models.estimate_native(chunk) for chunk in
            tensor(raw['images']).split(256)]),models.original_scale)
        dino = meters(lib['observation'],models.robust.distance_scale_m)
        if (len(groups)!=12 or len(gid)!=24000 or read(bank/'groups.json')!=[list(g) for g in groups]
                or not np.array_equal(gid,cache['appearance']) or not np.array_equal(lib['filename'],cache['files'])
                or not np.array_equal(raw['files'],cache['files']) or not np.array_equal(cache['distance_m'],lib['distance_m'])
                or not np.array_equal(native,cache['native_m']) or not np.array_equal(dino,cache['dino_m'])):
            raise ValueError('frozen 24k replay cache identity mismatch')
        for name,y in zip(methods,(native,dino)):
            data[name] = [(lib['distance_m'][gid==i],y[gid==i]) for i in range(12)]
    rng = np.random.default_rng(seed)
    initial = rng.uniform([15,2.5],[16,3],size=(n,2))
    appearance = rng.integers(12,size=n)
    counts, arms = {}, {}
    for name in methods:
        rows = []
        for begin in range(0,n,batch):
            subset = slice(begin,begin+batch)
            rows.append(collect(models,data,name,'robust_actor',initial[subset],appearance[subset]))
        r = {k: np.concatenate([row[k] for row in rows]) for k in rows[0]}
        if (not np.array_equal(initial,r['initial']) or not np.array_equal(appearance,r['appearance'])
                or not np.isfinite(r['max_truth_m']).all() or np.any(r['steps']<=0)):
            raise ValueError('invalid paired rollouts')
        inside = r['max_truth_m'] <= threshold
        np.savez_compressed(out/(name+'.npz'),in_tube=inside,**r)
        k = counts[name] = int(inside.sum())
        outcomes = dict(zip(*[v.tolist() for v in np.unique(r['outcomes'],return_counts=True)]))
        arms[name] = dict(in_tube=k, n=n, empirical=k/n, one_sided_95_lower=lower(k,n,.05), outcomes=outcomes)
    dino_l = lower(counts['rgb_dino'],n,.025)
    native_u = upper(counts['rgb_native'],n,.025)
    if any(sha256(f)!=inputs_hashes[str(f)] for f in inputs):
        raise ValueError('inputs changed during sampling')
    report = dict(threshold_m=threshold, arms=arms, actor_sha256=sha256(PACKAGE/'policy.pt'),
        barrier_sha256=sha256(PACKAGE/'sbc.pt'), parent_cells=certificate['checked_cells'],
        simultaneous_95=dict(dino_lower=dino_l,native_upper=native_u,
            in_tube_probability_difference_lower=dino_l-native_u, method='one-sided Clopper-Pearson at alpha=.025 each; Bonferroni'))
    write(out/'report.json',report)
    print(report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    run(a.bank,a.output_dir)
