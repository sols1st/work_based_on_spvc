"""Independent sign/coverage checks and global-wrapper ablation on the frozen crane boxes."""
from fractions import Fraction
import argparse
import math

import numpy as np

from ..conformal.trajectory_boxes import cuts, widths
from ..curve_braking.evaluate import digest, save
from ..experiments.analyze_crane_fixed_contracts import exact_geometry
from ..overhead_crane import cp_boxes, plant
from .collect_boxes import BASE, OUT, sealed
from .train import read


def global_wrapper(shape):
    # All fits extend flat beyond their outermost knots or clipped fit range.
    high = [Fraction(max(1, math.ceil(max(edges, default=0)) + 1)) for edges in cuts(shape)]
    half = cp_boxes.mathematical_widths(shape,[Fraction(0)]*2,high)
    outward = np.nextafter(half.astype(float),np.inf).tolist()
    return dict(name='fixed_global_wrapper',xcuts=[],ycuts=[],centers=[],halfwidths=[[outward]])


def motor_witness(contract,gains,scale):
    """A concrete initial corner and allowed symmetric error maximizing both positive commands."""
    if contract['shape']['name']!='fixed_axis':
        raise ValueError('Use the registered symmetric fixed-axis box')
    k=[Fraction(str(v)) for v in gains]
    b=[Fraction(str(v)) for v in plant.INITIAL]
    initial=[-int(np.sign(v))*scale*w for v,w in zip(k,b)]
    q=Fraction.from_float(contract['q'])
    width=[q*Fraction.from_float(v) for v in contract['shape']['halfwidths'][0][0][0]]
    err=[-int(np.sign(k[0]))*v for v in width[:2]]+[-int(np.sign(k[2]))*v for v in width[2:]]
    command=[-sum(j*x for j,x in zip(k,initial))-k[0]*err[i]-k[2]*err[i+2]/2 for i in range(2)]
    intercept=sum(command)-2*scale*sum(abs(j)*x for j,x in zip(k,b))
    critical=(Fraction(str(plant.TOTAL_ACCEL))-intercept)/(2*sum(abs(j)*x for j,x in zip(k,b)))
    return dict(scale=float(scale),initial=[[float(x) for x in initial]]*2,error=[float(v) for v in err],
                unclipped_l1=float(sum(command)),unclipped_l1_exact=str(sum(command)),
                motor_limit=plant.TOTAL_ACCEL,exceeds=bool(sum(command)>Fraction(str(plant.TOTAL_ACCEL))),
                necessary_scale_upper_for_no_clipping=float(critical),upper_exact=str(critical))


def run():
    seal=sealed()
    comparison=read(OUT/'comparison.json')
    gains=read(BASE/'protocol.json')['gains']
    records={}
    for name,arm in comparison['arms'].items():
        root=OUT/name
        rows=read(root/'boxes.json')['boxes']
        counts={}
        max_rank_delta=0.
        for split in ('calibration','holdout'):
            with np.load(root/(split+'.npz')) as data:
                f=cp_boxes.features(data['states'][:,:-1])
                err=data['errors']
                for key,row in rows.items():
                    lo,hi=widths(row['shape'],f)
                    scores=np.maximum((-err/lo).max((1,2)),(err/hi).max((1,2)))
                    if split=='calibration':
                        delta=abs(np.sort(scores)[189]-row['q'])
                        max_rank_delta=max(max_rank_delta,delta)
                        if delta>4*np.spacing(row['q']):
                            raise ArithmeticError('Independent calibration rank differs beyond roundoff')
                    elif int(np.sum(scores<=row['q']))!=arm['boxes'][key]['covered']:
                        raise ArithmeticError('Independent blind count differs')
        uniform=rows['fixed_uniform']['q']
        ref=exact_geometry(gains,uniform,Fraction(1))
        ours=arm['boxes']['fixed_uniform']['audit']
        if not (0<=ours['nominal']['worst_ratio']-ref['upper']<3e-5 and
                0<=ref['scale_supremum']-ours['certification_metric']<.01002):
            raise ArithmeticError('Fixed-uniform reference exact-support disagreement')
        for key,row in rows.items():
            local=arm['boxes'][key]['audit']
            global_bound=cp_boxes.audit(global_wrapper(row['shape']),row['q'])
            if global_bound['certification_metric']>local['certification_metric']+1e-12:
                raise ArithmeticError('Global wrapper unexpectedly certifies more')
            counts[key]=dict(local_scale=local['certification_metric'],
                             global_scale=global_bound['certification_metric'],
                             global_nominal_ratio=global_bound['nominal']['worst_ratio'])
        example=motor_witness(rows['fixed_axis'],gains,Fraction(3,2))
        records[name]=dict(boxes=counts,fixed_axis_at_scale_1_5=example,rank_max_abs_difference=float(max_rank_delta),exact_uniform_nominal_ratio=ref['upper'],
                           exact_uniform_scale_supremum=ref['scale_supremum'])
    save(OUT/'global_wrapper_check.json',dict(arms=records,seal=seal,comparison=digest(OUT/'comparison.json'),
        interpretation='Same calibrated set; global wrapper is a state-independent overapproximation, without recalibration.'))


def confirmation():
    from .confirm_boxes import OUT as directory, METHODS, SPLITS, FAMILIES, SEED
    from .train import ProxyObservation,arm_path
    from .check import replay
    import torch

    p=read(directory/'protocol.json')
    seal=read(directory/'all_boxes_frozen.json')
    recorded=read(directory/'summary.json')
    witnesses={}
    for fam in FAMILIES:
        name=f'{SEED}_{fam}'
        arm=directory/name
        c=read(arm/'boxes.json')
        if seal[name]!=digest(arm/'boxes.json') or c['calibration']!=digest(arm/'calibration.npz') or recorded['arms'][name]['holdout']!=digest(arm/'holdout.npz'):
            raise ValueError('Asset binding mismatch')
        old=arm_path(BASE,fam,SEED)
        model=ProxyObservation(fam).eval()
        model.load_state_dict(torch.load(old/'frontend.pt',map_location='cpu',weights_only=True))
        for split,(n,seed) in SPLITS.items():
            with np.load(arm/(split+'.npz')) as data:
                replay(data,model,n,seed,p['gains'],True)
                f=cp_boxes.features(data['states'][:,:-1])
                for k in METHODS:
                    row=c['boxes'][k]
                    w=widths(row['shape'],f)
                    e=data['errors']
                    score=np.maximum((-e/w[0]).max((1,2)),(e/w[1]).max((1,2)))
                    if split=='calibration':
                        if np.sort(score)[474]!=row['q']:
                            raise ArithmeticError('Unexpected independent calibration radius')
                        if row['audit']['certification_metric']!=cp_boxes.audit(row['shape'],row['q'])['certification_metric']:
                            raise ArithmeticError('Scale changed')
                    elif int((score<=row['q']).sum())!=recorded['arms'][name]['boxes'][k]['covered']:
                        raise ArithmeticError('Blind coverage changed')
        witnesses[name]=motor_witness(c['boxes']['fixed_axis'],p['gains'],
                                       Fraction(str(c['boxes']['state_bins']['audit']['certification_metric'])))
    save(directory/'independent_check.json',dict(boundary_at_verified_state_bins_scale=witnesses,
         complete_transitions=(499+500)*2*40,checked_thresholds=6,summary=digest(directory/'summary.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirmation',action='store_true')
    args=parser.parse_args()
    if args.confirmation:
        confirmation()
    else:
        run()
