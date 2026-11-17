from fractions import Fraction
import numpy as np
from Aebs.conformal.trajectory_boxes import fit_shapes,widths
from Aebs.overhead_crane import cp_boxes,plant
from Aebs.experiments.analyze_crane_fixed_contracts import exact_geometry


def test_uniform_directional_support_recovers_original_finite_horizon_bound():
    shape=dict(name='fixed_uniform',xcuts=[],ycuts=[],halfwidths=[[[[1.]*4,[1.]*4]]])
    k=cp_boxes.kernels()[0]
    q=.19
    result=cp_boxes.audit(shape,q)
    original=exact_geometry([float(v) for v in k],q,Fraction(1))
    assert original['upper']<=result['nominal']['worst_ratio']<original['upper']+.00003
    assert result['certification_metric']<=original['scale_supremum']<result['certification_metric']+.0101


def test_signed_state_boxes_enclose_all_sampled_command_errors():
    rng=np.random.default_rng(819773)
    x=rng.normal(0,.2,(12,40,2,4))
    f=cp_boxes.features(x)
    e=rng.normal(0,.07,(12,40,4))+np.array([.06,-.1,.09,.11])
    k=cp_boxes.kernels()[0]
    for shape in fit_shapes(f,e).values():
        lo,hi=widths(shape,f)
        command=cp_boxes.command_enclosure(shape,1.,k)
        bins_lo=(np.floor(f*40)).astype(int)
        bins_hi=np.ceil(f*40).astype(int)
        for i in range(12):
            limits=command(*bins_lo[i,0],*bins_hi[i,0])/cp_boxes.UNIT
            for sign in (-1,1):
                err=np.concatenate([np.where(-float(k[0])*sign>0,hi[i,0,:2],-lo[i,0,:2]),
                                    np.where(-float(k[2])*sign>0,hi[i,0,2:],-lo[i,0,2:])])
                projected=-(float(k[0])*err[:2]+.5*float(k[2])*err[2:])
                assert np.all(projected<=limits[0]) and np.all(-projected<=limits[1])
