"""Sufficient conditions for the saved piecewise-linear (ReLU/max) barrier.
Checks a continuum via scalar bounds, not samples or a 5-D grid.
"""
import itertools
import numpy as np
import torch
from . import plant
from .models import Barrier

def verify_checkpoint(path,q):
    model=Barrier().double()
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    ae,at,beta,ad=model.coefficients().detach().numpy()
    # Safe sublevel implies |e|<=E, |psi|<=P, D<d<U.
    E,P,D,U=.65,.19,1.92,6.15
    curvature=max(abs(plant.KAPPAS));station_gain=1/(1-curvature*E)
    yaw_gain=np.tan(plant.DMAX)/plant.L+curvature*station_gain
    rho=np.hypot(plant.HALF_WIDTH,plant.HALF_LENGTH)
    lat,gap=plant.margins(np.array([[E,P,D,0,0]]),curvature)
    corners=np.array(list(itertools.product(*zip(plant.INIT_LO,plant.INIT_HI))))
    # B is coordinatewise increasing in |e|, |psi|, |delta| and v,
    # and max of affine terms in d. The maximum therefore occurs at a corner.
    initial=float(model(torch.tensor(corners)).max().detach())
    spacing=1.2*plant.DT
    knots=model.knots.detach().numpy()
    hinges=np.arange(len(knots))*spacing
    checks=dict(
        finite_radius=bool(np.isfinite(q) and q>=0),
        brake_floor_guard_slack_m=7.50-(U+plant.SCALE[2]*q),
        lateral_step_slack=ae-np.sin(P),
        yaw_and_steering_step_slack=at-(yaw_gain+.16*beta),
        obstacle_progress_step_slack=ad-station_gain,
        forward_progress_cosine=np.cos(P),
        steering_weight=beta,body_lane_slack_m=float(lat[0]),body_station_slack_m=float(gap[0]),
        initial_level_slack=1-initial,
        braking_reserve_support=len(knots)*spacing-plant.VMAX,
        minimum_frenet_denominator=1-curvature*(E+rho))
    valid=checks['finite_radius'] and all(v>1e-9 for k,v in checks.items() if k!='finite_radius')
    valid=bool(valid and np.allclose(knots,hinges,rtol=0,atol=2e-15))
    return dict(passed=valid,method='global sufficient inequalities for structured neural barrier; numerical scalar checks',
                radius=q,coefficients=[ae,at,beta,ad],initial_upper=initial,unsafe_lower_by_separation=20,
                horizon_if_tube_holds='until absorbing stop, including all later sampled states',
                domain='0<=v<=2.6, |delta|<=0.07, kappa in {-0.02,+0.02}',checks=checks,
                outward_rounded=False,continuous_time_collision_claim=False)
