"""Paired frozen-policy comparison on prospectively registered crane operating boxes."""
import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from ..curve_braking.evaluate import digest, save
from ..lane_keeping.cp import conformal_radius
from ..overhead_crane import certificate, plant
from ..overhead_crane.camera import Camera, Observation
from ..overhead_crane.evaluate import bindings, collect


SCALES = (1., 1.2, 1.3, 1.4, 1.5, 1.6)
N_CAL = 199
N_TEST = 200
ALPHA = .05
JOINT_VISUAL_WEIGHT = .5


def contract_definitions(development_errors):
    # Only past development/pilot records set scales. Fresh split CP sets every q.
    axis_maxima = np.abs(development_errors).max(axis=1)
    widths = np.quantile(axis_maxima, .95, axis=0)
    widths = np.maximum(.01, widths / widths.max())
    return {name: dict(weights=w.tolist(), joint=joint)
            for name, w, joint in [('box', np.ones(4), False), ('per_axis', widths, False),
                                   ('box_and_action', np.ones(4), True),
                                   ('per_axis_and_action', widths, True)]}


def direction_scales(k, weights):
    return abs(k[0]) * weights[:2] + .5 * abs(k[2]) * weights[2:]


def trajectory_scores(errors, k, definition):
    weights = np.array(definition['weights'])
    visual = np.max(np.abs(errors)/weights, axis=(1,2))
    if not definition['joint']:
        return visual
    action = errors[:,:,:2]*k[0] + .5*errors[:,:,2:]*k[2]
    direction = np.max(np.abs(action)/direction_scales(k, weights), axis=(1,2))
    return np.maximum(JOINT_VISUAL_WEIGHT * visual, direction)


def certify(k, q, definition, initial_scale):
    """Use separate signed support bounds for each axis, including shared motor and clearance."""
    widths = [Fraction.from_float(v) for v in definition['weights']]
    kp, ks = abs(Fraction(str(k[0]))), abs(Fraction(str(k[2]))) / 2
    radius = Fraction.from_float(q)
    # Both contracts have command support q*L_i. The joint contract imposes that
    # bound explicitly and permits q/0.5 times the visual widths.
    gamma = [radius*(kp*widths[i]+ks*widths[i+2]) for i in (0,1)]
    planes = []
    for value in gamma:
        # Legacy routine accepts a float radius and uses it only through L*q.
        # Round this equivalent input upwards; do not use it as a visual radius.
        equiv = value/(kp+ks)
        as_float = float(equiv)
        if Fraction.from_float(as_float) < equiv:
            as_float = float(np.nextafter(as_float, np.inf))
        planes.append(certificate.planes(k, as_float, plant.HORIZON, exact=True))
    initial = [Fraction(str(initial_scale))*Fraction(str(x)) for x in plant.INITIAL]
    peaks = {}
    for first, second in zip(*planes):
        name, t, row, offset_x, limit = first
        offset_y = second[3]
        centre = abs(row)@initial
        if name in ('total_accel','payload_diagonal'):
            ratio = (2*centre + offset_x+offset_y)/limit
        else:
            ratio = (centre+max(offset_x,offset_y))/limit
        if name not in peaks or ratio > peaks[name][0]:
            peaks[name] = (ratio,t)
    upper = max(value for value,_ in peaks.values())
    visual_q = q/JOINT_VISUAL_WEIGHT if definition['joint'] else q
    return dict(q=q, visual_halfwidths=(visual_q*np.array(definition['weights'])).tolist(),
                action_halfwidths=[float(v) for v in gamma], upper=float(upper),
                upper_exact=str(upper), passes=bool(upper<1),
                constraints={n:dict(ratio=float(v),step=t) for n,(v,t) in peaks.items()})


def scenario(n, seed, scale):
    initial, noise, lights = plant.scenarios(n, seed)
    return initial*scale, noise, lights


def run(args):
    root, out = args.base, args.output
    out.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(3)
    previous = root/'evaluation'
    old_protocol = json.loads((previous/'protocol.json').read_text())
    k = old_protocol['gains']['image_full']
    frozen = bindings(root,args.repo,args.weights)
    if frozen != old_protocol['bindings']:
        raise RuntimeError('baseline source or frozen assets differ from original run')
    old_cal = previous/'calibration.npz'
    definitions = contract_definitions(np.load(old_cal)['errors'])
    protocol = dict(methods=definitions, k=k, visual_weight=JOINT_VISUAL_WEIGHT,
                    source=digest(Path(__file__)), baseline_protocol=digest(previous/'protocol.json'),
                    development=digest(old_cal), bindings=frozen, alpha=ALPHA,
                    primary='all preregistered scale cells, first report full-domain certificate; no selection on data',
                    score='max over 40 steps and all visual components or signed command projections',
                    n_cal=N_CAL,n_holdout=N_TEST,batch=args.batch,
                    grid=[dict(scale=s,cal_seed=1148701+2*i,hold_seed=1148702+2*i)
                          for i,s in enumerate(SCALES)])
    save(out/'protocol.json',protocol)
    protocol_sha=digest(out/'protocol.json')
    enc=Camera(args.repo,args.weights,args.device)
    model=Observation().to(args.device)
    model.load_state_dict(torch.load(root/'frontend.pt',map_location=args.device,weights_only=True))
    model.eval()
    reports=[]
    for spec in protocol['grid']:
        scale=spec['scale']
        folder=out/f'scale_{scale:.1f}'
        folder.mkdir(exist_ok=False)
        cal=collect(scenario(N_CAL,spec['cal_seed'],scale),k,enc,model,args.batch)
        np.savez_compressed(folder/'calibration.npz',**cal)
        audits={}
        for name,definition in definitions.items():
            scores=trajectory_scores(cal['errors'],k,definition)
            q,rank=conformal_radius(scores,ALPHA)
            audits[name]=certify(k,q,definition,scale)
            audits[name]['rank']=rank
        save(folder/'certificates.json',dict(scale=scale,protocol=protocol_sha,
            calibration=digest(folder/'calibration.npz'),methods=audits))
        cert_sha=digest(folder/'certificates.json')
        held=collect(scenario(N_TEST,spec['hold_seed'],scale),k,enc,model,args.batch)
        np.savez_compressed(folder/'holdout.npz',**held)
        arms={}
        for name,definition in definitions.items():
            scores=trajectory_scores(held['errors'],k,definition)
            arms[name]=dict(**audits[name],covered=int((scores<=audits[name]['q']).sum()))
        if digest(folder/'certificates.json')!=cert_sha:
            raise RuntimeError('frozen certificate changed')
        row=dict(scale=scale,performance=plant.metrics(held['states']),methods=arms,
                 certificate=cert_sha,holdout=digest(folder/'holdout.npz'))
        save(folder/'report.json',row)
        reports.append(row)
        print(scale,[(name,round(v['upper'],4),v['covered']) for name,v in arms.items()],flush=True)
    if bindings(root,args.repo,args.weights)!=frozen or digest(out/'protocol.json')!=protocol_sha:
        raise RuntimeError('frozen experimental inputs changed')
    save(out/'summary.json',dict(protocol=protocol_sha,rows=reports))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('base',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--repo',type=Path,default=Path('../external/dinov2'))
    p.add_argument('--weights',type=Path,default=Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth'))
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--device',default='cuda:0')
    run(p.parse_args())
