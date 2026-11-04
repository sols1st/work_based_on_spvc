"""Generate goal-reaching backups safe for every CP-compatible start box."""
import argparse,itertools,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from .core import INITIAL_STATES,ACTION_HIGH,STATE_LOW,STATE_HIGH,GOAL_LOW,GOAL_HIGH,OBSTACLES,nominal_observation,DT
from .planning_feedback import ReferenceBank
from .backup_governor import initial_belief,canonical_action
from .interval_backup import sequence_box
from .train_ppo import sha


def rollout(initial,u):
    x=initial;xs=[x]
    for action in u*torch.tensor(ACTION_HIGH,dtype=u.dtype):
        def ode(q):return torch.stack([q[:,3]*torch.cos(q[:,2]),q[:,3]*torch.sin(q[:,2]),
                                      action[0].expand(len(q)),action[1].expand(len(q))],1)
        k1=ode(x);k2=ode(x+DT*k1/2);k3=ode(x+DT*k2/2);k4=ode(x+DT*k3)
        x=x+DT*(k1+2*k2+2*k3+k4)/6;xs.append(x)
    return torch.stack(xs,1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',required=True);p.add_argument('--error-model',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--bound',type=float,default=.005)
    p.add_argument('--iterations',type=int,default=800)
    p.add_argument('--startup-samples')
    a=p.parse_args();root=Path(a.candidate);out=Path(a.output_dir)
    if out.exists():p.error('new output directory required')
    torch.set_num_threads(1);out.mkdir(parents=True);started=time.monotonic()
    bank=ReferenceBank(root/'references.npz');error=json.loads(Path(a.error_model).read_text())
    keys={}
    if a.startup_samples:
        with np.load(a.startup_samples) as z:
            selected=np.isin(z['splits'],['fit','center'])
            measurements=list(zip(z['groups'][selected],z['observations'][selected]))
    else:
        nominal=np.asarray([nominal_observation(x) for x in INITIAL_STATES])
        archives=['vision_sls_parking_20261103_residual_images_v3/residual_jitter_0.0.npz',
                  'vision_sls_parking_20261103_residual_images_v3/residual_jitter_0.04.npz',
                  'vision_sls_parking_20261103_fresh_residual_17001/residual_jitter_0.04.npz']
        observations=[nominal]
        for archive in archives:
            with np.load(Path('results')/archive) as z:observations.append(z['capture_observations'].reshape(15,31,3)[:,0])
        measurements=[(j,y) for ys in observations for j,y in enumerate(ys)]
    for scenario,y in measurements:
        goal=bank.goals[scenario]
        try:
            hypotheses,box=initial_belief(y,goal,error,a.bound)
        except ValueError:
            if a.startup_samples:continue  # A fit image need not be covered by split CP.
            raise
        if a.startup_samples:
            costs=[np.sum(((y-np.asarray(error['bias'][j])-nominal_observation(INITIAL_STATES[j]))/
                           np.maximum(error['radius'][j],1e-8))**2) for j in hypotheses]
            reference=int(hypotheses[int(np.argmin(costs))])
        else:reference=bank.select(y,goal)
        keys[(reference,tuple(hypotheses))]=box
    actions_out=[];refs=[];masks=[];records=[]
    circles=torch.tensor(OBSTACLES);low=torch.tensor(STATE_LOW);high=torch.tensor(STATE_HIGH)
    for (reference,hypotheses),box in sorted(keys.items()):
        idx=hypotheses.index(reference) if reference in hypotheses else None
        if idx is None:continue
        def certified(actions):
            end=sequence_box(box,actions)
            return end is not None and (end[0][idx]>=GOAL_LOW[reference]-.001).all() and (end[1][idx]<=GOAL_HIGH[reference]+.001).all()
        actions=np.asarray([canonical_action(u) for u in bank.actions[reference]])
        optimized=False;loss_value=None
        if not certified(actions):
            # Centers plus all box vertices; interval verification remains the acceptance test.
            corners=np.array([np.where(bits,box[1],box[0]) for bits in itertools.product([False,True],repeat=4)])
            initials=torch.tensor(np.concatenate([INITIAL_STATES[list(hypotheses)],corners.reshape(-1,4)]))
            goal=torch.tensor(bank.goals[reference]);initial_goal=idx
            def objective(flat):
                u=torch.tensor(flat.reshape(30,2),dtype=torch.float64,requires_grad=True)
                x=rollout(initials,u)
                margin=x.new_tensor([.03,.03,.03,.15])
                violations=torch.relu(low+margin-x)+torch.relu(x-high+margin)
                gaps=torch.linalg.vector_norm(x[:,:,:2,None]-circles[:,:2].T[None,None],dim=2)-circles[:,2]
                value=(15000*(x[initial_goal,-1,:2]-goal).square().sum()+50*x[initial_goal,-1,3].square()
                       +200000*violations.square().sum()/len(x)+200000*torch.relu(.08-gaps).square().sum()/len(x)
                       +.1*u.square().sum()+.5*(u[1:]-u[:-1]).square().sum())
                value.backward();return float(value.detach()),u.grad.numpy().ravel()
            fit=minimize(objective,(actions/ACTION_HIGH).ravel(),jac=True,method='L-BFGS-B',bounds=[(-1,1)]*60,
                         options=dict(maxiter=a.iterations,ftol=1e-11,maxls=40))
            actions=np.asarray([canonical_action(u) for u in fit.x.reshape(30,2)*ACTION_HIGH])
            optimized=True;loss_value=float(fit.fun)
        valid=bool(certified(actions))
        row=dict(reference=reference,hypotheses=[int(h) for h in hypotheses],certified_interval_backup=valid,optimized=optimized,objective=loss_value)
        print(json.dumps(row),flush=True);records.append(row)
        if valid:
            actions_out.append(actions);refs.append(reference);masks.append(np.isin(np.arange(15),hypotheses))
        np.savez_compressed(out/'library.npz',actions=np.asarray(actions_out),references=np.asarray(refs),masks=np.asarray(masks))
        (out/'metrics.json').write_text(json.dumps(dict(records=records,runtime_seconds=time.monotonic()-started),indent=2)+'\n')
    cfg=dict(vars(a),source_reference_sha256=sha(root/'references.npz'),error_model_sha256=sha(a.error_model),
             source='startup fit and deterministic center images only' if a.startup_samples else 'nominal and previously used development initial visual observations',
             accepted_by='outward interval rollout over all retained initial hypotheses and selected goal',
             true_state_runtime_input=False)
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')


if __name__=='__main__':main()
