"""Develop original SBC features using admissible tube counterexamples.

These are training/development trajectories, not conformal calibration samples.
An unsafe witness under the old policy is not reused as evidence for a new one.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from Aebs.VT.utils import MLP
from Aebs.system.env import Aebs
from .deploy import VisionSLSController
from .conformal import envelope
from .cp_ncbf import successors
from .train_sbc_cp import region_samples
from .audit_sbc_cp import region_grid
from .sbc_cp import barrier_bounds
from .run import sha256


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-dir',type=Path,required=True)
    p.add_argument('--perception-contract',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--count',type=int,default=16384)
    p.add_argument('--state-seed',type=int,default=44)
    p.add_argument('--feature-seed',type=int,default=52)
    p.add_argument('--warmup-epochs',type=int,default=0)
    args=p.parse_args(argv)
    if args.output_dir.exists():p.error('choose a new output directory')
    args.output_dir.mkdir(parents=True);out=args.output_dir
    torch.set_num_threads(1);torch.manual_seed(args.feature_seed)
    policy=VisionSLSController(args.artifact_dir).policy
    contract=json.loads(args.perception_contract.read_text())
    if contract['source_policy_sha256']!=sha256(args.artifact_dir/'policy.pt'):raise ValueError('actor identity mismatch')
    _,_,initial=region_samples(args.count,args.state_seed)
    state=initial.clone();active=torch.ones(len(state),dtype=torch.bool)
    bad=torch.zeros_like(active);steps=torch.zeros(len(state),dtype=torch.int64)
    with torch.no_grad():
        for step in range(400):
            ids=torch.where(active)[0]
            if not len(ids):break
            steps[ids]+=1;s=state[ids]
            radius=torch.tensor(envelope(s[:,0],contract['coefficients'],contract['calibration']['q_hat']),dtype=s.dtype)
            obs=(s[:,0,None]+radius[:,None]*torch.linspace(-1,1,9))/policy.distance_scale_m
            actions=policy.forward_observation(obs.reshape(-1,1),s[:,1,None].expand_as(obs).reshape(-1,1)).reshape_as(obs).amin(1)
            nxt,_=successors(s,torch.stack([actions,actions],1))
            fail=(nxt[:,0]<=6)&(nxt[:,1]>.5)
            goal=(nxt[:,0]>=5)&(nxt[:,0]<=6)&(nxt[:,1]<=.5)
            stopped=nxt[:,1]<=1e-6;outside=(nxt[:,0]<5)|(nxt[:,0]>16)
            bad[ids[fail|outside]]=True;active[ids[fail|goal|stopped|outside]]=False
            state[ids]=nxt
    np.savez(out/'predecessors.npz',states=initial.numpy(),unsafe_witness=bad.numpy(),timeout=active.numpy(),terminal=state.numpy(),steps=steps.numpy())
    result=dict(samples=len(state),unsafe_witnesses=int(bad.sum()),timeouts=int(active.sum()),
                source_policy_sha256=sha256(args.artifact_dir/'policy.pt'),source_observation_sha256=sha256(args.artifact_dir/'observation.pt'),
                perception_contract_sha256=sha256(args.perception_contract),state_seed=args.state_seed,feature_seed=args.feature_seed,
                uncertainty_scenarios=9,selector='minimum braking over admissible observation scenarios',
                sample_scope='full live domain, not only initial region; development only',formal_certificate=False)
    (out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    if not args.warmup_epochs:return
    scale=float(Aebs(.05).std1);div=torch.tensor([scale,1.]);good=initial[~bad&~active]
    ulo,uhi=region_grid([5,.5],[6,3],32);unsafe=torch.cat([initial[bad],ulo,uhi]);corner=torch.tensor([[6.,.5]])
    ilo,ihi=region_grid([15,2.5],[16,3],32);init=torch.cat([ilo,ihi])
    torch.manual_seed(args.feature_seed)
    model=MLP([2,16,8,1],activation='tanh',square_output=False)
    with torch.no_grad():
        center=torch.tensor([10.5/scale,1.5]);span=torch.tensor([5.5/scale,1.5])
        model.layers[0].weight.div_(span);model.layers[0].bias.copy_(-model.layers[0].weight@center)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);history=[]
    for epoch in range(args.warmup_epochs):
        g=good[torch.randint(len(good),(512,))];u=unsafe[torch.randint(len(unsafe),(512,))];i=init[torch.randint(len(init),(256,))]
        zg=model(g/div).flatten();zu=model(u/div).flatten();zi=model(i/div).flatten()
        target=5. if epoch<4000 else 25.
        loss=torch.relu(target-zu).mean()+2*torch.relu(zi).mean()+2*torch.relu(target-model(corner/div).flatten()).mean()
        loss=loss+.2*torch.nn.functional.smooth_l1_loss(zg,torch.log(torch.expm1(.1+.02*g[:,0])))
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        if (epoch+1)%2000==0:
            model.square_output=True
            with torch.no_grad():
                il,iu=region_grid([15,2.5],[16,3],48);ul,uu=region_grid([5,.5],[6,3],48)
                _,bi=barrier_bounds(model,il/div,iu/div);bu,_=barrier_bounds(model,ul/div,uu/div)
                row=dict(epoch=epoch+1,initial_upper=float(bi.max()),unsafe_lower=float(bu.min()))
            model.square_output=False;history.append(row);print(json.dumps(row),flush=True)
    torch.save(model.state_dict(),out/'pretrained_sbc.pt')
    (out/'warmup_history.json').write_text(json.dumps(history,indent=2))


if __name__=='__main__':main()
