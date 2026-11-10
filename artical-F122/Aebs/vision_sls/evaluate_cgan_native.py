"""Native original-SPVC cGAN closed loop vs a matched new-actor front-end.

Native mode: cGAN generator -> state_net -> original PPO (best_model).
Matched mode: same cGAN observation path -> the new AEBS actor (policy.pt),
so the actor is fixed and only the observation front-end differs.
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from cGAN.cGAN_common import Settings
from cGAN.taxi_models_and_data import AebsMLPGenerator
from Combined_network.model import SubNet, CombinedPolicyNetwork
from stable_baselines3 import PPO
from Aebs.vision_sls.model import VisionSLSPolicy
from Aebs.system.outcomes import classify_terminal_outcome, TIMEOUT

S = 3.18272066116333


def cp(scores, alpha=.05):
    v=np.sort(np.asarray(scores,float)); n=len(v); rank=math.ceil((n+1)*(1-alpha))
    if rank>n: return float('inf'), rank
    return float(v[rank-1]), rank


def interval(k,n):
    from scipy.stats import beta
    return [0.0 if k==0 else float(beta.ppf(.025,k,n-k+1)),
            1.0 if k==n else float(beta.ppf(.975,k+1,n-k))]


def run(args, n, seed):
    torch.set_num_threads(4); rng=np.random.default_rng(seed)
    initial=rng.uniform([15.,2.5],[16.,3.],size=(n,2))
    gen=AebsMLPGenerator(4,1); gen.load_state_dict(torch.load(args.cgan,map_location='cpu',weights_only=False)); gen.eval()
    state_net=SubNet([1024,256,64,1]); state_net.load_state_dict(torch.load(args.state_net,map_location='cpu',weights_only=True)); state_net.eval()
    if args.mode=='native':
        ppo=PPO.load(args.native_ppo,device='cpu')
    else:
        saved=torch.load(args.new_policy,map_location='cpu',weights_only=False)
        sizes=[2]+saved['actor_hidden']; layers=[]
        for a,b in zip(sizes[:-1],sizes[1:]): layers += [nn.Linear(a,b),nn.Tanh()]
        actor=CombinedPolicyNetwork(nn.Sequential(*layers),nn.Linear(sizes[-1],1))
        ppo=VisionSLSPolicy(actor,S); ppo.load_state_dict(saved['state_dict']); ppo.eval()
        ppo.set_state_distance_scale(saved['metadata']['state_distance_scale_m'])
    d=initial[:,0].copy(); v=initial[:,1].copy(); done=np.zeros(n,bool)
    outcome=np.empty(n,dtype=object); outcome[:]=''; steps=np.zeros(n,int)
    scores=np.zeros(n); actions=[]; errors=[]; states=[]
    z=torch.zeros(n,4)
    for t in range(args.steps):
        active=np.flatnonzero(~done)
        if not len(active): break
        dn=torch.tensor(d[active]/S,dtype=torch.float32)[:,None]
        with torch.no_grad():
            img=gen(z[active],dn)
            obs_norm=state_net(img.view(len(active),-1)).flatten().numpy()
        obs=obs_norm*S
        if args.mode=='native':
            act=ppo.policy.predict(np.c_[obs_norm,v[active]],deterministic=True)[0].reshape(-1)
        else:
            with torch.no_grad():
                act=ppo.forward_observation(torch.tensor(obs_norm,dtype=torch.float32)[:,None],
                                            torch.tensor(v[active],dtype=torch.float32)[:,None]).detach().numpy().reshape(-1)
        act=np.clip(act,-3.,3.)
        err=np.abs(obs-d[active]); scores[active]=np.maximum(scores[active],err)
        dnnext=d[active]-v[active]*.05; vnnext=np.clip(v[active]-act*.05,0.,3.)
        steps[active]+=1
        for j,i in enumerate(active):
            o=classify_terminal_outcome(dnnext[j],vnnext[j])
            if o is not None: outcome[i]=o; done[i]=True
        d[active]=dnnext; v[active]=vnnext
        actions.append((active.copy(),act.copy())); errors.append((active.copy(),err.copy()))
        states.append((active.copy(),d[active].copy(),v[active].copy()))
    outcome[~done]=TIMEOUT
    return dict(initial=initial,outcomes=outcome.tolist(),steps=steps.tolist(),
                scores=scores.tolist(),unsafe=(outcome=='unsafe'),success=(outcome=='success'),
                mean_frame_error=float(np.mean([e.mean() for _,e in errors])),
                max_frame_error=float(np.max([e.max() for _,e in errors])))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['native','matched'],required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--cgan',type=Path,default=Path('Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth'))
    p.add_argument('--state-net',type=Path,default=Path('Aebs/controller/state_net_trained.pth'))
    p.add_argument('--native-ppo',type=Path,default=Path('Aebs/controller/best_model/best_model.zip'))
    p.add_argument('--new-policy',type=Path,default=Path('results/real_aebs_repair_20261109_release/policy.pt'))
    p.add_argument('--calibration',type=int,default=199); p.add_argument('--holdout',type=int,default=100)
    p.add_argument('--calibration-seed',type=int,default=202611100101); p.add_argument('--holdout-seed',type=int,default=202611100102)
    p.add_argument('--steps',type=int,default=400)
    args=p.parse_args(); out=args.output_dir
    if out.exists(): raise SystemExit('output exists')
    out.mkdir(parents=True)
    cal=run(args,args.calibration,args.calibration_seed); hold=run(args,args.holdout,args.holdout_seed)
    q,rank=cp(cal['scores']); cov=int((np.asarray(hold['scores'])<=q).sum())
    report=dict(format='aebs_cgan_native_comparison_v1',mode=args.mode,
        cgan='Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth',state_net='Aebs/controller/state_net_trained.pth',
        actor=('original PPO best_model' if args.mode=='native' else 'new policy.pt actor, cGAN observation'),
        input='native cGAN generated 32x32 image -> state_net -> normalized distance',
        calibration=dict(n=len(cal['scores']),covered=int((np.asarray(cal['scores'])<=q).sum()),q=q,rank=rank),
        holdout=dict(n=len(hold['scores']),covered=cov,fraction=cov/len(hold['scores']),interval=interval(cov,len(hold['scores'])),
                     unsafe=int(sum(hold['unsafe'])),success=int(sum(hold['success'])),timeouts=int(sum(o=='timeout' for o in hold['outcomes']))),
        cal_behavior=dict(unsafe=int(sum(cal['unsafe'])),success=int(sum(cal['success'])),timeouts=int(sum(o=='timeout' for o in cal['outcomes']))),
        frame_error=dict(cal_mean=cal['mean_frame_error'],cal_max=cal['max_frame_error'],hold_mean=hold['mean_frame_error'],hold_max=hold['max_frame_error']),
        physical_radius_m=q,
        scope='Native/matched cGAN synthetic observation closed loop; generated images, not real camera or live CARLA. CP is trajectory max distance error under deterministic z=0 generation.')
    (out/'report.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/'calibration.npz',scores=cal['scores'],outcomes=cal['outcomes'],steps=cal['steps'],initial=cal['initial'])
    np.savez_compressed(out/'holdout.npz',scores=hold['scores'],outcomes=hold['outcomes'],steps=hold['steps'],initial=hold['initial'])
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
