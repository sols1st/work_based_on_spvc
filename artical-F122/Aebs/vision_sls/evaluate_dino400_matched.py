"""Same 400-image budget, same new actor: DINO-400 observation vs cGAN matched.

DINO observation is the existing 10/29 400-image DINO projection; its
observation scale is 3.237806 m. The new actor expects distance/S with
S=3.18272066. This is an A-matched data-budget comparison, not a retrain.
"""
import argparse, json, math
from pathlib import Path
import numpy as np, torch
from torch import nn
from Combined_network.model import CombinedPolicyNetwork
from Aebs.vision_sls.model import VisionSLSPolicy
from Aebs.system.outcomes import classify_terminal_outcome, TIMEOUT
S_ACTOR=3.18272066116333
S_OBS=3.2378063201904297

def cp(scores,alpha=.05):
 v=np.sort(np.asarray(scores,float));n=len(v);r=math.ceil((n+1)*(1-alpha));return (float(v[r-1]) if r<=n else float('inf')),r
def interval(k,n):
 from scipy.stats import beta
 return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))]
def run(n,seed,dist,obs,policy,steps):
 rng=np.random.default_rng(seed); init=rng.uniform([15.,2.5],[16.,3.],size=(n,2)); d=init[:,0].copy();v=init[:,1].copy();done=np.zeros(n,bool);out=np.empty(n,dtype=object);out[:]='';steps_done=np.zeros(n,int);scores=np.zeros(n)
 for t in range(steps):
  active=np.flatnonzero(~done)
  if not len(active):break
  idx=np.abs(dist[None,:]-d[active,None]).argmin(1); phys=obs[idx]*S_OBS; obs_actor=phys/S_ACTOR
  with torch.no_grad():
   act=policy.forward_observation(torch.tensor(obs_actor,dtype=torch.float32)[:,None],torch.tensor(v[active],dtype=torch.float32)[:,None]).detach().numpy().reshape(-1)
  act=np.clip(act,-3.,3.);scores[active]=np.maximum(scores[active],np.abs(phys-d[active]));dn=d[active]-v[active]*.05;vn=np.clip(v[active]-act*.05,0.,3.);steps_done[active]+=1
  for j,i in enumerate(active):
   o=classify_terminal_outcome(dn[j],vn[j]);
   if o is not None:out[i]=o;done[i]=True
  d[active]=dn;v[active]=vn
 out[~done]=TIMEOUT
 return dict(initial=init,scores=scores.tolist(),outcomes=out.tolist(),steps=steps_done.tolist())
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--calibration',type=int,default=199);p.add_argument('--holdout',type=int,default=100);p.add_argument('--calibration-seed',type=int,default=202611100101);p.add_argument('--holdout-seed',type=int,default=202611100102);p.add_argument('--steps',type=int,default=400);p.add_argument('--new-policy',type=Path,default=Path('results/real_aebs_repair_20261109_release/policy.pt'));p.add_argument('--dino-observations',type=Path,default=Path('results/vision_sls_spvc_sbc_pass_20261029/observations.npz'))
 a=p.parse_args();out=a.output_dir
 if out.exists():raise SystemExit('output exists')
 out.mkdir(parents=True);z=np.load(a.dino_observations);dist=np.asarray(z['distance_m']);obs=np.asarray(z['observation'])
 saved=torch.load(a.new_policy,map_location='cpu',weights_only=False);sizes=[2]+saved['actor_hidden'];layers=[]
 for x,y in zip(sizes[:-1],sizes[1:]):layers += [nn.Linear(x,y),nn.Tanh()]
 actor=CombinedPolicyNetwork(nn.Sequential(*layers),nn.Linear(sizes[-1],1));policy=VisionSLSPolicy(actor,S_ACTOR);policy.load_state_dict(saved['state_dict']);policy.eval();policy.set_state_distance_scale(saved['metadata']['state_distance_scale_m'])
 cal=run(a.calibration,a.calibration_seed,dist,obs,policy,a.steps);hold=run(a.holdout,a.holdout_seed,dist,obs,policy,a.steps);q,rank=cp(cal['scores']);cov=int((np.asarray(hold['scores'])<=q).sum())
 rep=dict(format='aebs_dino400_matched_v1',mode='DINO-400 real image nearest replay -> new actor',data='400-image Downsampled.h5 DINO projection (gray)',actor='new policy.pt',calibration=dict(n=len(cal['scores']),q=q,rank=rank,covered=int((np.asarray(cal['scores'])<=q).sum())),holdout=dict(n=len(hold['scores']),covered=cov,fraction=cov/len(hold['scores']),interval=interval(cov,len(hold['scores'])),unsafe=int(sum(o=='unsafe' for o in hold['outcomes'])),success=int(sum(o=='success' for o in hold['outcomes'])),timeouts=int(sum(o=='timeout' for o in hold['outcomes']))),scope='Same 400-image data budget and same actor as cGAN matched; DINO backbone pretraining not matched.')
 (out/'report.json').write_text(json.dumps(rep,indent=2));np.savez_compressed(out/'calibration.npz',**cal);np.savez_compressed(out/'holdout.npz',**hold);print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
