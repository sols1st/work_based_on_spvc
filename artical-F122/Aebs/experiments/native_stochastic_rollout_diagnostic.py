"""Finite rollout diagnostic for the original cGAN nominal map and additive uniform successor law."""
import argparse
from pathlib import Path
import numpy as np
import torch

from .native_spvc_camera_cross import H5, PPO_FILE, write
from stable_baselines3 import PPO
from .native_trained_actor_camera_cross import PairedActors
from Aebs.vision_sls.run import sha256
import h5py


@torch.inference_mode()
def run(native):
    out=native/"stochastic_rollouts"
    out.mkdir(exist_ok=False)
    ckpt=native/"alternating/best.pt"
    models=PairedActors(ckpt)
    original=PPO.load(PPO_FILE,device="cpu")
    torch.set_num_threads(2)
    with h5py.File(H5) as f:
        labels=f["y_train"][:].flatten()
    dlo,dhi=float(labels.min()),float(labels.max())
    n,h,seed=2000,400,202611170963
    rng=np.random.default_rng(seed)
    initial=rng.uniform([15,2.5],[16,3.],(n,2))
    noise=rng.uniform(-1,1,(h,n,2))*np.array([(dhi-dlo)*.05,.15])
    write(out/"protocol.json",dict(seed=seed,n=n,steps=h,
          policy=sha256(ckpt),script=sha256(Path(__file__)),law="source F(s,z=0), clip nominal velocity then add uniform independent noise",
          stops="first entry into source closed unsafe set, exit of declared state domain, or horizon; no safe label on exit",
          distance_domain_m=[dlo,dhi],speed_domain_mps=[0.,3.],time_step_s=.05))
    report={}
    for actor in ("initial","native_trained"):
        for noisy in (False,True):
            states=initial.copy()
            outcomes=np.full(n,"horizon",dtype="U16")
            steps=np.zeros(n,int)
            for t in range(h):
                ids=np.flatnonzero(outcomes=="horizon")
                if not len(ids):
                    break
                d,v=states[ids].T
                observed=models.generated_m(d)
                if actor=="native_trained":
                    a=models.action(actor,observed,v)
                else:
                    x=np.stack([observed/models.original_scale,v],axis=-1).astype(np.float32)
                    a=original.predict(x,deterministic=True)[0].flatten().clip(-3.,3.)
                d1=d-.05*v
                v1=np.clip(v-.05*a,0.,3.)
                new=np.stack([d1,v1],axis=-1)
                if noisy:
                    new+=noise[t,ids]
                states[ids]=new
                d1,v1=new.T
                unsafe=(d1>=5.)&(d1<=6.)&(v1>=.5)&(v1<=3.)
                outside=(d1<dlo)|(d1>dhi)|(v1<0)|(v1>3)
                outcomes[ids[outside]]="domain_exit"
                outcomes[ids[unsafe]]="unsafe"
                steps[ids]+=1
            name=actor+("_uniform" if noisy else "_nominal")
            np.savez_compressed(out/(name+".npz"),initial=initial,terminal=states,outcomes=outcomes,steps=steps)
            report[name]=dict(zip(*[v.tolist() for v in np.unique(outcomes,return_counts=True)]))
    write(out/"summary.json",report)
    print(report,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native",type=Path)
    args=parser.parse_args()
    run(args.native)
