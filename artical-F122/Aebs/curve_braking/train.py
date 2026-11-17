import argparse,copy,json,itertools
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import plant
from .models import Actor,Barrier,Proxy

def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def train_front(root):
    data=np.load(root/'static.npz');f=torch.load(root/'features.pt',map_location='cpu',weights_only=True)
    tr,va,te=[torch.from_numpy(np.flatnonzero(data['split']==s)) for s in ['train','val','test']]
    x=torch.tensor(plant.normalize(data['states']),dtype=torch.float32)
    mean=f[tr].mean(0);torch.manual_seed(1711)
    _,_,basis=torch.pca_lowrank(f[tr]-mean,q=192,center=False,niter=4)
    z=(f-mean)@basis;std=z[tr].std(0).clamp_min(.001);z=z/std
    torch.save(dict(mean=mean,basis=basis,std=std),root/'pca.pt')
    design=torch.cat([z[tr],torch.ones(len(tr),1)],1)
    ridge=torch.linalg.solve(design.T@design+2.*torch.eye(design.shape[1]),design.T@x[tr])
    for kind in ['linear','affine','neural']:
        torch.manual_seed(117);model=Proxy(kind,z.shape[1])
        with torch.no_grad():
            model.head.weight.copy_(ridge[:-1].T);model.head.bias.copy_(ridge[-1])
        initial=copy.deepcopy(model.state_dict());opt=torch.optim.Adam(model.parameters(),lr=.00015)
        rng=torch.Generator().manual_seed(117);best=None;hist=[]
        for ep in range(35):
            for ids in tr[torch.randperm(len(tr),generator=rng)].split(96):
                y=model.head(z[ids]);loss=(y-model.nominal(x[ids])).norm(dim=1).mean()+.1*(model.decode(y)-x[ids]).square().mean()
                opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():
                mse=float((model.decode(model.head(z[va]))-x[va]).square().mean());cond=float(torch.linalg.cond(model.matrix()));hist.append(mse)
                if cond<10 and (best is None or mse<best[0]):best=(mse,ep+1,copy.deepcopy(model.state_dict()))
        model.load_state_dict(best[2]);torch.save(model.state_dict(),root/f'{kind}.pt')
        with torch.no_grad():
            latent=model.head(z[te]);err=model.decode(latent)-x[te]
            metrics=dict(best_epoch=best[1],validation_mse=best[0],history=hist,
                test_mae_physical=(err.abs().mean(0)*torch.tensor(plant.SCALE)).tolist(),
                test_normalized_mse=float(err.square().mean()),test_raw_latent_l2=float((latent-model.nominal(x[te])).norm(dim=1).mean()),
                cond=float(torch.linalg.cond(model.matrix())),inverse_max=float((model.decode(model.nominal(x[te]))-x[te]).abs().max()),
                max_shift_from_Cx=float((model.nominal(x[te])-x[te]@model.matrix().T).abs().max()))
        write(root/f'{kind}_static.json',metrics)

def train_control(root):
    torch.manual_seed(1175);actor=Actor();opt=torch.optim.Adam(actor.parameters(),lr=.003)
    perturb=torch.tensor(list(itertools.product([-.12,.12],repeat=3))+[(0.,0.,0.)])
    for _ in range(1000):
        n=128
        s=torch.rand(n,5)*torch.tensor([.96,.28,3.6,2.4,.14])+torch.tensor([-.48,-.14,2.6,.05,-.07])
        e,t,d,v,delta=s.T;k=torch.where(torch.rand(n)<.5,-.02,.02)
        obs=(s[:,:3]-torch.tensor(plant.CENTER).float())/torch.tensor(plant.SCALE).float()
        # Expert trades braking and steering against remaining distance and current steering state.
        brake=(v*v/(2*(d-4.).clamp_min(.10))).clamp(1.20,3.5)
        target=((torch.atan(2.6*(k-.65*e-1.1*t))-delta)*1.25).clamp(-.16,.16)
        noisy=(obs[:,None]+perturb).reshape(-1,3);repeat=lambda a:a.repeat_interleave(len(perturb))
        pred=actor(noisy,repeat(v),repeat(delta),repeat(k)).reshape(n,-1,2)
        deviation=((pred[:,:,0]+brake[:,None])/2.3).square()+((pred[:,:,1]-target[:,None])/.32).square()
        loss=.6*deviation.amax(1).mean()+.4*deviation[:,-1].mean()
        opt.zero_grad();loss.backward();opt.step()
    torch.save(actor.state_dict(),root/'actor.pt')
    # Minimize conservatism of the barrier features subject to global robust
    # dominance inequalities, evaluated analytically again on the saved weights.
    b=Barrier().double();opt=torch.optim.Adam(b.parameters(),lr=.03)
    for _ in range(450):
        ae,at,beta,ad=b.coefficients()
        geom=1/(1-.02*.65)
        slack=torch.stack([ae-np.sin(.19),at-np.tan(.07)/2.6-.02*geom-.16*beta,ad-geom])
        loss=.1*ae+at+.01*beta+.05*ad+2000*F.relu(.001-slack).square().sum()
        opt.zero_grad();loss.backward();opt.step()
    torch.save(b.state_dict(),root/'barrier.pt')
    write(root/'training.json',dict(actor_seed=1175,actor_updates=1000,actor_train_radius=.12,actor_error_points=9,
        barrier_updates=450,coefficients=b.coefficients().detach().tolist(),actor_loss='worst of 8 corners and center, plus nominal imitation',
        barrier_type='structured trainable ReLU stopping-reserve barrier with sufficient-condition loss'))

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();torch.set_num_threads(3)
    train_front(a.root);train_control(a.root)
if __name__=='__main__':main()
