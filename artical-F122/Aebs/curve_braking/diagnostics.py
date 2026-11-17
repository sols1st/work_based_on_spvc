"""Descriptive holdout ablations after freezing the three primary results."""
import argparse
from pathlib import Path
import numpy as np
import torch
from . import plant
from .evaluate import collect,save
from .vision import Encoder
from .models import Proxy,Actor

class StraightSteering(torch.nn.Module):
    def __init__(self,actor):
        super().__init__();self.actor=actor
    def forward(self,*args):
        action=self.actor(*args);action[:,1]=0
        return action

def summarize(xs,k):
    lat,gap=plant.margins(xs,k[:,None])
    good=(lat>0).all(1)&(gap>0).all(1);parked=xs[:,-1,3]==0;last=xs[:,-1,2]
    return dict(n=len(xs),body_margin_positive=int(good.sum()),stopped=int(parked.sum()),
        target_success=int((good&parked&(last>=plant.TARGET[0])&(last<=plant.TARGET[1])).sum()),
        mean_abs_lateral_m=float(np.abs(xs[:,:,0]).mean()),max_abs_lateral_m=float(np.abs(xs[:,:,0]).max()))

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();r=a.root
    torch.set_num_threads(3)
    data=np.load(r/'evaluation/neural/holdout.npz');x0=data['states'][:,0];k=data['kappas']
    res={'image_feedback':summarize(data['states'],k)}
    for label,brake in [('coast',0.),('constant_brake',2.)]:
        x=x0.copy();xs=[x]
        for _ in range(plant.HORIZON):
            x=plant.step(x,k,np.broadcast_to([-brake,0.],(len(x),2)));xs.append(x)
        res[label]=summarize(np.stack(xs,1),k)
    load=lambda name:torch.load(r/name,weights_only=True,map_location='cpu')
    actor=Actor();actor.load_state_dict(load('actor.pt'))
    pca=load('pca.pt');m=Proxy('neural',pca['basis'].shape[1]);m.load_state_dict(load('neural.pt'))
    enc=Encoder('../external/dinov2','../external/dinov2_weights/dinov2_vits14_pretrain.pth','cuda:0')
    no_steer=collect(100,202611170512,enc,pca,m,StraightSteering(actor))
    assert np.array_equal(no_steer['states'][:,0],x0)
    res['zero_steering_command_image_brake']=summarize(no_steer['states'],k)
    save(r/'evaluation/baselines.json',res)
if __name__=='__main__':main()
