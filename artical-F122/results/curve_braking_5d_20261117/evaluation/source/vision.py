"""Perspective view of a circular lane and a stationary obstacle rear face."""
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F
from ..dino_latent.backbone import FrozenDinoV2

def render(state,k,size=144):
    e,psi,d=state[:3]
    def project(station,lateral,height):
        t=k*np.asarray(station)
        x=(1-np.cos(t))/k+np.cos(t)*lateral-e
        y=np.sin(t)/k-np.sin(t)*lateral
        u=np.cos(psi)*x-np.sin(psi)*y
        depth=np.sin(psi)*x+np.cos(psi)*y
        h=np.asarray(height)-1.10
        dep=.995*depth-.10*h
        return np.column_stack((size*(.5+.87*u/dep),size*(.5-.87*(.10*depth+.995*h)/dep)))
    im=Image.new('RGB',(size,size),(167,199,220));dr=ImageDraw.Draw(im)
    dr.rectangle((0,size*.42,size,size),fill=(97,119,73))
    def ribbon(lo,hi,start,end,color,n=44):
        s=np.linspace(start,end,n)
        pts=np.concatenate((project(s,lo,0),project(s,hi,0)[::-1]))
        dr.polygon([tuple(p) for p in pts],fill=color)
    ribbon(-1.75,1.75,.85,36,(73,75,79))
    for edge in [-1.75,1.65]: ribbon(edge,edge+.10,.85,36,(241,237,219))
    for st in [2.,6.,10.,14.,18.,22.,26.,30.]:ribbon(-.036,.036,st,st+1.4,(223,213,153),4)
    # Rear panel at station d, with known physical width and height.
    def panel(left,right,bottom,top,color):
        lateral=np.array([left,right,right,left]);z=np.array([bottom,bottom,top,top])
        dr.polygon([tuple(p) for p in project(np.full(4,d),lateral,z)],fill=color)
    panel(-.76,.76,.09,1.32,(138,48,41));panel(-.58,.58,.83,1.21,(57,83,102))
    panel(-.64,-.43,.42,.56,(249,143,80));panel(.43,.64,.42,.56,(249,143,80))
    panel(-.20,.20,.32,.45,(231,226,202))
    return np.asarray(im)

class Encoder:
    def __init__(self,repo,weights,device):
        self.model=FrozenDinoV2(repo,weights,device);self.device=device
    @torch.inference_mode()
    def __call__(self,images):
        x=torch.tensor(np.asarray(images),device=self.device).float().permute(0,3,1,2)/255.
        tok=self.model.backbone.forward_features(self.model.preprocess(x))['x_norm_patchtokens']
        grid=tok.transpose(1,2).reshape(len(x),384,16,16)
        return F.adaptive_avg_pool2d(grid,(4,4)).flatten(1).cpu()
