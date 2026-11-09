"""Frozen patch DINO -> spatial feature head -> three pose coordinates."""
import torch
from torch import nn
from Aebs.dino_latent.backbone import FrozenDinoV2


class PoseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial=nn.Sequential(nn.Conv2d(384,64,1),nn.GELU(),nn.Conv2d(64,32,3,padding=1),nn.GELU())
        self.attention=nn.Conv2d(32,2,1)
        self.readout=nn.Sequential(nn.Linear(100,128),nn.Tanh(),nn.Linear(128,3))
        axis=torch.linspace(-1,1,16);yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        self.register_buffer('coords',torch.stack([xx.flatten(),yy.flatten()],1))
        self.register_buffer('mean',torch.zeros(384));self.register_buffer('std',torch.ones(384))

    def forward(self,features):
        features=(features.float()-self.mean)/self.std
        x=self.spatial(features.reshape(-1,16,16,384).permute(0,3,1,2))
        weights=self.attention(x).flatten(2).softmax(-1)
        xy=(weights@self.coords).flatten(1)
        local=(weights@x.flatten(2).transpose(1,2)).flatten(1)
        glob=x.mean((2,3))
        return self.readout(torch.cat([xy,local,glob],1))


class ImageObserver(nn.Module):
    def __init__(self,checkpoint,dino_repo,dino_weights,device='cpu'):
        super().__init__()
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['format']!='mobile_robot_dino_pose_v1':raise ValueError('wrong observer format')
        self.head=PoseHead();self.head.load_state_dict(saved['state_dict']);self.head.to(device)
        self.device=torch.device(device)
        self.backbone=FrozenDinoV2(dino_repo,dino_weights,device)
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self,images):
        x=images.to(self.device).float()/255.
        if x.shape[-1]==3:x=x.permute(0,3,1,2)
        features=self.backbone.backbone.forward_features(self.backbone.preprocess(x))['x_norm_patchtokens']
        return self.head(features)
