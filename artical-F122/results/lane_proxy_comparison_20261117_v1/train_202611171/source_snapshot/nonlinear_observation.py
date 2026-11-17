"""Smooth, explicitly invertible neural observation proxy for lane keeping.

h(x)=coupling(Cx); calibration uses decode(y)-x. The raw observation set is
{h(x+delta)-h(x): ||delta||inf<=q}, not a constant latent box.
"""
import torch
from torch import nn
from .joint_observation import JointPatchProjection, require_invertible
from ..dino_latent.backbone import FrozenDinoV2
from .learned_sbc import sha256


class NonlinearPatchProjection(JointPatchProjection):
    def __init__(self, feature_dim=384, hidden=64, grid=16):
        super().__init__(feature_dim, hidden, grid)
        self.f = nn.Sequential(nn.Linear(1,16),nn.Tanh(),nn.Linear(16,1),nn.Tanh())
        self.g = nn.Sequential(nn.Linear(1,16),nn.Tanh(),nn.Linear(16,1),nn.Tanh())
        for net in (self.f,self.g):
            nn.init.zeros_(net[2].weight); nn.init.zeros_(net[2].bias)

    def nominal(self, state):
        z=state@self.matrix().T
        b=z[:,1:]+.25*self.f(z[:,:1])
        return torch.cat([z[:,:1]+.25*self.g(b),b],1)

    def decode(self, latent):
        a=latent[:,:1]-.25*self.g(latent[:,1:])
        z=torch.cat([a,latent[:,1:]-.25*self.f(a)],1)
        return torch.linalg.solve(self.matrix(),z.T).T

    @torch.no_grad()
    def constrain(self):
        # Frobenius product upper-bounds the derivative of each scalar network.
        for net in (self.f,self.g):
            bound=net[0].weight.norm()*net[2].weight.norm()
            if bound>1: net[2].weight.div_(bound)


def load_nonlinear(path,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('format')!='lane_nonlinear_observation_v1':
        raise ValueError('nonlinear observation checkpoint required')
    model=NonlinearPatchProjection(saved['feature_dim'],saved['hidden'],saved['grid'])
    model.load_state_dict(saved['state_dict'])
    require_invertible(model.matrix().detach(),saved['max_condition'])
    if not torch.equal(model.matrix().detach(),saved['C']): raise ValueError('C binding mismatch')
    for net in (model.f,model.g):
        if float(net[0].weight.norm()*net[2].weight.norm())>1.000001:
            raise ValueError('coupling derivative bound violated')
    return model.to(device).eval().requires_grad_(False),saved


class NonlinearImageObserver(nn.Module):
    def __init__(self,path,device,dino_repo,dino_weights):
        super().__init__()
        self.projection,saved=load_nonlinear(path,device)
        if sha256(dino_weights)!=saved['dino_weights_sha256']: raise ValueError('DINO mismatch')
        self.backbone=FrozenDinoV2(dino_repo,dino_weights,device,saved['dino_model'])
        self.device=device

    @torch.no_grad()
    def forward_with_latent(self,images):
        images=images.to(self.device)
        if images.dtype==torch.uint8: images=images.float()/255.
        if images.ndim==4 and images.shape[-1]==3: images=images.permute(0,3,1,2)
        if not torch.isfinite(images).all() or images.min()<0 or images.max()>1:
            raise ValueError('invalid RGB image input')
        patches=self.backbone.backbone.forward_features(self.backbone.preprocess(images))['x_norm_patchtokens']
        latent=self.projection(patches)
        return self.projection.decode(latent),latent

    def forward(self,images): return self.forward_with_latent(images)[0]
