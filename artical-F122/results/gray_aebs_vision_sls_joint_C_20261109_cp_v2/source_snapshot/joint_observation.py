"""AEBS joint visual/linear observation matching for an explicit static-image law.

Images in these libraries label distance, not speed. Training repeats each
image with specified auxiliary velocities under I=I(d,appearance). These are
synthetic state augmentations, NOT observed velocity labels or new images.
The learned y approximately equals C_d*d/S+C_v*v. Deployment decodes with
measured speed; true distance is never an input to the image branch.
"""
import torch
from torch import nn

from .model import ObservationProjection


class JointCLSProjection(nn.Module):
    def __init__(self, feature_dim=384, hidden_dim=128):
        super().__init__()
        base=ObservationProjection(feature_dim,hidden_dim)
        self.net=base.net
        self.register_buffer('feature_mean',base.feature_mean)
        self.register_buffer('feature_std',base.feature_std)
        self.raw_C=nn.Parameter(torch.tensor([[1.,.2]]))

    def matrix(self):
        return self.raw_C/self.raw_C.norm(dim=1,keepdim=True).clamp_min(1e-12)

    def forward(self,features):
        return self.net((features-self.feature_mean)/self.feature_std)

    def nominal(self,state):
        return state@self.matrix().T

    def decode(self,latent,measured_speed):
        C=self.matrix()
        return (latent-C[0,1]*measured_speed)/C[0,0]


def load_joint(path):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('format')!='aebs_joint_C_static_image_v1':raise ValueError('unexpected format')
    model=JointCLSProjection(saved['feature_dim'],saved['hidden_dim'])
    model.load_state_dict(saved['state_dict']);C=model.matrix().detach()
    if not torch.equal(C,saved['C']) or not torch.isfinite(C).all() or abs(C[0,0])<.2:
        raise ValueError('invalid or changed observation matrix')
    return model.eval().requires_grad_(False),saved


class JointLatentPolicy(nn.Module):
    """Adapter around a frozen semantic AEBS actor, taking latent y and speed."""
    def __init__(self,semantic_policy,C):
        super().__init__()
        if C.shape!=(1,2) or not torch.isfinite(C).all() or abs(C[0,0])<.2:
            raise ValueError('ill-conditioned AEBS observation interface')
        self.semantic_policy=semantic_policy
        self.register_buffer('C',C.detach().clone())

    def forward_observation(self,latent,measured_speed):
        distance_observation=(latent-self.C[0,1]*measured_speed)/self.C[0,0]
        return self.semantic_policy.forward_observation(distance_observation,measured_speed)
