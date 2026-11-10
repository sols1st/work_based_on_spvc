"""Raw image + measured speed deployment for the separately saved joint-C AEBS.

Loading a checkpoint does not grant a certificate. Optionally require the
new CP contract's exact frontend and semantic-actor bindings at construction.
"""
import json
from pathlib import Path
import torch
from torch import nn
from ..dino_latent.backbone import FrozenDinoV2
from .joint_observation import load_joint,JointLatentPolicy
from .joint_replay_cp import load_semantic_policy
from .run import sha256


class JointImageObservation(nn.Module):
    def __init__(self,checkpoint,device='cpu',dino_repo=None,dino_weights=None):
        super().__init__()
        self.projection,saved=load_joint(checkpoint)
        self.projection=self.projection.to(device)
        weights=Path(dino_weights or saved['dino_weights'])
        if sha256(weights)!=saved['provenance']['dino_weights_sha256']:
            raise ValueError('DINO weights differ from training provenance')
        self.backbone=FrozenDinoV2(dino_repo or saved['dino_repo'],weights,device,
            saved['provenance'].get('dino_model','dinov2_vits14'))
        self.distance_scale_m=saved['distance_scale_m'];self.device=device
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self,images):
        images=images.to(self.device)
        if images.dtype==torch.uint8:images=images.float()/255.
        if images.ndim==4 and images.shape[-1]==3:images=images.permute(0,3,1,2)
        if not torch.isfinite(images).all() or images.min()<0 or images.max()>1:
            raise ValueError('expected finite image values in [0,1]')
        return self.projection(self.backbone(images))


class JointImageController(nn.Module):
    def __init__(self,joint_dir,base_artifact,device='cpu',dino_repo=None,dino_weights=None,contract=None):
        super().__init__()
        root=Path(joint_dir);base=Path(base_artifact)
        if contract is not None:
            saved=json.loads(Path(contract).read_text())
            if (saved['source_observation_sha256']!=sha256(root/'joint_observation.pt') or
                saved['source_policy_sha256']!=sha256(base/'policy.pt')):
                raise ValueError('joint contract/frontend/actor binding mismatch')
        self.encoder=JointImageObservation(root/'joint_observation.pt',device,dino_repo,dino_weights)
        semantic=load_semantic_policy(base).to(device)
        if semantic.distance_scale_m!=self.encoder.distance_scale_m:
            raise ValueError('visual distance scale mismatch')
        self.policy=JointLatentPolicy(semantic,self.encoder.projection.matrix()).to(device)
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self,images,measured_speed):
        latent=self.encoder(images)
        speed=torch.as_tensor(measured_speed,device=latent.device,dtype=latent.dtype).reshape(-1,1)
        if len(speed)!=len(latent) or not torch.isfinite(speed).all():raise ValueError('one finite measured speed per image required')
        return self.policy.forward_observation(latent,speed)
