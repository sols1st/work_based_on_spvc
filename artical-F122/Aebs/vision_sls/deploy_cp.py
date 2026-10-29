"""Load the frozen vision/PPO + perception contract + discrete NCBF-QP."""
import json
from pathlib import Path
import torch
from torch import nn
from .deploy import VisionSLSController
from .cp_ncbf import MonotoneNCBF
from .cp_filter import CPNCBFFilter
from .run import sha256


class VisionSLSCPController(nn.Module):
    def __init__(self,artifact_dir,certificate_dir,device='cpu',dino_repo=None,dino_weights=None):
        super().__init__()
        root=Path(certificate_dir); source=Path(artifact_dir)
        saved=torch.load(root/'ncbf.pt',map_location='cpu',weights_only=False)
        contract=json.loads((root/'perception_contract.json').read_text())
        if saved['perception_contract_sha256']!=sha256(root/'perception_contract.json') or saved['source_policy_sha256']!=sha256(source/'policy.pt') or contract['source_observation_sha256']!=sha256(source/'observation.pt'):
            raise ValueError('CP/NCBF/vision/actor identity mismatch')
        original=VisionSLSController(source,device,dino_repo,dino_weights)
        self.encoder=original.encoder
        barrier=MonotoneNCBF(saved['knots'],saved['delta'])
        barrier.load_state_dict(saved['state_dict'])
        self.policy=CPNCBFFilter(original.policy,barrier,contract['coefficients'],contract['calibration']['q_hat'])
        self.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self,images,measured_speed):
        return self.policy.forward_observation(self.encoder(images),measured_speed)
