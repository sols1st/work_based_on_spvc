"""Load the saved, post-SPVC actor and use camera images plus measured speed."""
import hashlib
from pathlib import Path
import torch
from torch import nn
from Combined_network.model import CombinedPolicyNetwork
from .model import ImageObservation, VisionSLSPolicy


class VisionSLSController(nn.Module):
    def __init__(self, artifact_dir, device='cpu', dino_repo=None, dino_weights=None):
        super().__init__()
        root = Path(artifact_dir)
        saved = torch.load(root/'policy.pt', map_location=device, weights_only=False)
        digest = hashlib.sha256((root/'observation.pt').read_bytes()).hexdigest()
        if saved.get('format') != 'vision_sls_policy_v1' or saved['observation_sha256'] != digest:
            raise ValueError('actor and observation checkpoints do not match')
        self.encoder = ImageObservation(root/'observation.pt', device, dino_repo, dino_weights)
        sizes = [2] + saved['actor_hidden']
        layers = []
        for nin, nout in zip(sizes[:-1],sizes[1:]):
            layers.extend([nn.Linear(nin,nout), nn.Tanh()])
        actor = CombinedPolicyNetwork(nn.Sequential(*layers),nn.Linear(sizes[-1],1))
        self.policy = VisionSLSPolicy(actor,self.encoder.distance_scale_m).to(device)
        self.policy.load_state_dict(saved['state_dict'])
        self.policy.set_state_distance_scale(saved['metadata']['state_distance_scale_m'])
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, images, measured_speed):
        """Return clipped braking acceleration in [-3,3]; positive means braking."""
        return self.policy.forward_image(self.encoder,images,measured_speed)
