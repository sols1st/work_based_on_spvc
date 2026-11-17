"""Official frozen DINOv2 feature extraction for the 32x32 AEBS images."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrozenDinoV2(nn.Module):
    """Return the 384-D CLS embedding of official DINOv2-S/14."""

    def __init__(self, repo_path, weights_path, device, model_name="dinov2_vits14"):
        super().__init__()
        repo = Path(repo_path)
        weights = Path(weights_path)
        if not (repo / "hubconf.py").is_file():
            raise FileNotFoundError("DINOv2 repository not found: %s" % repo)
        if not weights.is_file():
            raise FileNotFoundError("DINOv2 weights not found: %s" % weights)
        backbone = torch.hub.load(
            str(repo), model_name, source="local", pretrained=False
        )
        backbone.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
        backbone.to(device).eval()
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        # The official backbone is an external immutable asset and is not
        # duplicated in downstream projection/controller checkpoints.
        self.__dict__["_backbone"] = backbone
        self.model_name = str(model_name)
        self.repo_path = str(repo_path)
        self.weights_path = str(weights_path)

    @property
    def backbone(self):
        return self.__dict__["_backbone"]

    @staticmethod
    def preprocess(image):
        if image.ndim == 3:
            image = image.unsqueeze(1)
        if image.ndim != 4 or image.shape[1] not in (1, 3):
            raise ValueError("image must be [B,H,W], [B,1,H,W], or [B,3,H,W]")
        image = image.float()
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        image = F.interpolate(
            image, size=(224, 224), mode="bicubic", align_corners=False,
            antialias=True,
        )
        mean = image.new_tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)
        std = image.new_tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
        return (image - mean) / std

    def forward(self, image):
        with torch.no_grad():
            return self.backbone(self.preprocess(image))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

