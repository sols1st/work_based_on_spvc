"""Small causal world-model pilot, inspired by arXiv:2507.13871.

This is a CLS-feature adaptation, NOT the paper's full patch-token model.
All inputs at step t are observed features, executed/planned actions and speed.
Distances are labels/diagnostics only and are never model inputs.
"""
import numpy as np
import torch
from torch import nn


def split_trajectories(count, seed=7, fraction=.8):
    if count < 4 or not 0 < fraction < 1:
        raise ValueError("Need >=4 whole trajectories and a valid split")
    order = np.random.default_rng(seed).permutation(count)
    cut = min(count - 1, max(1, int(count * fraction)))
    return np.sort(order[:cut]), np.sort(order[cut:])


def window_indices(valid, trajectories, context=3, horizon=3):
    """Return [trajectory, last-context-time]; never cross gaps/episodes."""
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2 or min(context, horizon) < 1:
        raise ValueError("Invalid mask/context/horizon")
    result = []
    for episode in trajectories:
        for end in range(context - 1, valid.shape[1] - horizon):
            if valid[episode, end-context+1:end+horizon+1].all():
                result.append((int(episode), end))
    return np.asarray(result, dtype=np.int64).reshape(-1, 2)


def advance_speed(speed, action):
    return (speed - .05 * action.clamp(-3., 3.)).clamp(0., 3.)


class TemporalPredictor(nn.Module):
    def __init__(self, dimension, kind="transformer", context=3, width=128,
                 speed_mode="concat", use_action=True):
        super().__init__()
        if kind not in ("transformer", "mlp") or width % 4 or context < 1:
            raise ValueError("Invalid predictor settings")
        if speed_mode not in ("concat", "none", "add") or (kind == "mlp" and speed_mode == "add"):
            raise ValueError("Invalid speed conditioning")
        self.dimension, self.context, self.kind = dimension, context, kind
        self.speed_mode, self.use_action = speed_mode, bool(use_action)
        input_dimension = dimension + int(speed_mode == "concat") + int(use_action)
        if kind == "transformer":
            self.input = nn.Linear(input_dimension, width)
            self.position = nn.Parameter(torch.zeros(1, context, width))
            self.speed_embedding = nn.Linear(1, width) if speed_mode == "add" else None
            layer = nn.TransformerEncoderLayer(
                width, 4, width * 2, dropout=0., activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Linear(width, dimension)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dimension * context, width), nn.GELU(),
                nn.Linear(width, width), nn.GELU(),
            )
            self.head = nn.Linear(width, dimension)
        # Start from persistence so random initial outputs are not a handicap.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def sequence(self, latent, speed, action):
        if latent.ndim != 3 or latent.shape[1:] != (self.context, self.dimension):
            raise ValueError("Invalid context latent shape")
        if speed.shape != latent.shape[:2] or action.shape != speed.shape:
            raise ValueError("Invalid action/speed shape")
        parts = [latent]
        if self.speed_mode == "concat": parts.append(speed[..., None] / 3.)
        if self.use_action: parts.append(action[..., None].clamp(-3., 3.) / 3.)
        tokens = torch.cat(parts, dim=-1)
        if self.kind == "mlp":
            return latent[:, -1:] + .1 * self.head(self.encoder(tokens.flatten(1)))[:, None]
        mask = torch.triu(latent.new_full((self.context, self.context), float("-inf")), diagonal=1)
        embedded = self.input(tokens) + self.position
        if self.speed_mode == "add":
            embedded = embedded + self.speed_embedding(speed[..., None] / 3.)
        encoded = self.encoder(embedded, mask=mask)
        return latent + .1 * self.head(encoded)

    def forward(self, latent, speed, action):
        return self.sequence(latent, speed, action)[:, -1]


def rollout(model, latent_context, speed_context, action_sequence, horizon):
    """Autoregressive rollout. No future true latent or speed is consumed.

    action_sequence covers context + horizon - 1 planned/recorded controls.
    Future actions are an open-loop query, not claimed to be online known.
    """
    context = latent_context.shape[1]
    if action_sequence.shape != (len(latent_context), context + horizon - 1):
        raise ValueError("Wrong action horizon")
    history, speeds = latent_context, speed_context
    outputs = []
    for step in range(horizon):
        actions = action_sequence[:, step:step+context]
        predicted = history[:, -1] if model is None else model(history, speeds, actions)
        outputs.append(predicted)
        next_speed = advance_speed(speeds[:, -1], actions[:, -1])
        history = torch.cat((history[:, 1:], predicted[:, None]), dim=1)
        speeds = torch.cat((speeds[:, 1:], next_speed[:, None]), dim=1)
    return torch.stack(outputs, dim=1)
