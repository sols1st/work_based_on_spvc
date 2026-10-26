"""Controlled feature-prediction pilots, not full paper reproductions.

RSSM uses Gaussian prior/posterior (PlaNet-inspired, not categorical DreamerV3).
Lifted linear model uses learned coordinates with action/speed forcing; it does
not establish an invariant Koopman subspace. Original PPO/SBC are untouched.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from Aebs.dino_latent.temporal_world_model import TemporalPredictor, rollout, advance_speed


def split_three(count, seed=71):
    if count < 12:
        raise ValueError("Need >=12 whole trajectories")
    order = np.random.default_rng(seed).permutation(count)
    a, b = int(.7 * count), int(.85 * count)
    return tuple(np.sort(x) for x in (order[:a], order[a:b], order[b:]))


def temporal_offsets(context, horizon):
    if min(context,horizon)<1: raise ValueError("Invalid context/horizon")
    return (np.arange(1-context,1),np.arange(1,horizon+1),np.arange(1-context,horizon))


def paired_cluster_bootstrap(candidate, baseline, groups, draws=2000, seed=7):
    """Equal-trajectory mean paired difference, resampling whole trajectories.

    This is not an independent-frame CI, nor a multi-training-seed CI.
    """
    candidate, baseline, groups = map(np.asarray, (candidate, baseline, groups))
    if candidate.ndim != 1 or candidate.shape != baseline.shape or groups.shape != candidate.shape:
        raise ValueError("Need paired scalar per-window errors and group IDs")
    if draws < 10 or not len(candidate) or not np.isfinite(candidate-baseline).all():
        raise ValueError("Invalid bootstrap input")
    delta = np.array([(candidate[groups == g]-baseline[groups == g]).mean() for g in np.unique(groups)])
    if len(delta) < 2:
        raise ValueError("Need >=2 trajectories")
    samples = delta[np.random.default_rng(seed).integers(len(delta), size=(draws, len(delta)))].mean(1)
    return dict(trajectory_count=len(delta), mean_paired_delta=float(delta.mean()),
                ci95=[float(x) for x in np.quantile(samples, [.025, .975])],
                lower_error_trajectory_fraction=float((delta < 0).mean()),
                estimand="equal-trajectory mean absolute-error difference; negative favors candidate",
                scope="conditional on this trained pair and old development distribution")


class GRUPredictor(nn.Module):
    def __init__(self, dimension=32, width=128):
        super().__init__()
        self.cell = nn.GRUCell(dimension+2, width)
        self.head = nn.Linear(width, dimension)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def predict(self, z, v, a, horizon):
        h = z.new_zeros(len(z), self.cell.hidden_size)
        for i in range(z.shape[1]):
            h = self.cell(torch.cat((z[:, i], v[:, i:i+1]/3, a[:, i:i+1]/3), -1), h)
        current, speed, result = z[:, -1], v[:, -1], []
        for t in range(horizon):
            current = current + .1*self.head(h)
            result.append(current)
            if t+1 < horizon:
                speed = advance_speed(speed, a[:, z.shape[1]-1+t])
                control = a[:, z.shape[1]+t]
                h = self.cell(torch.cat((current, speed[:, None]/3, control[:, None]/3), -1), h)
        return torch.stack(result, 1), z.new_zeros(())


class LiftedLinearPredictor(nn.Module):
    def __init__(self, dimension=32, context=3, width=128, lift=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(context*(dimension+2), width), nn.GELU(), nn.Linear(width, lift))
        self.transition = nn.Linear(lift, lift, bias=False)
        self.forcing = nn.Linear(2, lift)
        self.decoder = nn.Sequential(nn.Linear(lift, width), nn.GELU(), nn.Linear(width, dimension))
        with torch.no_grad():
            self.transition.weight.copy_(torch.eye(lift)+.001*torch.randn(lift, lift))
        self.reconstruct = nn.Linear(lift, dimension)

    def predict(self, z, v, a, horizon):
        c = z.shape[1]
        h = self.encoder(torch.cat((z, v[..., None]/3, a[:, :c, None]/3), -1).flatten(1))
        auxiliary = .1*F.mse_loss(self.reconstruct(h), z[:, -1])
        anchor, speed, result = z[:, -1], v[:, -1], []
        for t in range(horizon):
            control = a[:, c-1+t]
            h = self.transition(h) + self.forcing(torch.stack((speed/3, control/3), -1))
            result.append(anchor+.1*self.decoder(h))
            speed = advance_speed(speed, control)
        return torch.stack(result, 1), auxiliary


class GaussianRSSM(nn.Module):
    def __init__(self, dimension=32, hidden=128, stochastic=16):
        super().__init__()
        self.hidden, self.stochastic = hidden, stochastic
        self.cell = nn.GRUCell(stochastic+2, hidden)
        self.prior = nn.Linear(hidden, 2*stochastic)
        self.posterior = nn.Sequential(nn.Linear(hidden+dimension, hidden), nn.ELU(), nn.Linear(hidden, 2*stochastic))
        self.decoder = nn.Sequential(nn.Linear(hidden+stochastic, hidden), nn.ELU(), nn.Linear(hidden, dimension))

    def distribution(self, parameters):
        mean, raw = parameters.chunk(2, -1)
        return mean, F.softplus(raw)+.1

    @staticmethod
    def kl(qmean, qstd, pmean, pstd):
        return (torch.log(pstd/qstd)+(qstd.square()+(qmean-pmean).square())/(2*pstd.square())-.5).mean()

    def predict(self, z, v, a, horizon, targets=None):
        c = z.shape[1]
        h = z.new_zeros(len(z), self.hidden)
        state = z.new_zeros(len(z), self.stochastic)
        # The observed context may update the posterior; future targets may not.
        for i in range(c):
            if i:
                h = self.cell(torch.cat((state, v[:, i-1:i]/3, a[:, i-1:i]/3), -1), h)
            state, _ = self.distribution(self.posterior(torch.cat((h, z[:, i]), -1)))
        start_h, start_state = h, state
        speed, anchor, result = v[:, -1], z[:, -1], []
        for t in range(horizon):
            control = a[:, c-1+t]
            h = self.cell(torch.cat((state, speed[:, None]/3, control[:, None]/3), -1), h)
            mu, sigma = self.distribution(self.prior(h))
            state = mu + sigma*torch.randn_like(mu) if self.training else mu
            result.append(anchor+.1*self.decoder(torch.cat((h, state), -1)))
            speed = advance_speed(speed, control)
        auxiliary = z.new_zeros(())
        # Separate teacher-filtered reconstruction/KL training path. Predictions
        # above never consume targets, even when this auxiliary path is enabled.
        if targets is not None:
            h, state, speed = start_h, start_state, v[:, -1]
            terms = []
            for t in range(horizon):
                control = a[:, c-1+t]
                h = self.cell(torch.cat((state, speed[:, None]/3, control[:, None]/3), -1), h)
                pm, ps = self.distribution(self.prior(h))
                qm, qs = self.distribution(self.posterior(torch.cat((h, targets[:, t]), -1)))
                state = qm+qs*torch.randn_like(qm)
                reconstruction = anchor+.1*self.decoder(torch.cat((h, state), -1))
                terms.append(.1*F.mse_loss(reconstruction, targets[:, t])+.001*self.kl(qm, qs, pm, ps))
                speed = advance_speed(speed, control)
            auxiliary = torch.stack(terms).mean()
        return torch.stack(result, 1), auxiliary


def make_model(kind, dimension=32):
    if kind == "mlp_matched":
        return TemporalPredictor(dimension, "mlp", width=456)
    if kind in ("transformer", "mlp", "no_speed", "task_aware"):
        return TemporalPredictor(dimension, "mlp" if kind == "mlp" else "transformer",
                                 speed_mode="none" if kind == "no_speed" else "concat")
    if kind == "gru": return GRUPredictor(dimension)
    if kind == "lifted_linear": return LiftedLinearPredictor(dimension)
    if kind == "rssm": return GaussianRSSM(dimension)
    raise ValueError("Unknown model kind")


def predict(model, z, v, a, horizon, targets=None):
    if isinstance(model, GaussianRSSM): return model.predict(z, v, a, horizon, targets)
    if isinstance(model, (GRUPredictor, LiftedLinearPredictor)): return model.predict(z, v, a, horizon)
    return rollout(model, z, v, a, horizon), z.new_zeros(())
