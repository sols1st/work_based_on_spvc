"""Development sensor surrogate from FIT-only real-image residuals.

Uses simulator pose to emulate a sensor offline; the actor sees only its 3D
output. Nearest-residual interpolation is neither real image inference nor a
certified error model. Calibration/audit residuals are never used for training.
"""
import numpy as np
from scipy.spatial import cKDTree
from .core import nominal_observation


class ResidualReplaySensor:
    def __init__(self, archive, nominal_fraction=.5, seed=17, jitter=0.):
        if not 0 <= nominal_fraction <= 1:
            raise ValueError('nominal fraction must be in [0,1]')
        with np.load(archive) as data:
            fit = data['splits'].astype(str) == 'fit'
            self.states = data['states'][fit, :3].copy()
            self.residuals = data['residuals'][fit].copy()
        if not len(self.states) or self.residuals.shape != self.states.shape:
            raise ValueError('paired fit-only pose/residual rows required')
        if not np.isfinite(self.states).all() or not np.isfinite(self.residuals).all():
            raise ValueError('finite residual data required')
        self.tree = cKDTree(self.states)
        self.rng = np.random.default_rng(seed)
        self.nominal_fraction = nominal_fraction
        if not np.isfinite(jitter) or jitter < 0:
            raise ValueError('nonnegative finite training jitter required')
        self.jitter = jitter
        self.reset()

    def reset(self):
        self.nominal = self.rng.random() < self.nominal_fraction
        self.bias = self.rng.uniform(-self.jitter, self.jitter, 3) if self.jitter else np.zeros(3)

    def __call__(self, state):
        y = nominal_observation(state)
        if self.nominal:
            return y
        distances, ids = self.tree.query(np.asarray(state)[:3], k=min(4, len(self.states)))
        weights = 1/(np.atleast_1d(distances)+.03)**2
        error = (self.residuals[np.atleast_1d(ids)]*weights[:, None]).sum(0)/weights.sum()
        return y+error+self.bias

    def close(self):
        pass
