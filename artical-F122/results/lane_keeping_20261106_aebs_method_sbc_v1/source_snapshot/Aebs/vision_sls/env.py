"""AEBS dynamics with nominal or saved-image low-dimensional observations."""
import numpy as np
import gymnasium as gym
from Aebs.system.env import AebsEnv


class ObservationEnv(gym.Env):
    def __init__(self, scale, distances=None, observations=None, mixed=False, max_steps=400):
        super().__init__()
        self.base = AebsEnv(scale, max_episode_steps=max_steps)
        self.action_space = self.base.action_space
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
        self.mixed = mixed
        self.use_images = distances is not None
        self.distances = None if distances is None else np.asarray(distances).reshape(-1)
        self.observations = None if observations is None else np.asarray(observations).reshape(-1)
        if self.use_images and (len(self.distances) == 0 or self.distances.shape != self.observations.shape
                                or not np.isfinite(self.distances).all() or not np.isfinite(self.observations).all()):
            raise ValueError('finite paired image distances and observations required')
        if mixed and not self.use_images:
            raise ValueError('mixed training requires image observations')
        self.image_episode = self.use_images
        self.lookup_errors = []

    def observe(self, state):
        d, v = state
        if self.image_episode:
            distance = d*self.base.std1
            index = int(np.argmin(np.abs(self.distances-distance)))
            self.lookup_errors.append(float(abs(self.distances[index]-distance)))
            d = self.observations[index]
        return np.array([d, v], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.image_episode = self.use_images and (not self.mixed or self.np_random.random() < .5)
        self.base.state = np.array([self.np_random.uniform(5.,16.)/self.base.std1,
                                    self.np_random.uniform(0.,3.)], dtype=np.float32)
        self.base.elapsed_steps = 0
        return self.observe(self.base.state), {}

    def reset_to(self, distance, speed):
        self.base.state = np.array([distance/self.base.std1, speed], dtype=np.float32)
        self.base.elapsed_steps = 0
        return self.observe(self.base.state)

    def step(self, action):
        state, reward, done, truncated, info = self.base.step(action)
        return self.observe(state), reward, done, truncated, info
