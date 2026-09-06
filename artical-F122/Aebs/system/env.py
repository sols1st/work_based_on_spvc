import torch
import h5py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple

# ----------------------
# Dynamics Model
# ----------------------
def next_state_vec(d, v, acc, dt=0.05):
    d_next = d - v * dt
    v_next = v - acc.squeeze() * dt

    return d_next, v_next


# -----------------------
# PPO Reinforcement Learning Environment Definition
# -----------------------
class AebsEnv(gym.Env):
    def __init__(self, std1):
        super(AebsEnv, self).__init__()
        self.std1 = std1
        self.dt = 0.05  # time step

        # ===== Observation space: normalized distance and speed =====
        d_min_norm = 5.0 / std1
        d_max_norm = 16.0 / std1
        v_min = 0.0
        v_max = 3.0

        self.observation_space = spaces.Box(
            low=np.array([d_min_norm, v_min], dtype=np.float32),
            high=np.array([d_max_norm, v_max], dtype=np.float32),
            dtype=np.float32
        )

        # ===== Action space: acceleration acc, range [-3.0, 3.0] =====
        self.action_space = spaces.Box(
            low=np.array([-3.0], dtype=np.float32),
            high=np.array([3.0], dtype=np.float32),
            dtype=np.float32
        )

        # Internal environment state: [d_norm, v]
        self.state = None

    def reset(self, seed=None, options=None):
        # Random initialization: real distance d ∈ [5.0, 16.0], speed v ∈ [0.0, 3.0]
        d_init = np.random.uniform(5.0, 16.0)
        v_init = np.random.uniform(0.0, 3.0)

        # Convert to normalized distance
        d_init_norm = d_init / self.std1

        # Set initial state [d_norm, v]
        self.state = np.array([d_init_norm, v_init], dtype=np.float32)

        # Must return: (observation, info)
        return self.state, {}  # info can be an empty dict

    def step(self, action):
        # Action is acceleration acc, clipped to valid range
        acc = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])

        # Current state: normalized distance and speed
        d_norm, v = self.state[0], self.state[1]
        d = d_norm * self.std1  # convert back to real distance

        # Update state using the original dynamics model
        d_next, v_next = next_state_vec(d, v, acc, self.dt)
        v_next = np.clip(v_next, 0.0, 3.0)  # limit maximum speed

        # Convert back to normalized distance
        d_next_norm = d_next / self.std1
        next_state = np.array([d_next_norm, v_next], dtype=np.float32)

        # ===== Reward function design =====
        reward = 0.0

        SAFETY_DIST = 6.0
        SAFETY_SPEED = 0.5

        # A. Distance progress reward (main goal: move closer)
        progress = d - d_next  # positive means approaching
        reward += 2.0 * progress  # tunable coefficient

        # B. Time penalty (encourages faster completion)
        reward -= 0.001

        # C. Termination conditions
        done = False
        truncated = False  # for non-task-related truncation (unused here)

        # Case 1: entering the safety distance
        if d_next <= SAFETY_DIST:
            if v_next <= SAFETY_SPEED:
                # Success: close and slow → bonus reward
                reward += 2
            else:
                # Inside safety zone but too fast → penalty
                reward -= (v_next - SAFETY_SPEED) * 3

        # Case 2: leave distance range [5.0, 16.0], or stop → terminate
        if d_next >= 16.0 or d_next <= 5.0 or v_next <= 0.0:
            done = True

        # Update internal state
        self.state = next_state

        # Must return: (observation, reward, terminated, truncated, info)
        return next_state, reward, done, truncated, {}
    

class Box:
    def __init__(self, low: np.ndarray, high: np.ndarray):
        """
        :param low: lower bound of the state space, np.ndarray
        :param high: upper bound of the state space, np.ndarray
        """
        self.low = low
        self.high = high
        self.shape = low.shape


class Aebs:
    def __init__(self, factor=0.01):
        # Load dataset
        fn = "./Aebs/data/Downsampled.h5"
        with h5py.File(fn, 'r') as f:
            y_data = np.array(f["y_train"], dtype=np.float32)

        self.std1 = np.std(y_data)
        y_data = y_data / self.std1

        # 1. Define state space and action space
        # observation_space: reachable state space (used for grid generation)
        self.observation_space = Box(
            low=np.array([np.min(y_data), 0.0]),  # example lower bound for 2D state
            high=np.array([np.max(y_data), 3.0])  # example upper bound for 2D state
        )

        # action_space: action space
        self.action_space = Box(
            low=np.array([-3.0]),
            high=np.array([3.0])     
        )

        # 2. Grid sizes for verification
        self.space_split: np.ndarray = np.array([100, 100])
        self.train_space_split: np.ndarray = np.array([100, 100])

        # init_spaces: list of initial state sets
        self.init_spaces: List[Box] = [
            Box(np.array([15.0/self.std1, 2.5]), np.array([16.0/self.std1, 3.0])),
        ]

        # unsafe_spaces: list of unsafe state sets
        self.unsafe_spaces: List[Box] = [
            Box(np.array([5.0/self.std1, 0.5]), np.array([6.0/self.std1, 3.0]))
        ]

        # self.noise_bounds: Tuple[np.ndarray, np.ndarray] = (
        #     np.array([-0.000005, -0.015]),   
        #     np.array([0.000005, 0.015])
        # )
        # self.noise = [0.000005, 0.015]

        # 3. State perturbation magnitude
        self.noise_bounds: Tuple[np.ndarray, np.ndarray] = (
            # noise lower bound
            (self.observation_space.low - self.observation_space.high) * factor,
            # noise upper bound
            (self.observation_space.high - self.observation_space.low) * factor
        )
        self.noise = (self.observation_space.high - self.observation_space.low) * factor

    # 4. Noise integration method (for computing probability mass of each noise cell)
    def integrate_noise(self, grid_lb: np.ndarray, grid_ub: np.ndarray) -> np.ndarray:
        """
        Compute the probability mass of a noise hyper-rectangle cell defined by its
        lower and upper bounds. For uniform noise, this equals the cell volume.

        :param grid_lb: list of lower bounds for each noise dimension
        :param grid_ub: list of upper bounds for each noise dimension
        :return: probability mass of each cell (np.ndarray)
        """
        # Assume noise is uniformly distributed within noise_bounds
        low = self.noise_bounds[0]
        high = self.noise_bounds[1]
        
        # Total noise space volume
        noise_volume = np.prod(high - low)
        
        # Cell volume
        grid_volume = np.prod(
            grid_ub - grid_lb, 
            axis=1
        )
        
        # Probability mass = volume(cell) / volume(noise space)
        pmass = grid_volume / noise_volume 

        return pmass
    
    def v_next(self, s, a):
        a = torch.clamp(a, min=self.action_space.low[0], max=self.action_space.high[0])

        d_next, v_next = next_state_vec(s[:,0]*self.std1, s[:,1], a)
        v_next = torch.clamp(v_next, 0.0, 3.0)

        # Normalize
        d_next = d_next / self.std1
        return torch.stack([d_next, v_next], dim=1)
