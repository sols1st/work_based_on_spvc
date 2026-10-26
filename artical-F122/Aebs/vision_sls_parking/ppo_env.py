"""History-based parking PPO interface, with no privileged state in observations.

Task/delay/dynamics are unchanged. Reward and PPO history interface are new.
Training reward may inspect simulator state; the actor/critic observations may not.
"""
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .env import ParkingEnv
from .core import (ACTION_HIGH, ACTION_LOW, GOAL_LOW, GOAL_HIGH, HORIZON,
                   STATE_LOW, STATE_HIGH, OBSTACLES)


class ParkingPPOEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, image_sensor=None, scenario=None):
        super().__init__()
        self.sensor, self.fixed_scenario = image_sensor, scenario
        # The sensor renders the delayed state selected by this original env.
        self.task = ParkingEnv(observation_delay=1, horizon=HORIZON)
        self.action_space = spaces.Box(-1., 1., (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (16,), dtype=np.float32)
        self.measurements, self.actions = deque(maxlen=3), deque(maxlen=2)

    def _measurement(self, raw):
        if self.sensor is None:
            return raw['nominal_observation'].copy()
        return np.asarray(self.sensor(raw['observation_state']), dtype=np.float64)

    def _observation(self):
        # Only measurements, executed actions, commanded goal, and clock.
        goal = (GOAL_LOW[self.task.scenario, :2]+GOAL_HIGH[self.task.scenario, :2])/2
        return np.concatenate([np.asarray(self.measurements).reshape(-1)/3.,
                               np.asarray(self.actions).reshape(-1), goal/3.,
                               [self.task.step_index/HORIZON]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        scenario = (self.fixed_scenario if self.fixed_scenario is not None
                    else int(self.np_random.integers(15)))
        if options is not None:
            scenario = options.get('scenario', scenario)
        if self.sensor is not None:
            self.sensor.reset()
        raw = self.task.reset(scenario=scenario)
        y = self._measurement(raw)
        self.measurements.clear()
        self.actions.clear()
        self.measurements.extend([y.copy() for _ in range(3)])
        self.actions.extend([np.zeros(2) for _ in range(2)])
        self.return_sum = 0.
        self.collision_seen, self.out_of_bounds_seen = False, False
        return self._observation(), {}

    def step(self, action):
        normalized = np.clip(np.asarray(action, dtype=np.float64), -1., 1.)
        if normalized.shape != (2,) or not np.isfinite(normalized).all():
            raise ValueError('finite normalized 2D action required')
        applied = np.clip(normalized*ACTION_HIGH, ACTION_LOW, ACTION_HIGH)
        goal = (GOAL_LOW[self.task.scenario, :2]+GOAL_HIGH[self.task.scenario, :2])/2
        old_distance = float(np.linalg.norm(self.task.state[:2]-goal))
        raw, done, info = self.task.step(applied)
        self.measurements.append(self._measurement(raw))
        self.actions.append(applied/ACTION_HIGH)
        x = self.task.state
        distance = float(np.linalg.norm(x[:2]-goal))
        # New shaping reward; NOT an upstream VISION-SLS reward/theorem.
        clearance = np.linalg.norm(x[:2]-OBSTACLES[:, :2], axis=1)-OBSTACLES[:, 2]
        bound_excess = np.maximum(STATE_LOW-x, 0)+np.maximum(x-STATE_HIGH, 0)
        reward = (10*(old_distance-distance)-.2*distance-.02*float(normalized@normalized)
                  -2*float(np.maximum(.15-clearance, 0).sum())
                  -20*float(info['collision'])-5*float(bound_excess.sum()))
        if done:
            reward += (30. if info['in_goal'] and not info['unsafe_seen'] else -10*distance)
        self.return_sum += reward
        self.collision_seen |= info['collision']
        self.out_of_bounds_seen |= not info['in_bounds']
        info.update(collision_seen=self.collision_seen,
                    out_of_bounds_seen=self.out_of_bounds_seen,
                    goal_at_horizon=bool(done and info['in_goal']),
                    episode_success=bool(done and info['in_goal'] and not info['unsafe_seen']),
                    final_goal_distance=distance)
        # Always run the original 30 steps, even after a collision or goal entry.
        return self._observation(), float(reward), False, done, info

    def close(self):
        self.task.close()


class FrozenImageSensor:
    """Actual OpenGL frame -> frozen official DINO-G -> frozen 3D projection."""
    def __init__(self, vision_root, dino_repo, weights, projection, device='cuda'):
        import torch
        from torchvision import transforms
        from .renderer import ParkingRenderer
        from .perception import load_projection
        self.torch, self.device = torch, device
        self.projection, self.c, self.info = load_projection(projection, device)
        self.backbone = torch.hub.load(str(dino_repo), 'dinov2_vitg14',
                                       source='local', pretrained=False)
        self.backbone.load_state_dict(torch.load(weights, map_location='cpu'), strict=True)
        self.backbone.to(device).eval().requires_grad_(False)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        self.renderer = ParkingRenderer(vision_root, backend='opengl')
        self.reset()

    def reset(self):
        self.last_state, self.last_y = None, None
        self.calls, self.outside_adaptation_domain = 0, 0

    def __call__(self, state):
        from PIL import Image
        # At t=0 and t=1 the original one-frame delay repeats the initial image.
        if self.last_state is not None and np.array_equal(state, self.last_state):
            return self.last_y.copy()
        rgb = self.renderer.render(state)
        x = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            f = self.backbone.forward_features(x)['x_norm_clstoken']
            y = self.projection(f).cpu().numpy()[0].astype(np.float64)
        self.calls += 1
        self.outside_adaptation_domain += int(not (-3 <= state[0] <= 0 and
            -1.5 <= state[1] <= 1.5 and abs(state[2]) <= .75*np.pi))
        self.last_state, self.last_y = state.copy(), y.copy()
        return y

    def close(self):
        self.renderer.close()
