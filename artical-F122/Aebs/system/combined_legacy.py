import os
import torch
import h5py
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 
import matplotlib.pyplot as plt
from cGAN.cGAN_common import Settings
from cGAN.taxi_models_and_data import AebsMLPGenerator
from Combined_network.model import AebsEnd2EndNet
from stable_baselines3 import PPO
from Aebs.system.env import AebsEnv


# -----------------------------
# 1. Data Loading
# -----------------------------
fn = "./Aebs/data/Downsampled.h5"
with h5py.File(fn, 'r') as f:
    # Input state data
    y_data = np.array(f["y_train"], dtype=np.float32)

std1 = np.std(y_data)

# -----------------------------
# 2. Load gen_net
# -----------------------------
s = Settings(latent_dim=4, nclasses=1)
gen_net = AebsMLPGenerator(s.latent_dim, s.nclasses)
gen_net.load_state_dict(torch.load("./Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth"))

# -----------------------------
# 3. Define the combined network
# -----------------------------
state_layer_sizes = [1024, 256, 64, 1]
# Load best saved PPO model
model = PPO.load('./Aebs/controller/best_model/best_model.zip')
policy = model.policy
mlp_extractor = policy.mlp_extractor.policy_net
action_net = policy.action_net
combined_net = AebsEnd2EndNet(gen_net, state_layer_sizes, mlp_extractor, action_net)
combined_net.state_net.load_state_dict(torch.load("./Aebs/controller/state_net_trained.pth"))
# combined_net.load_state_dict(torch.load("./Aebs/VT/model_03/Aebs_p_net.pth"))

# =========================
# Parameter Settings
# =========================
env = AebsEnv(std1)
num_episodes = 100
max_steps = 500
dt = 0.05
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
combined_net.to(device)
combined_net.eval()

# =========================
# Initialize batched trajectories
# =========================
# Real distance d ∈ [15, 16], speed v ∈ [2.5, 3]
d_init = np.random.uniform(15.0, 16.0, size=num_episodes)
v_init = np.random.uniform(2.5, 3.0, size=num_episodes)
d_batch = d_init.copy()
v_batch = v_init.copy()
d_norm_batch = d_batch / std1

# Data containers
s_trajs = [[] for _ in range(num_episodes)]
v_trajs = [[] for _ in range(num_episodes)]
done_mask = np.zeros(num_episodes, dtype=bool)

# =========================
# Batched simulation loop
# =========================
for step in range(max_steps):
    # Record data for trajectories that are not finished
    active_idx = ~done_mask
    if not active_idx.any():
        break

    s_trajs_step = d_batch[active_idx]
    v_trajs_step = v_batch[active_idx]

    for idx, (s, v) in zip(np.where(active_idx)[0], zip(s_trajs_step, v_trajs_step)):
        s_trajs[idx].append(s)
        v_trajs[idx].append(v)

    # Prepare inputs for the end-to-end network (batched)
    z = torch.zeros((active_idx.sum(), 4), device=device)
    ny = torch.tensor(d_norm_batch[active_idx], dtype=torch.float32, device=device).unsqueeze(1)
    speed_raw = torch.tensor(v_batch[active_idx], dtype=torch.float32, device=device).unsqueeze(1)

    with torch.no_grad():
        acc_tensor = combined_net(z, torch.cat([ny, speed_raw], dim=1)).squeeze(1).cpu().numpy()

    acc_tensor = np.clip(acc_tensor, env.action_space.low[0], env.action_space.high[0])

    # Vectorized state propagation
    d_next = d_batch[active_idx] - v_batch[active_idx] * dt
    v_next = v_batch[active_idx] - acc_tensor * dt
    v_next = np.clip(v_next, 0.0, 3.0)

    # Update state
    d_batch[active_idx] = d_next
    v_batch[active_idx] = v_next
    d_norm_batch[active_idx] = d_next / std1

    # Termination conditions
    SAFETY_DIST = 6.0
    SAFETY_SPEED = 0.5
    done_cond = (d_next <= 5.0) | (d_next >= 16.0) | (v_next <= 0.0)
    done_mask[active_idx] = done_cond

print("Batched simulation completed")

# =========================
# Plot 100 trajectories
# =========================
plt.figure(figsize=(8, 6))
for i in range(num_episodes):
    plt.plot(s_trajs[i], v_trajs[i], linewidth=0.8, alpha=0.6)

plt.xlabel('Distance s (m)')
plt.ylabel('Speed v (m/s)')
plt.title('s-v Trajectories (100 Episodes, Batch Inference)')
plt.grid(True)
plt.show()
