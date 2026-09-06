import os
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from cGAN.cGAN_common import Settings
from cGAN.taxi_models_and_data import AebsMLPGenerator
from Combined_network.model import AebsEnd2EndNet
from stable_baselines3 import PPO
from Aebs.system.env import AebsEnv
import math

# -----------------------------
# 1. Load data
# -----------------------------
fn = "./Aebs/data/Downsampled.h5"
with h5py.File(fn, 'r') as f:
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
model = PPO.load('./Aebs/controller/best_model/best_model.zip')
policy = model.policy
mlp_extractor = policy.mlp_extractor.policy_net
action_net = policy.action_net
combined_net = AebsEnd2EndNet(gen_net, state_layer_sizes, mlp_extractor, action_net)
combined_net.state_net.load_state_dict(torch.load("./Aebs/controller/state_net_trained.pth"))
# combined_net.load_state_dict(torch.load("./Aebs/Aebs_p_net.pth"))

# -----------------------------
# 4. Initialize environment and device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
combined_net.to(device)
dt = 0.05
env = AebsEnv(std1)
combined_net.to(device)

# -----------------------------
# 5. Define number of states and disturbances
# -----------------------------
num_states = 5000
num_z = 10000
batch_size = 100000  # Sample size for each inference, adjust based on GPU memory

# -----------------------------
# 6. Randomly sample initial states
# -----------------------------
s_batch = np.random.uniform(5.0, 16.0, size=num_states)
v_batch = np.random.uniform(0.0, 3.0, size=num_states)
s_norm_batch = s_batch / std1

# -----------------------------
# 7. Baseline inference with z = 0
# -----------------------------
z0 = torch.zeros(num_states, 4, device=device)
s_tensor_base = torch.tensor(s_norm_batch, dtype=torch.float32, device=device).unsqueeze(1)
v_tensor_base = torch.tensor(v_batch, dtype=torch.float32, device=device).unsqueeze(1)

with torch.no_grad():
    acc0 = combined_net(z0, torch.cat([s_tensor_base, v_tensor_base],dim=1)).squeeze(1).cpu().numpy()

acc0 = np.clip(acc0, env.action_space.low[0], env.action_space.high[0])
s0_next = s_batch - v_batch * dt
v0_next = np.clip(v_batch - acc0 * dt, 0.0, 3.0)

# -----------------------------
# 8. Construct disturbed states, each state with num_z different z
# -----------------------------
s_tensor = s_tensor_base.unsqueeze(1).repeat(1, num_z, 1).view(-1, 1)
v_tensor = v_tensor_base.unsqueeze(1).repeat(1, num_z, 1).view(-1, 1)
z_rand = torch.tensor(np.random.uniform(-1.0, 1.0, size=(num_states*num_z, 4)),
                      dtype=torch.float32, device=device)

# -----------------------------
# 9. Batched inference
# -----------------------------
num_total = num_states * num_z
acc_z_flat = np.zeros(num_total, dtype=np.float32)
num_batches = math.ceil(num_total / batch_size)

for i in range(num_batches):
    start = i * batch_size
    end = min((i+1) * batch_size, num_total)
    with torch.no_grad():
        acc_batch = combined_net(z_rand[start:end], torch.cat([s_tensor[start:end], v_tensor[start:end]], dim=1)).squeeze(1).cpu().numpy()
        acc_z_flat[start:end] = acc_batch

# -----------------------------
# 10. Update states
# -----------------------------
acc_z_flat = np.clip(acc_z_flat, env.action_space.low[0], env.action_space.high[0])
s_next = s_tensor.squeeze(1).cpu().numpy() * std1 - v_tensor.squeeze(1).cpu().numpy() * dt
v_next = np.clip(v_tensor.squeeze(1).cpu().numpy() - acc_z_flat * dt, 0.0, 3.0)

s_next = s_next.reshape(num_states, num_z)
v_next = v_next.reshape(num_states, num_z)

# -----------------------------
# 11. Compute disturbance magnitudes Δs, Δv relative to baseline z=0
# -----------------------------
s0_next_matrix = s0_next[:, None]
v0_next_matrix = v0_next[:, None]

delta_s = s_next - s0_next_matrix
delta_v = v_next - v0_next_matrix

# -----------------------------
# 12. Visualization of mean/std of disturbances per state
# -----------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(delta_s.mean(axis=1), bins=30, alpha=0.7)
plt.xlabel('Δs mean per state')
plt.ylabel('Frequency')
plt.title('Distribution of Distance Perturbation (mean per state)')
plt.grid(True)

plt.subplot(1,2,2)
plt.hist(delta_v.std(axis=1), bins=30, alpha=0.7, color='orange')
plt.xlabel('Δv std per state')
plt.ylabel('Frequency')
plt.title('Distribution of Speed Perturbation (std per state)')
plt.grid(True)

plt.tight_layout()
plt.show()

# -----------------------------
# 13. Output overall statistics
# -----------------------------
print(f"Δs mean: {delta_s.mean():.4f}, std: {delta_s.std():.4f}, min: {delta_s.min():.4f}, max: {delta_s.max():.4f}")
print(f"Δv mean: {delta_v.mean():.4f}, std: {delta_v.std():.4f}, min: {delta_v.min():.4f}, max: {delta_v.max():.4f}")
