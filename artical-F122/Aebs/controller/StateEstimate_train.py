import os
import torch
import h5py
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 
from cGAN.cGAN_common import Settings
from torch.utils.data import DataLoader, TensorDataset
from cGAN.taxi_models_and_data import AebsMLPGenerator
from Combined_network.model import SubNet


# -----------------------------
# 1. Load data
# -----------------------------
fn = "./Aebs/data/Downsampled.h5"
with h5py.File(fn, 'r') as f:
    # Input state data
    y_data = np.array(f["y_train"], dtype=np.float32)

std1 = np.std(y_data)
y_data /= std1

labels_tensor = torch.from_numpy(y_data).float()

dataset = TensorDataset(labels_tensor)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=False)


# -----------------------------
# 2. Load observation model
# -----------------------------
s = Settings(latent_dim=4, nclasses=1)
gen_net = AebsMLPGenerator(s.latent_dim, s.nclasses)
gen_net.load_state_dict(torch.load("./Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth"))
gen_net.eval()
for p in gen_net.parameters():
    p.requires_grad = False

# -----------------------------
# 3. Define state_net
# -----------------------------
state_layer_sizes = [1024, 256, 64, 1]
state_net = SubNet(state_layer_sizes)
state_net.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen_net.to(device)
state_net.to(device)

# -----------------------------
# 4. Loss and optimizer
# -----------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(state_net.parameters(), lr=1e-4)


# -----------------------------
# 5. Lipschitz loss
# -----------------------------
def lipschitz_spectral_loss(model, L_max=2.0):
    """
    Estimate the Lipschitz constant of the network and compute
    penalty loss if it exceeds L_max.
    """
    lipschitz = torch.tensor(1.0, device=next(model.parameters()).device)

    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            W = layer.weight
            # Power iteration to estimate the largest singular value
            u = torch.randn(W.size(0), 1, device=W.device)
            for _ in range(1):  # one-step power iteration
                v = F.normalize(W.t() @ u, dim=0, eps=1e-12)
                u = F.normalize(W @ v, dim=0, eps=1e-12)
            sigma_max = (u.t() @ W @ v).squeeze()
            lipschitz = lipschitz * sigma_max

        elif isinstance(layer, (nn.ReLU, nn.Tanh)):
            lipschitz = lipschitz * 1.0

        elif isinstance(layer, nn.BatchNorm1d):
            gamma = layer.weight
            running_var = layer.running_var.detach()
            eps = layer.eps
            bn_lip = (gamma / torch.sqrt(running_var + eps)).abs().max()
            lipschitz = lipschitz * bn_lip

    penalty = F.relu(lipschitz - L_max) ** 2
    return lipschitz, penalty

# -----------------------------
# 6. Training loop
# -----------------------------
epochs = 50
lip_weight = 0.1
for epoch in range(epochs):
    epoch_loss = 0
    for y_batch in data_loader:
        y_batch = y_batch[0].to(device)

        z_batch = torch.rand(y_batch.size(0), 4, device=device) * 2 - 1
        # 1. Generate images
        with torch.no_grad():
            imgs = gen_net(z_batch, y_batch)

        # 2. Flatten input for state_net
        imgs_flat = imgs.view(imgs.size(0), -1)

        # 3. Forward
        pred_state = state_net(imgs_flat)

        # 4. Compute loss and backpropagate
        mse_loss = criterion(pred_state, y_batch)
        _, lip_penalty = lipschitz_spectral_loss(state_net, L_max=2.0)
        loss = mse_loss + lip_weight * lip_penalty
        
        # 5. Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss/len(data_loader):.6f}")

# Save model
# Get current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Save model to this directory
save_path = os.path.join(current_dir, "state_net_trained.pth")
torch.save(state_net.state_dict(), save_path)

print(f"✅ state_net has been saved to: {save_path}")
