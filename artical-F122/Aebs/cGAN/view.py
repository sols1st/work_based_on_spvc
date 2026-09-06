import os
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cGAN.cGAN_common import Settings
from cGAN.taxi_models_and_data import AebsMLPGenerator 

# -----------------------------
# 1. Load data
# -----------------------------
fn = "./Aebs/data/Downsampled.h5"
with h5py.File(fn, 'r') as f:
    y_data = np.array(f["y_train"], dtype=np.float32)
    X_data = np.array(f["X_train"], dtype=np.float32)

std1 = np.std(y_data)
s = Settings(latent_dim=4, nclasses=1)

# -----------------------------
# 2. Load generator
# -----------------------------
gen_net = AebsMLPGenerator(s.latent_dim, s.nclasses)
# Note: If the weights file does not exist, this will raise FileNotFoundError
gen_net.load_state_dict(torch.load("./Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth", map_location='cpu'))
    
gen_net.eval()

# -----------------------------
# 3. Construct input and generate images
# -----------------------------
idx = [50, 200, 350]
z = torch.zeros(len(idx), s.latent_dim)
y_vals = y_data[idx].reshape(-1, 1)
y = torch.tensor(y_vals / std1, dtype=torch.float32)

with torch.no_grad():
    fake_images = gen_net(z, y).cpu()  # [N, 1, 32, 32]
    # Normalize
    fake_images = (fake_images - fake_images.min()) / (fake_images.max() - fake_images.min())

# -----------------------------
# 4. Get Real (Downsampled) images
# -----------------------------
real_images = torch.tensor(X_data[idx]).unsqueeze(1)  # [N, 1, 32, 32]
real_images = (real_images - real_images.min()) / (real_images.max() - real_images.min())

# -----------------------------
# 5. Load Raw (High-resolution) images
# -----------------------------
raw_images = []
raw_dir = "./Aebs/carla_data"

for i in idx:
    filename = os.path.join(raw_dir, f"{i:04d}.png")
    img = Image.open(filename).convert("RGB")

    img = np.array(img) / 255.0
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # HWC -> CHW
    raw_images.append(img)
raw_images = torch.stack(raw_images)

# -----------------------------
# 6. Prepare visualization
# -----------------------------
display_size = 128
num_indices = len(idx)

def prepare_row(images, mode='bilinear', channels=3):
    """Prepare a single row of images: resize and concatenate horizontally"""
    row = []
    for img in images:
        img_4d = img.unsqueeze(0)
        # Convert grayscale to 3 channels
        if img_4d.shape[1] == 1 and channels == 3:
            img_4d = img_4d.repeat(1, 3, 1, 1)
        
        resized = torch.nn.functional.interpolate(img_4d, size=(display_size, display_size), mode=mode)
        row.append(resized.squeeze(0))
    return torch.cat(row, dim=2)

# Prepare three rows
raw_row = prepare_row(raw_images, mode='bilinear', channels=3)
# real_images is DownSample images; use nearest mode for upscaling
real_row = prepare_row(real_images.squeeze(1).unsqueeze(1), mode='nearest', channels=3) 
fake_row = prepare_row(fake_images, mode='nearest', channels=3)

# Concatenate three rows vertically
compare_grid = torch.cat([raw_row, real_row, fake_row], dim=1) 

# -----------------------------
# 7. Visualize and add labels
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))

ax.imshow(np.transpose(compare_grid.numpy(), (1, 2, 0)))
ax.axis('off')

# Label columns (state/condition)
for i in range(num_indices):
    x_center = i * display_size + display_size / 2
    ax.text(x_center, -10, f'd={y_vals[i, 0]:.2f}m', 
            ha='center', va='bottom', fontsize=12, color='black')

# Label rows (Raw, DownSample, Generated)
row_labels = ['Real', 'DownSample', 'Generated']
for i in range(len(row_labels)):
    y_center = i * display_size + display_size / 2
    ax.text(-10, y_center, row_labels[i], 
            ha='right', va='center', fontsize=12, color='black')

plt.tight_layout()
plt.show()