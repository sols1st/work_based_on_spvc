import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import os
import random
import torchvision.utils as vutils
from torch.utils.data import DataLoader, TensorDataset
from typing import List
from cGAN.spectral_norm import DenseSN, ConvSN
from cGAN.cGAN_common import Settings

# --- Discriminator Models ---

class TaxiDConvSpectral(nn.Module):
    def __init__(self, s: Settings):
        super().__init__()
        self.d_labels = nn.Sequential(
            DenseSN(s.nclasses, 16 * 8), # Output size: 128
            ReshapeLayer((-1, 1, 8, 16)) # PyTorch: (B, C=1, H=8, W=16)
        )
        
        leaky_relu = nn.LeakyReLU(0.1) 

        self.d_common = nn.Sequential(
            ConvSN(in_channels=2, out_channels=64, kernel_size=3, padding=1, stride=1, activation=leaky_relu),
            ConvSN(in_channels=64, out_channels=128, kernel_size=4, padding=1, stride=2, activation=leaky_relu),
            ConvSN(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1, activation=leaky_relu),
            ConvSN(in_channels=128, out_channels=256, kernel_size=4, padding=1, stride=2, activation=leaky_relu),
            ConvSN(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1, activation=leaky_relu),
            ConvSN(in_channels=256, out_channels=512, kernel_size=4, padding=1, stride=2, activation=leaky_relu),
            ConvSN(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1, activation=leaky_relu),
            nn.Flatten(),
            nn.Linear(1024, 1)
        )

    def forward(self, x, y):
        # 1. Process labels and reshape them into feature maps
        y_map = self.d_labels(y)
        # 2. Concatenate the image and feature maps along the channel dimension
        t = torch.cat([y_map, x], dim=1)
        # 3. Discriminate
        return self.d_common(t)

class TaxiDMLP(nn.Module):
    def __init__(self, s: Settings):
        super().__init__()
        in_features = 16 * 8 + s.nclasses
        
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, y):
        x_flat = x.view(x.size(0), -1) 
        t = torch.cat([x_flat, y], dim=1) 
        return self.net(t)


# --- Generator Models ---

# Custom Reshape Layer for PyTorch Sequential
class ReshapeLayer(nn.Module):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, x):
        # target_shape should include batch size -1
        return x.view(*self.target_shape)

class TaxiGConv(nn.Module):
    def __init__(self, s: Settings):
        super().__init__()
        
        # Orthogonal Initialization
        def init_orthogonal(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 1. g_labels
        self.g_labels = nn.Sequential(
            nn.Linear(s.nclasses, 32),
            ReshapeLayer((-1, 16, 1, 2)) # (B, C=16, H=1, W=2)
        )
        self.g_labels.apply(init_orthogonal)
        
        # 2. g_latent
        self.g_latent = nn.Sequential(
            nn.Linear(s.latent_dim, 992),
            ReshapeLayer((-1, 496, 1, 2)) # (B, C=496, H=1, W=2)
        )
        self.g_latent.apply(init_orthogonal)

        # 3. g_common
        self.g_common = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        self.g_common.apply(init_orthogonal)


    def forward(self, x, y):
        # x: (B, latent_dim) latent noise; y: (B, nclasses) label
        x_map = self.g_latent(x)
        y_map = self.g_labels(y)

        t = torch.cat([y_map, x_map], dim=1)
    
        return self.g_common(t)

class TaxiGMLP(nn.Module):
    def __init__(self, s: Settings, big_mlp=True):
        super().__init__()
        in_features = s.latent_dim + s.nclasses
        
        if big_mlp:
            hidden_layers = [
                nn.Linear(256, 256),
                nn.ReLU()
            ] * 3 
            
            self.net = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                *hidden_layers,
                nn.Linear(256, 16 * 8), 
                ReshapeLayer((-1, 1, 8, 16)) 
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 16 * 8), 
                ReshapeLayer((-1, 1, 8, 16))
            )

    def forward(self, x, y):
        t = torch.cat([x, y], dim=1) 

        return self.net(t)


class AebsGConv(nn.Module):
    def __init__(self, s):
        super().__init__()

        def init_orthogonal(m):
            if isinstance(m, (nn.Linear, nn.ConvTranspose2d)):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        in_dim = s.latent_dim + s.nclasses
        self.initial_projection = nn.Sequential(
            nn.Linear(in_dim, 128 * 2 * 2),
            nn.BatchNorm1d(128 * 2 * 2),
            nn.ReLU(),
            ReshapeLayer((-1, 128, 2, 2))
        )
        self.initial_projection.apply(init_orthogonal)

        self.g_common = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Tanh()
        )
        self.g_common.apply(init_orthogonal)

    def forward(self, x, y):
        t = torch.cat([x, y], dim=1)
        t_map = self.initial_projection(t)
        return self.g_common(t_map)


class AebsDConvSpectral(nn.Module):
    def __init__(self, s):
        super().__init__()
        leaky_relu = nn.LeakyReLU(0.1)

        self.d_labels = nn.Sequential(
            DenseSN(s.nclasses, 16*16),
            ReshapeLayer((-1,1,16,16)),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        self.d_common = nn.Sequential(
            ConvSN(2, 32, 3, 1, 1, activation=leaky_relu),
            ConvSN(32, 64, 4, 2, 1, activation=leaky_relu),  # 32->16
            ConvSN(64, 128, 4, 2, 1, activation=leaky_relu), # 16->8
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            DenseSN(128, 1)
        )

    def forward(self, x, y):
        y_map = self.d_labels(y)
        t = torch.cat([y_map, x], dim=1)
        return self.d_common(t)
    
class AebsMLPGenerator(nn.Module):
    def __init__(self, latent_dim, nclasses):
        super().__init__()
        def init_orthogonal(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        input_dim = latent_dim + nclasses
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1*32*32),
            nn.Tanh(),
            ReshapeLayer((-1, 1, 32, 32))
        )
        self.net.apply(init_orthogonal)
    
    def forward(self, z, y):
        x = torch.cat([z, y], dim=1)
        return self.net(x)
# --- 3. Data and Input Functions ---

def taxi_input(s: Settings, device: torch.device, ranges: List[float]):
    """
    Function that returns random input for the generator (z, ny).
    ranges: list of floats, each defining the range [-r, r] for one dimension.
    """
    # Generate latent variable z in the range (-1.0, 1.0) with a uniform distribution
    z = (torch.rand(s.batch_size, s.latent_dim, device=device) * 2.0 - 1.0).float()

    # Generate conditional inputs for each dimension and concatenate them
    ys = []
    for r in ranges:
        y = (torch.rand(s.batch_size, 1, device=device) * 2 * r - r).float()
        ys.append(y)

    # Concatenate into a tensor of shape (B, len(ranges))
    ny = torch.cat(ys, dim=1)

    return z, ny

def gen_taxi_images(s: Settings, device: torch.device, data_path: str):
    """
    Data loading, fixed noise generation, and label preparation
    """
    # Data storage path
    with h5py.File(data_path, 'r') as f:
        # Input state data
        y_data = np.array(f["y_train"], dtype=np.float32) 
        # Output image data
        images_data = np.array(f["X_train"], dtype=np.float32)

    # 1. Input normalization
    std1, std2 = np.std(y_data[:, 0]), np.std(y_data[:, 1])
    y_data[:, 0] /= std1
    y_data[:, 1] /= std2

    print("extrema of dim1: ", np.min(y_data[:, 0]), np.max(y_data[:, 0]), 
          " extrema of dim2: ", np.min(y_data[:, 1]), np.max(y_data[:, 1]))
    
    s.ranges = [np.max(np.abs(y_data[:, 0])), np.max(np.abs(y_data[:, 1]))]
    # Use only the first two dimensions as labels for training
    labels_train = y_data[:, :2]

    # 2. Image data processing
    images_train = (2.0 * images_data - 1.0)
    images_train = np.expand_dims(images_train, axis=1)

    images_tensor = torch.from_numpy(images_train).float()
    labels_tensor = torch.from_numpy(labels_train).float()
    
    # 3. Wrap images and labels into an indexable dataset
    dataset = TensorDataset(images_tensor, labels_tensor)
    data_loader = DataLoader(dataset, batch_size=s.batch_size, shuffle=True, drop_last=False)

    # 4. Number of images to evaluate during training
    N = s.output_x * s.output_y
    
    # Generate N samples of fixed noise
    fixed_noise = (torch.rand(N, s.latent_dim, device=device) * 2.0 - 1.0).float()
    
    # Generate N fixed labels
    random.seed(0)
    indices = random.sample(range(len(labels_train)), N)
    fixed_labels = labels_tensor[indices].to(device)

    # 5. Save real images for comparison
    real_images_samples = images_tensor[indices]
    
    # Map back to real image range
    real_images_norm = (real_images_samples + 1.0) / 2.0

    # Save the real image grid
    image_grid = vutils.make_grid(
        real_images_norm,
        nrow=s.output_x,
        padding=2,
        normalize=False
    )
    
    if not os.path.exists(s.output_dir):
        os.makedirs(s.output_dir)
        
    vutils.save_image(image_grid, os.path.join(s.output_dir, "real_images_sk.png"))
    print(f"Saved real images grid to {os.path.join(s.output_dir, 'real_images_sk.png')}")
    
    return data_loader, fixed_noise, fixed_labels



def gen_aebs_images(s: Settings, device: torch.device, data_path: str):
    """
    Data loading, fixed noise generation, and label preparation
    """
    # Data storage path
    with h5py.File(data_path, 'r') as f:
        y_data = np.array(f["y_train"], dtype=np.float32) 
        images_data = np.array(f["X_train"], dtype=np.float32)

    # 1. Input normalization
    std1 = np.std(y_data[:, 0])
    y_data[:, 0] /= std1
    print("extrema of dim1: ", np.min(y_data[:, 0]), np.max(y_data[:, 0]))
    
    s.ranges = [np.max(np.abs(y_data[:, 0]))]

    labels_train = y_data

    # 2. Image data processing
    images_train = (2.0 * images_data - 1.0)
    images_train = np.expand_dims(images_train, axis=1)

    images_tensor = torch.from_numpy(images_train).float()
    labels_tensor = torch.from_numpy(labels_train).float()
    
    # 3. Wrap images and labels into an indexable dataset
    dataset = TensorDataset(images_tensor, labels_tensor)
    data_loader = DataLoader(dataset, batch_size=s.batch_size, shuffle=True, drop_last=False)

    # 4. Number of images for evaluation during training
    N = s.output_x * s.output_y
    
    # Generate N samples of fixed noise
    fixed_noise = (torch.rand(N, s.latent_dim, device=device) * 2.0 - 1.0).float()
    
    # Generate N fixed labels
    # Fix random seed for reproducibility
    random.seed(0)
    # Randomly sample N labels from the original dataset
    indices = random.sample(range(len(labels_train)), N)
    fixed_labels = labels_tensor[indices].to(device)

    # 5. Save real images for comparison
    real_images_samples = images_tensor[indices]
    
    # Map images back to real image range
    real_images_norm = (real_images_samples + 1.0) / 2.0

    # Save the image grid
    image_grid = vutils.make_grid(
        real_images_norm,
        nrow=s.output_x,
        padding=2,
        normalize=False
    )
    
    if not os.path.exists(s.output_dir):
        os.makedirs(s.output_dir)
        
    vutils.save_image(image_grid, os.path.join(s.output_dir, "real_images_sk.png"))
    print(f"Saved real images grid to {os.path.join(s.output_dir, 'real_images_sk.png')}")
    
    return data_loader, fixed_noise, fixed_labels