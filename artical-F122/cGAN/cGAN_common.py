import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import os
from dataclasses import dataclass, field
from typing import Any, Callable, List
from cGAN.spectral_norm import DenseSN, ConvSN

@dataclass
class Settings:
    G: Any = None # Generator model
    D: Any = None # Discriminator model
    loss: Any = None # Loss object (DCGANLoss, LSLoss, WLossGP, HingeLoss)
    img_fun: Callable = None # Data loading function
    rand_input: Callable = None # Function to generate random noise
    ranges: List[float] = None

    # Training Hyperparameters
    batch_size: int = 128
    latent_dim: int = 100
    nclasses: int = 2
    epochs: int = 120
    verbose_freq: int = 5

    # Output Settings
    output_x: int = 6
    output_y: int = 6
    output_dir: str = "output"

    # Optimizers: Now storing configuration dictionaries
    optD: dict = field(default_factory=lambda: {'lr': 0.0002, 'betas': (0.5, 0.99)})
    optG: dict = field(default_factory=lambda: {'lr': 0.0002, 'betas': (0.5, 0.99)})
    
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # New fields to store the configuration parameters internally
    optD_params: dict = field(init=False, default_factory=dict)
    optG_params: dict = field(init=False, default_factory=dict)


    def __post_init__(self):
        if isinstance(self.optD, dict):
            self.optD_params = self.optD
        else:
            raise TypeError("optD configuration must be a dictionary (e.g., {'lr': 1e-4, 'betas': (0.5, 0.99)})")
            
        if isinstance(self.optG, dict):
            self.optG_params = self.optG
        else:
            raise TypeError("optG configuration must be a dictionary (e.g., {'lr': 1e-4, 'betas': (0.5, 0.99)})")

        self.optD = None
        self.optG = None

# --- Loss Functions ---
def logitbinarycrossentropy_mean(logits, target_value):
    """
    Computes logitbinarycrossentropy loss and returns the mean.
    target_value: 1.0 for real, 0.0 for fake
    """
    targets = torch.full_like(logits, target_value)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')


## Regular DCGAN Loss
@dataclass
class DCGANLoss:
    """Standard DCGAN Minimax Loss."""
    def L_D(self, G, D, z, ny, x, y):
        # The discriminator should classify real images as 1
        real_loss = logitbinarycrossentropy_mean(D(x, y), 1.0)
        
        # Fake loss: D(G(z, ny), ny) should be 0
        fake_images = G(z, ny).detach()
        fake_loss = logitbinarycrossentropy_mean(D(fake_images, ny), 0.0)
        
        return real_loss + fake_loss

    def L_G(self, G, D, z, ny):
        # Generator loss: D(G(z, ny), ny) should be 1
        return logitbinarycrossentropy_mean(D(G(z, ny), ny), 1.0)


## Least Squares Loss (LSLoss)
@dataclass
class LSLoss:
    """Least Squares Loss (LSGAN)."""
    def L_D(self, G, D, z, ny, x, y):
        # Equivalent to Flux.mse(D(x, y), 1f0) + Flux.mse(D(G(z, ny), ny), 0f0)
        
        # Real loss: D(x, y) should be 1
        real_loss = F.mse_loss(D(x, y), torch.full_like(D(x, y), 1.0))
        
        # Fake loss: D(G(z, ny), ny) should be 0
        fake_images = G(z, ny).detach()
        fake_loss = F.mse_loss(D(fake_images, ny), torch.full_like(D(fake_images, ny), 0.0))
        
        return real_loss + fake_loss

    def L_G(self, G, D, z, ny):
        # Generator loss: D(G(z, ny), ny) should be 1
        return F.mse_loss(D(G(z, ny), ny), torch.full_like(D(G(z, ny), ny), 1.0))


## WGAN-GP (Partial implementation, full GP requires careful PyTorch porting)
@dataclass
class WLossGP:
    λ: float = 10.0
    gp_type: str = "exact"
    noise_range: float = 1.0

    def gradient_penalty(self, D, x_hat, y_hat, device):
        alpha = torch.rand(x_hat.size(0), 1, 1, 1, device=device) # Batch size, 1, 1, 1
        alpha = alpha.expand_as(x_hat)
        # 1. Compute D output for interpolated samples
        interpolates = x_hat.requires_grad_(True)
        d_interpolates = D(interpolates, y_hat)
        
        # 2. Compute gradients of D output w.r.t interpolated samples
        gradients = torch.autograd.grad(outputs=d_interpolates,
                                        inputs=interpolates,
                                        grad_outputs=torch.ones_like(d_interpolates, device=device),
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        
        # 3. Compute gradient penalty
        gradients = gradients.view(gradients.size(0), -1)
        
        # For simplicity, we use the standard WGAN-GP on image only.
        gradient_norm = gradients.norm(2, dim=1)
        
        # Penalty: (||∇D(x_hat)|| - 1)^2
        penalty = ((gradient_norm - 1)**2).mean()
        
        return penalty


    def L_D(self, G, D, z, ny, x, y):
        # Equivalent to mean(D(xtilde, ny) - D(x, y))
        
        # Real term (should be maxed): -D(x, y)
        real_loss = -D(x, y).mean()
        
        # Fake term (should be minimized): D(G(z, ny), ny)
        fake_images = G(z, ny).detach()
        fake_loss = D(fake_images, ny).mean()
        
        # WGAN-GP Loss D = -E[D(x)] + E[D(G(z))] + λ * GP
        loss = real_loss + fake_loss
        
        return loss

    def L_G(self, G, D, z, ny):
        # Equivalent to -mean(D(G(z, ny), ny))
        # WGAN Loss G = -E[D(G(z))] (maximize D(G(z)))
        return -D(G(z, ny), ny).mean()


## Hinge Loss
@dataclass
class HingeLoss:
    """Hinge Loss (for non-saturating GAN)."""
    def L_D(self, G, D, z, ny, x, y):
        # real_loss = mean(relu.(1f0 .- D(x, y)))
        real_loss = F.relu(1.0 - D(x, y)).mean()
        
        # fake_loss = mean(relu.(1f0 .+ D(G(z, ny), ny)))
        fake_images = G(z, ny).detach()
        fake_loss = F.relu(1.0 + D(fake_images, ny)).mean()
        
        return real_loss + fake_loss

    def L_G(self, G, D, z, ny):
        # Lᴳ(t::HingeLoss, G, D, z, ny) = -mean(D(G(z, ny), ny)) 
        return -D(G(z, ny), ny).mean()


# --- 3. Training Functions ---
def orthogonal_regularization(model, β=1e-4):
    reg = torch.tensor(0.0, device=next(model.parameters()).device)
    
    # Iterate over all layers that have a 'weight' attribute (Conv/Linear)
    for name, module in model.named_modules():
        if isinstance(module, (ConvSN, DenseSN)):
            # The weights for SN layers are stored in module.W (DenseSN) or module.conv.weight (ConvSN)
            if hasattr(module, 'W'): # DenseSN
                W = module.W
            elif hasattr(module.conv, 'weight'): # ConvSN
                W = module.conv.weight
            else:
                continue

            # Reshape W from [out_ch, in_ch, kH, kW] or [out, in] to [out, flat_in]
            if W.dim() > 2: # Convolutional weight
                W_flat = W.view(W.size(0), -1)
            else: # Linear weight
                W_flat = W

            # Compute W' * W
            prod = torch.matmul(W_flat.T, W_flat)
            
            # Identity matrix I
            I = torch.eye(prod.size(0), device=prod.device)
            
            # Julia code computes: norm(prod .* mat)^2
            # mat is ones - I (off-diagonal elements)
            # This is equivalent to summing the square of all off-diagonal elements in W'W
            off_diag_prod = prod * (1.0 - I)
            reg += torch.linalg.norm(off_diag_prod, 'fro')**2

    return β * reg

# train D
def train_discriminator(loss_obj, G, D, z, ny, x, y, optD):
    """One step of discriminator training."""
    D.zero_grad()
    loss_D = loss_obj.L_D(G, D, z, ny, x, y)
    loss_D.backward()
    optD.step()
    return loss_D.item()
# train G
def train_generator(loss_obj, G, D, z, ny, optG):
    """One step of generator training with orthogonal regularization."""
    G.zero_grad()
    loss_G = loss_obj.L_G(G, D, z, ny)
    
    # Add orthogonal regularization
    reg = orthogonal_regularization(G)
    total_loss_G = loss_G + reg

    total_loss_G.backward()
    optG.step()
    return total_loss_G.item()

def to_image(G, fixed_noise, fixed_labels, s: Settings):
    with torch.no_grad():
        fake_images = G(fixed_noise, fixed_labels).cpu()
        grid = vutils.make_grid(fake_images, nrow=s.output_x, padding=2, normalize=True)

    return grid

# def train(s: Settings):
#     # Initialize generator and discriminator
#     G = s.G(s).to(s.device)
#     D = s.D(s).to(s.device) 

#     # Create optimizer instances
#     s.optD = optim.Adam(D.parameters(), **s.optD_params)
#     s.optG = optim.Adam(G.parameters(), **s.optG_params)

#     # Load training dataset
#     # fixed_noise and fixed_labels are fixed for evaluating generator performance during training
#     data_loader, fixed_noise, fixed_labels = s.img_fun(s) 
    
#     fixed_noise = fixed_noise.to(s.device)
#     fixed_labels = fixed_labels.to(s.device)

#     # Create output directory
#     if not os.path.exists(s.output_dir):
#         os.makedirs(s.output_dir)

#     # Store training history
#     Dhist, Ghist = [], []
    
#     print(f"Starting Training on {s.device}...")
#     for epoch in range(1, s.epochs + 1):
#         print(f"Epoch {epoch}!")
#         avg_loss_D, avg_loss_G, steps = 0, 0, 0
        
#         for i, (x, y) in enumerate(data_loader):
#             # Move images to device
#             x = x.to(s.device)
#             # Move corresponding labels to device
#             y = y.to(s.device)
            
#             # Generate random noise z and fake labels
#             z, ny = s.rand_input(s) 
#             z = z.to(s.device)
#             ny = ny.to(s.device)

#             # Train discriminator
#             loss_D = train_discriminator(s.loss, G, D, z, ny, x, y, s.optD)
            
#             # Train generator with new noise and fake labels
#             z_G, ny_G = s.rand_input(s)
#             z_G = z_G.to(s.device)
#             ny_G = ny_G.to(s.device)
#             loss_G = train_generator(s.loss, G, D, z_G, ny_G, s.optG)
            
#             # Update history and averages
#             Dhist.append(loss_D)
#             Ghist.append(loss_G)
#             avg_loss_D += loss_D
#             avg_loss_G += loss_G
#             steps += 1			
        
#         avg_loss_D /= steps
#         avg_loss_G /= steps

#         # Save visualization every verbose_freq epochs
#         if epoch % s.verbose_freq == 0:
#             print(f"Discriminator loss = {avg_loss_D:.4f}, Generator loss = {avg_loss_G:.4f}")
            
#             # Save generated images during training for visualization
#             image_grid = to_image(G, fixed_noise, fixed_labels, s)
#             name = f"cgan_epochs_{epoch:06d}.png"
#             save_path = os.path.join(s.output_dir, name)
#             vutils.save_image(image_grid, save_path)
#             print(f"Saved generated image to {save_path}")

#     return G, D, Ghist, Dhist


def train(s: Settings):
    # Initialize generator and discriminator
    G = s.G(s).to(s.device)
    D = s.D(s).to(s.device) 

    # Create optimizer instances
    s.optD = optim.Adam(D.parameters(), **s.optD_params)
    s.optG = optim.Adam(G.parameters(), **s.optG_params)

    # Load training dataset
    # fixed_noise and fixed_labels are fixed for evaluating generator performance during training
    data_loader, fixed_noise, fixed_labels = s.img_fun(s) 
    fixed_noise = fixed_noise.to(s.device)
    fixed_labels = fixed_labels.to(s.device)

    # Create output directory
    if not os.path.exists(s.output_dir):
        os.makedirs(s.output_dir)

    # Store training history
    Dhist, Ghist = [], []
    
    # Track best generator
    best_loss_G = float('inf')
    best_model_path = os.path.join(s.output_dir, "best_G.pth")
    
    print(f"Starting Training on {s.device}...")
    for epoch in range(1, s.epochs + 1):
        print(f"Epoch {epoch}!")
        avg_loss_D, avg_loss_G, steps = 0, 0, 0
        
        for i, (x, y) in enumerate(data_loader):
            # Move images to device
            x = x.to(s.device)
            # Corresponding labels
            y = y.to(s.device)
            
            # Generate random noise z and fake labels
            z, ny = s.rand_input(s) 
            z = z.to(s.device)
            ny = ny.to(s.device)

            # Train discriminator
            loss_D = train_discriminator(s.loss, G, D, z, ny, x, y, s.optD)
            
            # Train generator with new noise and fake labels
            z_G, ny_G = s.rand_input(s)
            z_G = z_G.to(s.device)
            ny_G = ny_G.to(s.device)
            loss_G = train_generator(s.loss, G, D, z_G, ny_G, s.optG)
            
            # Update history and averages
            Dhist.append(loss_D)
            Ghist.append(loss_G)
            avg_loss_D += loss_D
            avg_loss_G += loss_G
            steps += 1			
        
        avg_loss_D /= steps
        avg_loss_G /= steps

        # Save visualization every verbose_freq epochs
        if epoch % s.verbose_freq == 0:
            print(f"Discriminator loss = {avg_loss_D:.4f}, Generator loss = {avg_loss_G:.4f}")
            
            image_grid = to_image(G, fixed_noise, fixed_labels, s)
            name = f"cgan_epochs_{epoch:06d}.png"
            save_path = os.path.join(s.output_dir, name)
            vutils.save_image(image_grid, save_path)
            print(f"Saved generated image to {save_path}")

            # Save the current best generator
            if avg_loss_G < best_loss_G:
                best_loss_G = avg_loss_G
                torch.save(G.state_dict(), best_model_path)
                print(f"Saved new best generator model at epoch {epoch}, loss {best_loss_G:.4f}")

    print(f"Training finished. Best generator saved at {best_model_path}")
    return G, D, Ghist, Dhist
