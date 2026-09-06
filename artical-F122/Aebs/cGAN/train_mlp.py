import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import torchvision.utils as vutils
from cGAN.cGAN_common import Settings, to_image
from cGAN.taxi_models_and_data import AebsMLPGenerator, gen_aebs_images

# -------------------------------
# Training function
# -------------------------------
def train_mlp_supervised(ld: int, settings_class=Settings):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Settings
    current_dir = os.path.dirname(os.path.abspath(__file__))
    s = settings_class(
        latent_dim=ld,
        nclasses=1,
        batch_size=16,
        output_dir=os.path.join(current_dir, f"mlp_supervised_test_ld{ld}")
    )
    os.makedirs(s.output_dir, exist_ok=True)

    # 2. Create model
    m = AebsMLPGenerator(ld, s.nclasses).to(device)

    # 3. Load data
    # gen_aebs_images returns dataloader and fixed test set for visualization
    data_loader, fixed_noise, fixed_labels = gen_aebs_images(s, device, "./Aebs/data/Downsampled.h5")
    fixed_noise = fixed_noise.to(device)
    fixed_labels = fixed_labels.to(device)

    # 4. Optimizer
    lr = 1e-4
    optimizer = optim.Adam(m.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Supervised training, use L2 loss directly

    # 5. Training Loop
    num_epochs = 500
    verbose_freq = 10

    losses = []

    for epoch in range(num_epochs):
        m.train()
        epoch_loss = 0.0

        for batch_idx, (x_real, y_real) in enumerate(data_loader):
            x_real = x_real.to(device)
            y_real = y_real.to(device)

            # Random latent vector
            z_rand = torch.randn(x_real.size(0), s.latent_dim, device=device)

            optimizer.zero_grad()

            # Forward
            pred = m(z_rand, y_real)   # Replace random label with real label

            # Loss
            loss = criterion(pred, x_real)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(data_loader)
        losses.append(avg_loss)

        if epoch % verbose_freq == 0:
            print(f"Epoch {epoch:03d}, Avg Loss: {avg_loss:.6f}")

            # Save generated images
            m.eval()
            with torch.no_grad():
                image_grid = to_image(m, fixed_noise, fixed_labels, s)
            save_path = os.path.join(s.output_dir, f"mlp_epoch_{epoch:03d}.png")
            vutils.save_image(image_grid, save_path)
            print(f"Saved generated image to {save_path}")
            m.train()

    # 6. Save final model and losses
    final_model_path = os.path.join(s.output_dir, "mlp_supervised.pth")
    torch.save(m.state_dict(), final_model_path)
    np.save(os.path.join(s.output_dir, "losses.npy"), np.array(losses))
    print(f"Training finished. Model saved to {final_model_path}")

# -------------------------------
# Main
# -------------------------------
if __name__ == '__main__':
    train_mlp_supervised(ld=4)