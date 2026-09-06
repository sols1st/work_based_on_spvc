import torch
import os
from cGAN.cGAN_common import Settings, DCGANLoss, LSLoss, WLossGP, HingeLoss, train
from cGAN.taxi_models_and_data import AebsGConv, AebsDConvSpectral, taxi_input, gen_aebs_images


def main():
    # Latent dimension setting
    latent_dim = 4
    # batch_sizes = [128, 256, 1024]
    batch_sizes = [16]
    # all_epochs = [500, 750, 1000]
    all_epochs = [750]
    learning_rates = [1e-4]
    # Loss function selection
    # losses = [DCGANLoss(), LSLoss(), WLossGP(), HingeLoss()]
    losses = [LSLoss()]
    # Loss function names
    # losses_name = ["BCE", "LS", "WLoss", "Hinge"]
    losses_name = ["LS"]
    # dirs = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Loop over all hyperparameter combinations
    for loss, loss_name in zip(losses, losses_name):
        for batch_size, n_epoch in zip(batch_sizes, all_epochs):
            for lr in learning_rates:
                output_dir = f"{loss_name}_BS{batch_size}_LR{lr:.0e}"
                # Get the current file directory
                current_dir = os.path.dirname(os.path.abspath(__file__))
                # Construct full path
                output_dir = os.path.join(current_dir, output_dir)
                # dirs.append(output_dir)
                
                print(f"\n=======================================================")
                print(f"Starting training for: {output_dir}")
                print(f"=======================================================")

                # Define optimizer configuration dictionaries
                optimizer_config_G = {'lr': lr, 'betas': (0.5, 0.9)}
                optimizer_config_D = {'lr': lr, 'betas': (0.5, 0.9)}
                
                settings = Settings(
                    G=AebsGConv,
                    D=AebsDConvSpectral,
                    epochs=n_epoch,
                    batch_size=batch_size,
                    rand_input=lambda s: taxi_input(s, s.device, s.ranges),
                    loss=loss,
                    img_fun=lambda s: gen_aebs_images(s, device, "./Aebs/data/Downsampled.h5"),
                    nclasses=1,
                    latent_dim=latent_dim,
                    verbose_freq=10,
                    # **Pass optimizer configs directly**
                    optD=optimizer_config_D,
                    optG=optimizer_config_G,
                    output_dir=output_dir,
                    device=device
                )

                # Start training
                G, D, Ghist, Dhist = train(settings)
                
                print(f"Training for {output_dir} finished.")

                # --- Save models and histories (code unchanged) ---
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                G_path = os.path.join(output_dir, f"conv_generator_ld{latent_dim}.pth")
                D_path = os.path.join(output_dir, f"conv_discriminator_ld{latent_dim}.pth")
                
                torch.save(G.cpu().state_dict(), G_path)
                torch.save(D.cpu().state_dict(), D_path)
                print(f"Saved Generator state_dict to {G_path}")
                print(f"Saved Discriminator state_dict to {D_path}")
                
                Dhist_path = os.path.join(output_dir, f"d_hist_ld{latent_dim}.pt")
                Ghist_path = os.path.join(output_dir, f"g_hist_ld{latent_dim}.pt")
                
                torch.save(Dhist, Dhist_path)
                torch.save(Ghist, Ghist_path)
                print(f"Saved D/G history to {output_dir}")
                
                # torch.save(dirs, "dirs.pt")

    print("\nAll training runs completed.")
    # print(f"Total configurations run: {len(dirs)}")

if __name__ == '__main__':
    main()
