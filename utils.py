import os
import torch
import matplotlib.pyplot as plt 

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def plot_image_grid(original_images, reconstructed_images, grid_size=5, save_plot=False):
    """
    Plots a grid of images with original images on the top row and reconstructed images on the bottom row.
    """
    fig, axs = plt.subplots(2, grid_size, figsize=(15, 6))
    axs = axs.flatten()
    
    # Plot original images
    for i in range(grid_size):
        ax = axs[i]
        ax.imshow(original_images[i].cpu().numpy().reshape(28, 28), cmap='gray')
        ax.axis('off')
        ax.set_title(f'Original {i+1}')
    
    # Plot reconstructed images
    for i in range(grid_size):
        ax = axs[grid_size + i]
        ax.imshow(reconstructed_images[i].cpu().numpy().reshape(28, 28), cmap='gray')
        ax.axis('off')
        ax.set_title(f'Reconstructed {i+1}')
    
    plt.tight_layout()

    if save_plot:
        if not os.path.exists('./plots'):
            os.mkdir('./plots')
        plt.savefig('./plots/samples.png')

    plt.show()

def reconstruct(model, test_loader, num_samples, save_plot, device):
    model.eval()

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.view(x.size(0), -1)
            x = x.to(device)
            _, _, x_hat = model(x)
            
            # Plot the first batch of original and reconstructed images
            plot_image_grid(x, x_hat, num_samples, save_plot)
            
            break

def infer_vae(model, num_samples=5, latent_dim=300, save_samples=False):
    """
    Plots a grid of images with noise vectors on the top row and generated images on the bottom row.
    """
    model.eval()
    noise_vectors = torch.randn(num_samples, latent_dim)


    with torch.no_grad():
        generated_images = model.Decoder(noise_vectors)

    fig, axs = plt.subplots(1, num_samples, figsize=(15, 6))
    axs = axs.flatten()

    for i in range(num_samples):
        ax = axs[i]
        gen_image = generated_images[i].cpu().numpy().reshape(28, 28)  # Adjust if necessary
        ax.imshow(gen_image, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Generated {i+1}')
    
    plt.tight_layout()

    if save_samples:
        if not os.path.exists('./images'):
            os.mkdir('./images')
        plt.savefig('./images/inference_samples.png')

    plt.show()
    