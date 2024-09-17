import os
import argparse
import matplotlib.pyplot as plt

from utils import set_seed, count_params
from vae import VAE

from tqdm import tqdm
import torch 
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def loss_fn(x, x_hat, mean, log_var):
    """
    This form of the KLD occurs when we are measuring the KLD between two Gaussian Distributions.
    """
    reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    
    return reconstruction_loss + KLD_loss

def train_model(model, train_dataloader, n_epochs, optimizer, device):
    train_losses = []
    
    model.to(device)
    
    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0
        # Training loop
        for X, _ in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            X = X.to(device)
            X = X.view(X.shape[0], -1)
            
            optimizer.zero_grad()
            mean, log_var, x_hat = model(X)
            
            loss = loss_fn(X, x_hat, mean, log_var)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
       
        print(f'\nEpoch: {epoch+1}/{n_epochs} -- Train Loss: {avg_train_loss:.4f}')
        print('-'*50)

    return train_losses

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--in_dims', type=int, default=784)
    parser.add_argument('--h_dims', type=int, default=500)
    parser.add_argument('--l_dims', type=int, default=300)
    parser.add_argument('--download_mnist', type=bool, default=True)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--save_plots', type=bool, default=False)

    set_seed()
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=args.download_mnist
    )


    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    model = VAE(
        args.in_dims,
        args.h_dims,
        args.l_dims,
        device
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print('-' * 64)
    print(f"Training VAE with {count_params(model) * 1e-6: .4f}M parameters on {str(device).upper()} for {args.epochs} epochs....")
    print('-' * 64 + '\n')

    train_losses = train_model(model, train_dataloader, args.epochs, optimizer, device)

    if args.save_model:
        if not os.path.exists('./models'):
            os.mkdir('./models')
        torch.save(model.state_dict(), './models/vae.pth')

    plt.figure(figsize=(8,6))
    plt.grid(True)
    plt.plot(train_losses, color='orange', linewidth=2)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('BCE Loss')

    if args.save_plots:
        if not os.path.exists('./images'):
            os.mkdir('./images')
        plt.savefig('images/loss.png')

    plt.show()

    print(f'\nTraining Complete!')

