import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vae import VAE 
from utils import infer_vae

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dims', type=int, default=784)
    parser.add_argument('--h_dims', type=int, default=500)
    parser.add_argument('--l_dims', type=int, default=300)
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--save_samples', type=bool, default=False)

    args = parser.parse_args()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        transform=transform,
        download=False
    )
    dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # load the model saved at model_path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = VAE(
        args.in_dims,
        args.h_dims,
        args.l_dims,
        device
    )
    model.load_state_dict(torch.load(args.model_path, weights_only=True))

    # Infer the model 
    infer_vae(model, args.num_samples, args.l_dims, args.save_samples)


