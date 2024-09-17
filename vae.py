import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Enocder learns the mean and variance of the proxy probability distribution Q(Z|X) which 
    gives us a better estimate of the true underlying P(Z).
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.hidden_fc = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_fc = nn.Linear(hidden_dim, latent_dim)
        self.var_fc = nn.Linear(hidden_dim, latent_dim)
        
        self.act = nn.LeakyReLU(0.15)
        
    def forward(self, x):
        """
        Given an input, pass it through the encoder to fetch the mean and var of Q(z|x)
        """
        hidden = self.input_fc(x)
        hidden = self.act(hidden)
        
        hidden = self.hidden_fc(hidden)
        hidden = self.act(hidden)
        
        mean = self.mean_fc(hidden)
        log_var = self.var_fc(hidden)
        
        return mean, log_var
    
class Decoder(nn.Module):
    """
    Decoder maps a sampled latent from Q(Z|X) to the output space P(X|Z).
    """
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        
        self.input_fc = nn.Linear(latent_dim, hidden_dim)
        self.hidden_fc = nn.Linear(hidden_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        
        self.act = nn.LeakyReLU(0.15)
        
    def forward(self, z):
        """
        We have the latent vector z which now needs to be projected to the output space.
        """
        hidden = self.input_fc(z)
        hidden = self.act(hidden)
        
        hidden = self.hidden_fc(hidden)
        hidden = self.act(hidden)
        
        output = self.output_fc(hidden)
        output = torch.sigmoid(output)
        
        return output        
    
class VAE(nn.Module):
    """
    We take a input and pass it through the encoder to estimate the variables of Q(Z|X).
    We then sample a gaussian vector and shift it to the space of Q(Z|X). 
    This shifted gaussian is passed through the Decoder to reconstruct the input.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, device):
        super(VAE, self).__init__()
        
        self.Encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.Decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.device = device

    def reparameterise(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        # Now we shift the noise vector to the latent space 
        z = mean + (var * epsilon)
        
        return z
    
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        var = torch.exp(0.5 * log_var) # convert log_var to var
        
        z = self.reparameterise(mean, var) 
        
        output = self.Decoder(z)
        
        return mean, log_var, output
    