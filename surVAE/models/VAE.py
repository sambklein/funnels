import torch
from torch import nn

from surVAE.models.nn.MLPs import dense_net


class transform(nn.Module):

    def __init__(self, encoder, decoder, latent_size):
        super(transform, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_size = latent_size

    def forward(self, data):
        z_mean, z_log_sigma = self.encoder(data).split(self.latent_size, dim=1)
        epsilon = torch.normal(mean=torch.zeros_like(z_mean), std=torch.ones_like(z_mean))
        return z_mean + torch.exp(z_log_sigma) * epsilon

    def inverse(self, data):
        return self.decoder(data), torch.zeros(data.shape[0])


class VAE(nn.Module):

    def __init__(self, input_dim, latent_size, layers, activation=torch.relu):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.encoder = dense_net(input_dim, latent_size * 2, layers=layers, int_activ=activation)
        self.decoder = dense_net(latent_size, input_dim, layers=layers, int_activ=activation)
        self.recon_loss = nn.MSELoss()
        self._transform = transform(self.encoder, self.decoder, latent_size)

    def transform_to_noise(self, data):
        return self._transform(data)

    def log_prob(self, data):
        z_mean, z_log_sigma = self.encoder(data).split(self.latent_size, dim=1)
        kl_loss = 1 + z_log_sigma - torch.square(z_mean) - torch.exp(z_log_sigma)
        kl_loss = torch.sum(kl_loss, axis=-1)
        kl_loss = -0.5 * kl_loss

        x_prime = self.sample([z_mean, z_log_sigma])
        l_recon = self.recon_loss(data, x_prime)

        return - kl_loss - l_recon

    def forward(self, *args):
        raise RuntimeError("This implementation is not intended for use in this fashion.")

    def base_dist_sample(self, encoding):
        z_mean, z_log_sigma = encoding
        epsilon = torch.normal(mean=torch.zeros_like(z_mean), std=torch.ones_like(z_mean))
        return z_mean + torch.exp(z_log_sigma) * epsilon

    def sample(self, encoding):
        return self.decoder(self.base_dist_sample(encoding))
