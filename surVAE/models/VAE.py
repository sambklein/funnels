import torch
from nflows.distributions import StandardNormal
from torch import nn
import numpy as np

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

    def __init__(self, input_dim, latent_size, layers, activation=torch.relu, encoder=None, decoder=None,
                 preprocess=None):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.preprocess = preprocess
        if encoder is None:
            self.encoder = dense_net(input_dim, latent_size * 2, layers=layers, int_activ=activation)
        else:
            self.encoder = encoder
        if decoder is None:
            self.decoder = dense_net(latent_size, input_dim, layers=layers, int_activ=activation)
        else:
            self.decoder = decoder
        self.recon_loss = nn.MSELoss(reduce=False)
        self._transform = transform(self.encoder, self.decoder, latent_size)

        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def transform_to_noise(self, data):
        return self._transform(data)

    def set_preprocessing(self, preprocess):
        self.preprocess = preprocess

    def likelihood(self, x_prime, log_scale, x):
        scale = log_scale.exp()
        return torch.distributions.Normal(x_prime, scale).log_prob(x).view(x.shape[0], -1).mean(-1)

    def log_prob(self, data):
        if self.preprocess is not None:
            data, prep_likelihood = self.preprocess(data)
        else:
            prep_likelihood = 0
        z_mean, z_log_sigma = self.encoder(data).split(self.latent_size, dim=1)
        kl_loss = 1 + z_log_sigma - z_mean.square() - z_log_sigma.exp()
        kl_loss = torch.sum(kl_loss, dim=1)
        kl_loss = -0.5 * kl_loss

        x_prime = self.sample_dec([z_mean, (z_log_sigma / 2).exp()])
        per_sample_likelihood = self.likelihood(x_prime, self.log_scale, data + 0.5)

        # TODO: only add if self.training is False?
        return per_sample_likelihood - kl_loss + prep_likelihood 

    def forward(self, *args):
        raise RuntimeError("This implementation is not intended for use in this fashion.")

    def base_dist_sample(self, encoding):
        z_mean, z_log_sigma = encoding
        epsilon = torch.normal(mean=torch.zeros_like(z_mean), std=torch.ones_like(z_mean))
        return z_mean + torch.exp(z_log_sigma) * epsilon

    def decode(self, sample):
        return torch.distributions.Normal(self.decoder(sample), self.log_scale).sample([1])[0]

    def sample_dec(self, encoding):
        # return self.decoder(self.base_dist_sample(encoding))
        return self.decoder(self.base_dist_sample(encoding))

    def sample(self, num):
        # return self.decoder(torch.normal(mean=torch.zeros((num, self.latent_size)),
        #                                  std=torch.ones((num, self.latent_size))))
        return self.decoder(StandardNormal([self.latent_size]).sample(num))
