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
        return torch.distributions.Normal(x_prime, scale).log_prob(x).view(x.shape[0], -1).sum(-1)

    def kl_divergence(self, z, mu, std):
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def log_prob(self, data):
        if self.preprocess is not None:
            data, prep_likelihood = self.preprocess(data)
        else:
            prep_likelihood = 0
        # z_mean, z_log_sigma = self.encoder(data).split(self.latent_size, dim=1)
        # kl_loss = 1 + z_log_sigma - z_mean.square() - z_log_sigma.exp()
        # kl_loss = torch.sum(kl_loss, dim=1)
        # kl_loss = -0.5 * kl_loss
        #
        # x_prime = self.sample_dec([z_mean, (z_log_sigma / 2).exp()])
        # per_sample_likelihood = self.likelihood(x_prime, self.log_scale, data + 0.5)
        #
        # # TODO: only add if self.training is False?
        # return per_sample_likelihood - kl_loss + prep_likelihood
        z_mean, z_log_sigma = self.encoder(data).split(self.latent_size, dim=1)
        std = (z_log_sigma / 2).exp()
        q = torch.distributions.Normal(z_mean, std)
        z = q.rsample()
        kl_loss = self.kl_divergence(z, z_mean, std)

        x_prime = self.decoder(z)
        per_sample_likelihood = self.likelihood(x_prime, self.log_scale, data + 0.5)

        # TODO: only add if self.training is False?
        return per_sample_likelihood - kl_loss + prep_likelihood

    def autoencode(self, data):
        data, prep_likelihood = self.preprocess(data)
        z_mean, z_log_sigma = self.encoder(data).split(self.latent_size, dim=1)
        return self.sample_dec([z_mean, (z_log_sigma / 2).exp()])

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

    def sample(self, num, temperature=1):
        return self.decoder(StandardNormal([self.latent_size]).sample(num) * temperature)
