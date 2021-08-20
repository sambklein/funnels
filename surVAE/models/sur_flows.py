from nflows import transforms
from nflows import flows
import nflows
import torch
import torch.nn as nn

from surVAE.models.flows import get_transform
from surVAE.models.nn.MLPs import dense_net


def not_an_image_exception():
    raise Exception('Convolutions only work on images, need a channel dimension even if there is only one channel.')


def convolving_not_possible():
    # TODO: better error message...
    raise RuntimeError('Cannot properly reshape images of this size for this convolution.')


class get_net(nn.Module):

    def __init__(self, features, hidden_features, num_blocks, output_multiplier):
        super(get_net, self).__init__()
        self.feature_list = [1, 1]
        self.makers = nn.ModuleList(
            [dense_net(self.feature_list[i], output_multiplier, layers=[hidden_features] * num_blocks) for i in
             range(features)])

    def forward(self, data, context=None):
        splines = []
        for i, function in enumerate(self.makers):
            # All outputs are a function of the dimension which is dropped, garuantees right invertibility.
            splines += [function(data[:, 0].view(-1, 1))]
        return torch.cat(splines, 1)


class SurNSF(nflows.transforms.Transform):

    def __init__(self, features, hidden_features, num_blocks=2, num_bins=10, tail_bound=4., tails='linear', **kwargs):
        super(SurNSF, self).__init__()

        self.features = features

        inp_dim = 1
        nstack = 2

        transform_list = []
        for i in range(nstack):
            transform_list += [
                transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(inp_dim, hidden_features,
                                                                                   num_blocks=num_blocks,
                                                                                   tail_bound=tail_bound,
                                                                                   num_bins=num_bins,
                                                                                   tails=tails,
                                                                                   context_features=features - 1)]

            transform_list += [transforms.ReversePermutation(inp_dim)]

        flow_transform = transforms.CompositeTransform(transform_list[:-1])
        base_dist = nflows.distributions.StandardNormal([inp_dim])
        # flow_transform = get_transform(inp_dim=1, context_features=features - 1)
        # base_dist = nflows.distributions.StandardNormal([1])
        self.one_dim_flow = flows.Flow(flow_transform, base_dist)

        self.transform = get_transform(inp_dim=features - 1, context_features=1, tails='linear')

    def forward(self, inputs, context=None):
        input_dropped = inputs[:, 0].view(-1, 1)
        output, output_likelihood = self.transform.forward(inputs[:, 1:].view(-1, self.features - 1),
                                                           context=input_dropped)
        likelihood_contribution = self.one_dim_flow.log_prob(input_dropped, context=output)
        return output, likelihood_contribution + output_likelihood
        # return output, torch.exp(-output_likelihood) * (likelihood_contribution + output_likelihood)

    def inverse(self, inputs, context=None):
        input_dropped = self.one_dim_flow.sample(1, context=inputs).squeeze().view(-1, 1)
        input_mapped = self.transform.inverse(inputs, context=input_dropped)[0]
        # TODO: this is wrong...
        likelihood_contribution = self.one_dim_flow.log_prob(input_dropped.view(-1, 1), context=inputs)
        f_return = torch.cat((input_dropped, input_mapped), 1)
        return f_return, likelihood_contribution


class NByOneConv(nflows.transforms.Transform):
    """
    An N x 1 funnel convolution with the stride fixed to N.
    """

    def __init__(self, num_channels, width=2, hidden_channels=1, hidden_features=128, num_blocks=2,
                 num_bins=10, tail_bound=4., tails='linear', **kwargs):
        super(NByOneConv, self).__init__()

        self.num_channels = num_channels
        self.width = width
        # TODO: Need to make a transformer and a flow for each layer, these could be instances of the SurNSF if it was more general
        # TODO: need to implement hidden_channels kwargs, what does hidden channels even mean?
        inp_dim = width - 1
        # TODO: this is probably equivalent to steps per level in GLOW
        nstack = 2

        transform_list = []
        for i in range(nstack):
            transform_list += [
                transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(inp_dim, hidden_features,
                                                                                   num_blocks=num_blocks,
                                                                                   tail_bound=tail_bound,
                                                                                   num_bins=num_bins,
                                                                                   tails=tails,
                                                                                   context_features=1)]

            transform_list += [transforms.ReversePermutation(inp_dim)]

        flow_transform = transforms.CompositeTransform(transform_list[:-1])
        base_dist = nflows.distributions.StandardNormal([inp_dim])
        self.one_dim_flow = flows.Flow(flow_transform, base_dist)
        # TODO: this should be a proper transformation
        self.component_mixer = transforms.CompositeTransform([
            transforms.LULinear(width),
            transforms.LULinear(width)
        ])
        self.transform = get_transform(inp_dim=1, context_features=width - 1, tails='linear')

    def forward(self, inputs, context=None):
        if inputs.dim() != 4:
            not_an_image_exception()
        batch_size, c, h, w = inputs.shape
        if h * w % self.width:
            convolving_not_possible()
        new_w = int(w / self.width)
        output = torch.zeros((c, batch_size, h, new_w))
        likelihood_contribution = 0
        inputs = inputs.permute(1, 0, 2, 3)
        for i, channel in enumerate(inputs):
            channel, mixer_contrib = self.component_mixer.forward(channel.reshape(-1, self.width))
            faux_put, output_likelihood = self.transform.forward(channel[:, :1], context=channel[:, 1:])
            output[i] = faux_put.view(batch_size, h, new_w)
            likelihood_contribution += (
                    self.one_dim_flow.log_prob(channel[:, 1:], context=faux_put) + output_likelihood + mixer_contrib
            ).view(batch_size, -1).mean(-1)
        return output.permute(1, 0, 2, 3), likelihood_contribution

    def inverse(self, inputs, context=None):

        if inputs.dim() != 4:
            not_an_image_exception()
        batch_size, c, h, w = inputs.shape
        new_w = int(w * self.width)
        output = torch.zeros((c, batch_size, h, new_w))
        likelihood_contribution = 0
        inputs = inputs.permute(1, 0, 2, 3)
        for i, channel in enumerate(inputs):
            channel = channel.reshape(-1, 1)
            input_dropped = self.one_dim_flow.sample(1, context=channel).squeeze().view(-1, 1)
            input_mapped, output_likelihood = self.transform.inverse(channel, context=input_dropped)
            o_p, o_p_contrib = self.component_mixer.inverse(torch.cat([input_mapped, input_dropped], 1))
            likelihood_contribution += (
                    self.one_dim_flow.log_prob(input_dropped, context=input_mapped) + output_likelihood + o_p_contrib
            ).view(batch_size, -1).mean(-1)
            output[i] = o_p.view(batch_size, h, new_w)
        return output.permute(1, 0, 2, 3), likelihood_contribution


class MakeAnImage(nflows.transforms.Transform):

    def forward(self, inputs, context=None):
        batch_size, width = inputs.shape
        return inputs.view(batch_size, 1, 1, width), torch.zeros(batch_size)

    def inverse(self, inputs, context=None):
        batch_size, _, _, width = inputs.shape
        return inputs.view(batch_size, width), torch.zeros(batch_size)

class UnMakeAnImage(MakeAnImage):
    def forward(self, inputs, context=None):
        return super(UnMakeAnImage, self).inverse(inputs)

    def inverse(self, inputs, context=None):
        return super(UnMakeAnImage, self).forward(inputs)
