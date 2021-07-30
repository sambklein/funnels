from nflows import transforms
from nflows import flows
import nflows
import torch
import torch.nn as nn

from surVAE.models.flows import get_transform
from surVAE.models.nn.MLPs import dense_net


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

    def __init__(self, features, hidden_features, num_blocks=2, num_bins=10, **kwargs):
        super(SurNSF, self).__init__()

        self.features = features

        inp_dim = 1
        nodes = 64
        num_blocks = 2
        nstack = 2
        tails = 'linear'
        tail_bound = 4
        num_bins = 10

        transform_list = []
        for i in range(nstack):
            transform_list += [
                transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(inp_dim, nodes,
                                                                                   num_blocks=num_blocks,
                                                                                   tail_bound=tail_bound,
                                                                                   num_bins=num_bins,
                                                                                   tails=tails,
                                                                                   context_features=features - 1)]

            transform_list += [transforms.ReversePermutation(inp_dim)]

        transform = transforms.CompositeTransform(transform_list[:-1])
        base_dist = nflows.distributions.StandardNormal([inp_dim])
        self.one_dim_flow = flows.Flow(transform, base_dist)

        self.transform = get_transform(inp_dim=features - 1)

    def forward(self, inputs, context=None):
        input_dropped = inputs[:, 0].view(-1, 1)
        output = self.transform.forward(inputs[:, 1:].view(-1, self.features - 1), context=input_dropped)[0]
        likelihood_contribution = self.one_dim_flow.log_prob(input_dropped, context=output)
        return output, likelihood_contribution

    def inverse(self, inputs, context=None):
        input_dropped = self.one_dim_flow.sample(1, context=inputs).squeeze().view(-1, 1)
        input_mapped = self.transform.inverse(inputs, context=input_dropped)[0]
        likelihood_contribution = self.one_dim_flow.log_prob(input_dropped.view(-1, 1), context=inputs)
        f_return = torch.cat((input_dropped, input_mapped), 1)
        return f_return, likelihood_contribution
