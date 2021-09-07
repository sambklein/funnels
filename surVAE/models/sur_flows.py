from nflows import transforms
from nflows import flows
import nflows
import torch
import torch.nn as nn
from nflows.transforms import splines
from nflows.transforms.coupling import CouplingTransform

from surVAE.models.GLOW import create_flow
from surVAE.models.flows import get_transform, coupling_spline
from surVAE.models.nn.MLPs import dense_net
import numpy as np


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

    def __init__(self, features, hidden_features, num_blocks=2, num_bins=10, tail_bound=4., tails='linear', spline=True,
                 **kwargs):
        super(SurNSF, self).__init__()

        self.features = features
        inp_dim = 1
        nstack = 2
        self.flow_transform = get_transform(inp_dim=inp_dim,
                                            nodes=hidden_features,
                                            num_blocks=num_blocks,
                                            tails=tails,
                                            num_bins=num_bins,
                                            tail_bound=tail_bound,
                                            nstack=nstack,
                                            context_features=features - 1,
                                            spline=spline)
        base_dist = nflows.distributions.StandardNormal([inp_dim])
        self.one_dim_flow = flows.Flow(self.flow_transform, base_dist)

        self.transform = get_transform(inp_dim=features - 1, context_features=1, tails='linear', spline=spline)

    def forward(self, inputs, context=None):
        input_dropped = inputs[:, 0].view(-1, 1)
        output, jacobian = self.transform.forward(inputs[:, 1:].view(-1, self.features - 1),
                                                  context=input_dropped)
        likelihood_contribution = self.one_dim_flow.log_prob(input_dropped, context=output)
        # return output, likelihood_contribution + output_likelihood
        return output, jacobian + likelihood_contribution

    def inverse(self, inputs, context=None):
        input_dropped = self.one_dim_flow.sample(1, context=inputs).squeeze().view(-1, 1)
        input_mapped = self.transform.inverse(inputs, context=input_dropped)[0]
        # TODO: this is wrong...
        likelihood_contribution = self.one_dim_flow.log_prob(input_dropped.view(-1, 1), context=inputs)
        f_return = torch.cat((input_dropped, input_mapped), 1)
        return f_return, likelihood_contribution


class IdentityTransform(nflows.transforms.Transform):
    """
    An N x 1 funnel convolution with the stride fixed to N.
    """

    def forward(self, inputs, context=None):
        return inputs, torch.zeros(inputs.shape[0]).to(inputs.device)

    def inverse(self, inputs, context=None):
        return inputs, torch.zeros(inputs.shape[0]).to(inputs.device)


# TODO: refactor this to be a single channel convolution, then build a many channel version on top
class NByOneConv(nflows.transforms.Transform):
    """
    An N x 1 funnel convolution with the stride fixed to N.
    """

    def __init__(self, num_channels, width=2, hidden_channels=1, hidden_features=128, num_blocks=2,
                 num_bins=10, tail_bound=1., tails='linear', nstack=10, **kwargs):
        super(NByOneConv, self).__init__()

        # nstack = int(nstack / 2)
        # self.component_mixer = get_transform(inp_dim=width, tails='linear', context_features=None,
        #                                      tail_bound=tail_bound, nstack=nstack, nodes=hidden_features)
        self.component_mixer = [IdentityTransform() for _ in range(num_channels)]

        self.num_channels = num_channels
        self.width = width
        self.ind_to_drop = width // 2
        self.ind_mask = np.arange(self.width) != self.ind_to_drop

        spline = 0
        self.one_dim_flow = [
            self.get_one_dim_flow(nstack, hidden_features, num_blocks, tail_bound, num_bins, tails, width,
                                  spline=spline) for _ in range(num_channels)]

        def maker(input_dim, output_dim):
            return dense_net(input_dim, output_dim, layers=[hidden_features] * num_blocks)

        mx = [0] + [1] * (self.width - 1)
        self.transform = [coupling_spline(width, maker, tails='linear', num_bins=num_bins, mask=mx,
                                          tail_bound=tail_bound, nstack=nstack) for _ in range(num_channels)]
        # self.transform = [get_transform(inp_dim=width, context_features=None, tails='linear', nodes=hidden_features,
        #                                 tail_bound=tail_bound, nstack=nstack) for _ in range(num_channels)]
        # self.transform = [get_transform(inp_dim=width - 1, context_features=1, tails='linear', nodes=hidden_features,
        #                                 tail_bound=tail_bound, nstack=nstack, spline=1) for _ in
        #                   range(num_channels)]

    def get_one_dim_flow(self, nstack, hidden_features, num_blocks, tail_bound, num_bins, tails, width, spline=True):
        inp_dim = 1
        transform_list = []
        tail_bound = 4.
        for i in range(nstack):
            if spline:
                transform_list += [
                    transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(inp_dim, hidden_features,
                                                                                       num_blocks=num_blocks,
                                                                                       tail_bound=tail_bound,
                                                                                       num_bins=num_bins,
                                                                                       tails=tails,
                                                                                       context_features=width - 1)]
            else:
                transform_list += [transforms.MaskedAffineAutoregressiveTransform(inp_dim, 128, num_blocks=4,
                                                                                  context_features=width - 1)]

            # transform_list += [transforms.BatchNorm(inp_dim)]

            transform_list += [transforms.ReversePermutation(inp_dim)]

        flow_transform = transforms.CompositeTransform(transform_list[:-1])
        base_dist = nflows.distributions.StandardNormal([inp_dim])
        # base_dist = nflows.distributions.uniform.BoxUniform(-tail_bound * torch.ones(inp_dim),
        #                                                     tail_bound * torch.ones(inp_dim))
        return flows.Flow(flow_transform, base_dist)

    def _get_context(self, z):
        return z
        # return torch.cat((z, z, z), 1)

    def _unpadded_forward(self, inputs, context=None):
        if inputs.dim() != 4:
            not_an_image_exception()
        batch_size, c, h, w = inputs.shape
        # if h * w % self.width:
        #     convolving_not_possible()
        # new_w = w - int(w / self.width)
        new_w = w - w // self.width
        output = torch.zeros((c, batch_size, h, new_w))
        likelihood_contribution = torch.zeros((batch_size,))  # 0
        inputs = inputs.permute(1, 0, 2, 3)
        for i, channel in enumerate(inputs):
            channel, mixer_contrib = self.component_mixer[i].forward(channel.reshape(-1, self.width))
            inputs_to_drop = channel[:, self.ind_to_drop].view(-1, 1)
            inputs_to_map = channel[:, self.ind_mask]
            # faux_put, output_likelihood = self.transform[i].forward(inputs_to_map, context=inputs_to_drop)
            # output[i] = faux_put.view(batch_size, h, new_w)
            faux_put, jacobian = self.transform[i].forward(channel)
            faux_put = faux_put[:, 1:]
            output[i] = faux_put.reshape(batch_size, h, new_w)
            # n_in = torch.cat((inputs_to_drop, inputs_to_map), 1)
            # TODO: is this the correct likelihood?
            # faux_put, output_likelihood = self.transform[i].forward(n_in)
            # faux_put = faux_put[:, 1:]
            output[i] = faux_put.reshape(batch_size, h, new_w)
            one_d_context = faux_put
            likelihood_contribution += (
                    self.one_dim_flow[i].log_prob(inputs_to_drop, context=one_d_context) + jacobian
                    + mixer_contrib
            ).view(batch_size, -1).mean(-1)
        return output.permute(1, 0, 2, 3), likelihood_contribution

    def forward(self, inputs, context=None):
        batch_size, c, h, w = inputs.shape
        n_w_to_take = w - w % self.width
        baggage = inputs[..., n_w_to_take:]
        output, likelihood = self._unpadded_forward(inputs[..., :n_w_to_take], context=context)
        # output = torch.cat((output, baggage), -1)
        return output, likelihood

    def _unpadded_inverse(self, inputs, context=None):

        if inputs.dim() != 4:
            not_an_image_exception()
        batch_size, c, h, w = inputs.shape
        new_w = w + w // (self.width - 1)
        output = torch.zeros((c, batch_size, h, new_w))
        likelihood_contribution = 0
        inputs = inputs.permute(1, 0, 2, 3)
        # TODO: this is only going to be single channel!
        for i, channel in enumerate(inputs):
            channel = channel.reshape(-1, self.width - 1)
            output_temp = torch.zeros((channel.shape[0], channel.shape[1] + 1))
            one_d_context = channel
            input_dropped = self.one_dim_flow[i].sample(1, context=one_d_context).squeeze().view(-1, 1)

            # input_mapped, output_likelihood = self.transform[i].inverse(channel, context=input_dropped)
            # output_temp[:, self.ind_to_drop] = input_dropped.view(-1)
            # output_temp[:, self.ind_mask] = input_mapped
            # un_mixed_output = torch.cat([input_mapped, input_dropped], 1)
            input_mapped, output_likelihood = self.transform[i].inverse(torch.cat((input_dropped, channel), 1))
            output_temp[:, self.ind_to_drop] = input_mapped[:, 0].view(-1)
            input_mapped = input_mapped[:, 1:]
            output_temp[:, self.ind_mask] = input_mapped

            o_p, o_p_contrib = self.component_mixer[i].inverse(output_temp)
            one_d_context = input_mapped
            likelihood_contribution += (
                    self.one_dim_flow[i].log_prob(input_dropped,
                                                  context=one_d_context) + output_likelihood + o_p_contrib
            ).view(batch_size, -1).mean(-1)
            output[i] = o_p.view(batch_size, h, new_w)
        return output.permute(1, 0, 2, 3), likelihood_contribution

    def inverse(self, inputs, context=None):
        batch_size, c, h, w = inputs.shape
        # TODO: you shouldn't be setting this by hand, but its the end of the day, so...
        n_w_to_take = 0  # w - w % (self.width - 1)
        # baggage = inputs[..., n_w_to_take:]
        baggage = torch.zeros_like(inputs)[..., :n_w_to_take]
        output, likelihood = self._unpadded_inverse(inputs, context=context)
        output = torch.cat((output, baggage), -1)
        return output, likelihood


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


class make_generator(flows.Flow):

    def __init__(self, dropped_entries_shape, context_shape):
        # TODO: this could/should be a proper flow using half the image with coupling layers etc...
        """
        :param dropped_entries_shape: the shape of the data that needs to be sampled and evaluated (for likelihood)
        :param context_shape: the shape of the data that will be passed as context
        :return: a flow capable of generating, and evaluating the likelihood, data of shape dropped_entries_shape given
                 data of shape context_shape as context.
        """
        self.dropped_entries_shape = dropped_entries_shape
        self.context_shape = context_shape
        self.input_size = int(np.prod(dropped_entries_shape))
        self.context_size = int(np.prod(context_shape))
        self.tail_bound = 4.
        transform = get_transform(inp_dim=self.input_size, tails='linear', context_features=self.context_size,
                                  tail_bound=self.tail_bound, nstack=5, nodes=256, spline=True)
        base_dist = nflows.distributions.StandardNormal([self.input_size])
        # base_dist = nflows.distributions.uniform.BoxUniform(-self.tail_bound * torch.ones(self.input_size),
        #                                                     self.tail_bound * torch.ones(self.input_size))
        super(make_generator, self).__init__(transform, base_dist)

    def _log_prob(self, inputs, context):
        inputs = inputs.view(-1, self.input_size)
        context = context.view(-1, self.context_size)
        return super(make_generator, self)._log_prob(inputs, context)

    def _sample(self, num_samples, context):
        context = context.view(-1, self.context_size)
        return super(make_generator, self)._sample(num_samples, context).view(-1, *self.dropped_entries_shape)


class SurVaeCoupling(CouplingTransform):

    def __init__(self, dropped_entries_shape, context_shape, mask, transform_net_create_fn, *args,
                 unconditional_transform=None, **kwargs):
        super(SurVaeCoupling, self).__init__(mask, transform_net_create_fn,
                                             unconditional_transform=unconditional_transform)
        self.drop_mask = torch.ones(self.features, dtype=torch.bool)
        feature_to_drop = self.identity_features[-1]
        self.drop_mask[feature_to_drop] = 0
        _, self.sorted_indices = torch.sort(
            torch.cat((torch.arange(self.features)[self.drop_mask], torch.tensor([feature_to_drop]))))

        # A flow that can generate one dropped index of the input data given the other data entries
        self.generator = make_generator(dropped_entries_shape, context_shape)

    def forward(self, inputs, context=None):
        faux_output, log_contr = super().forward(inputs, context=context)
        output = faux_output[:, self.drop_mask, ...]
        likelihood = self.generator.log_prob(faux_output[:, ~self.drop_mask, ...], context=output)
        # return output, log_contr + likelihood
        return output, log_contr + likelihood

    def inverse(self, inputs, context=None):
        input_dropped = self.generator.sample(1, context=inputs)
        likelihood = self.generator.log_prob(input_dropped, context=inputs)
        inputs = torch.cat((inputs, input_dropped), 1)[:, self.sorted_indices, ...]
        output, log_contr = super().inverse(inputs, context=context)
        return output, log_contr + likelihood


class wrap_rqct(transforms.PiecewiseRationalQuadraticCouplingTransform):

    def __init__(
            self,
            *args,
            mask=None,
            transform_net_create_fn=None,
            num_bins=10,
            tails=None,
            tail_bound=1.0,
            apply_unconditional_transform=False,
            img_shape=None,
            min_bin_width=splines.rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
            min_bin_height=splines.rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
            min_derivative=splines.rational_quadratic.DEFAULT_MIN_DERIVATIVE,
            **kwargs
    ):
        super(wrap_rqct, self).__init__(
            self,
            mask,
            transform_net_create_fn,
            # TODO: why isn't this accepted?
            # num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            apply_unconditional_transform=apply_unconditional_transform,
            img_shape=img_shape,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative
        )


# class surRqNSF(wrap_rqct, SurVaeCoupling):

# def __init__(self, *args, **kwargs):
#     super(surRqNSF, self).__init__(*args, **kwargs)

# def __init__(self, dropped_entries_shape, context_shape, *args, **kwargs):
# transforms.PiecewiseRationalQuadraticCouplingTransform.__init__(self, *args, **kwargs)
# SurVaeCoupling.__init__(self, dropped_entries_shape, context_shape, *args, **kwargs)

class surRqNSF(transforms.PiecewiseRationalQuadraticCouplingTransform):
    def __init__(self, dropped_entries_shape, context_shape, mask, transform_net_create_fn,
                 num_bins=10,
                 tails=None,
                 tail_bound=1.0,
                 apply_unconditional_transform=False,
                 img_shape=None,
                 min_bin_width=splines.rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height=splines.rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
                 min_derivative=splines.rational_quadratic.DEFAULT_MIN_DERIVATIVE,
                 generator_function=create_flow):
        super(surRqNSF, self).__init__(mask=mask,
                                       transform_net_create_fn=transform_net_create_fn,
                                       tails=tails,
                                       tail_bound=tail_bound,
                                       num_bins=num_bins,
                                       apply_unconditional_transform=apply_unconditional_transform,
                                       min_bin_width=min_bin_width,
                                       min_bin_height=min_bin_height,
                                       min_derivative=min_derivative,
                                       img_shape=img_shape
                                       )
        self.drop_mask = torch.ones(self.features, dtype=torch.bool)
        feature_to_drop = self.identity_features[-1]
        self.drop_mask[feature_to_drop] = 0
        _, self.sorted_indices = torch.sort(
            torch.cat((torch.arange(self.features)[self.drop_mask], torch.tensor([feature_to_drop]))))

        # A flow that can generate one dropped index of the input data given the other data entries
        self.generator = generator_function(dropped_entries_shape, context_shape)
        # self.generator = generator_function(dropped_entries_shape, context_channels=context_shape)

    def forward(self, inputs, context=None):
        faux_output, log_contr = super().forward(inputs, context=context)
        output = faux_output[:, self.drop_mask, ...]
        likelihood = self.generator.log_prob(faux_output[:, ~self.drop_mask, ...], context=output)
        # return output, log_contr + likelihood
        return output, log_contr + likelihood

    def inverse(self, inputs, context=None):
        input_dropped = self.generator.sample(1, context=inputs).squeeze().unsqueeze(1)
        likelihood = self.generator.log_prob(input_dropped, context=inputs)
        inputs = torch.cat((inputs, input_dropped), 1)[:, self.sorted_indices, ...]
        output, log_contr = super().inverse(inputs, context=context)
        # return output, log_contr + likelihood
        # TODO: return the correct likelihood
        return output, torch.zeros(inputs.shape[0]).to(inputs.device)


class surAffineCoupling(transforms.AffineCouplingTransform):
    def __init__(self, dropped_entries_shape, context_shape, mask, transform_net_create_fn,
                 generator_function=make_generator):
        super(surAffineCoupling, self).__init__(mask=mask,
                                       transform_net_create_fn=transform_net_create_fn,
                                       )
        self.drop_mask = torch.ones(self.features, dtype=torch.bool)
        feature_to_drop = self.identity_features[-1]
        self.drop_mask[feature_to_drop] = 0
        _, self.sorted_indices = torch.sort(
            torch.cat((torch.arange(self.features)[self.drop_mask], torch.tensor([feature_to_drop]))))

        # A flow that can generate one dropped index of the input data given the other data entries
        self.generator = generator_function(dropped_entries_shape, context_shape)
        # self.generator = generator_function(dropped_entries_shape, context_channels=context_shape)

    def forward(self, inputs, context=None):
        faux_output, log_contr = super().forward(inputs, context=context)
        output = faux_output[:, self.drop_mask, ...]
        likelihood = self.generator.log_prob(faux_output[:, ~self.drop_mask, ...], context=output)
        # return output, log_contr + likelihood
        return output, log_contr + likelihood
        # return output, -likelihood * (log_contr + likelihood)

    def inverse(self, inputs, context=None):
        input_dropped = self.generator.sample(1, context=inputs).squeeze().unsqueeze(1)
        likelihood = self.generator.log_prob(input_dropped, context=inputs)
        inputs = torch.cat((inputs, input_dropped), 1)[:, self.sorted_indices, ...]
        output, log_contr = super().inverse(inputs, context=context)
        # return output, log_contr + likelihood
        return output, torch.zeros(inputs.shape[0]).to(inputs.device)


# class baseCouplingFunnel(nn.Module):
#     def __init__(self, *args, context_shape=None, dropped_entries_shape=None, generator_function=None, **kwargs):
#         super(baseCouplingFunnel, self).__init__()
#         for arg in [context_shape, generator_function, dropped_entries_shape]:
#             if arg is None:
#                 self.arg_missing_exception(arg)
#         self.drop_mask = torch.ones(self.features, dtype=torch.bool)
#         feature_to_drop = self.identity_features[-1]
#         self.drop_mask[feature_to_drop] = 0
#         _, self.sorted_indices = torch.sort(
#             torch.cat((torch.arange(self.features)[self.drop_mask], torch.tensor([feature_to_drop]))))
#
#         # A flow that can generate one dropped index of the input data given the other data entries
#         self.generator = generator_function(dropped_entries_shape, context_shape)
#         # self.generator = generator_function(dropped_entries_shape, context_channels=context_shape)
#
#     def arg_missing_exception(self, arg):
#         raise Exception('Need to pass {}')
#
#     def forward(self, inputs, context=None):
#         # TODO: how to get this to call the method which is subclassed?
#         faux_output, log_contr = super().forward(inputs, context=context)
#         output = faux_output[:, self.drop_mask, ...]
#         likelihood = self.generator.log_prob(faux_output[:, ~self.drop_mask, ...], context=output)
#         # return output, log_contr + likelihood
#         return output, torch.exp(-log_contr) * (log_contr + likelihood)
#         # return output, -likelihood * (log_contr + likelihood)
#
#     def inverse(self, inputs, context=None):
#         input_dropped = self.generator.sample(1, context=inputs).squeeze().unsqueeze(1)
#         likelihood = self.generator.log_prob(input_dropped, context=inputs)
#         inputs = torch.cat((inputs, input_dropped), 1)[:, self.sorted_indices, ...]
#         # TODO: how to get this to call the method which is subclassed?
#         output, log_contr = super().inverse(inputs, context=context)
#         # return output, log_contr + likelihood
#         return output, torch.zeros(inputs.shape[0]).to(inputs.device)
#
#
# class wrapAffineCoupling(transforms.AffineCouplingTransform):
#
#     def __init__(
#             self, *args, mask=None, transform_net_create_fn=None, **kwargs
#     ):
#         super(wrapAffineCoupling, self).__init__(
#             self, mask, transform_net_create_fn,
#         )
#
#
# class surAffineCoupling(wrapAffineCoupling, baseCouplingFunnel):
#     def __init__(self, dropped_entries_shape, context_shape, mask, transform_net_create_fn,
#                  generator_function=make_generator):
#         # super(surAffineCoupling, self).__init__(mask=mask,
#         #                                         transform_net_create_fn=transform_net_create_fn,
#         #                                         dropped_entries_shape=dropped_entries_shape,
#         #                                         context_shape=context_shape,
#         #                                         generator_function=generator_function
#         #                                         )
#         wrapAffineCoupling.__init__(self, mask=mask, transform_net_create_fn=transform_net_create_fn)
#         baseCouplingFunnel.__init__(self, dropped_entries_shape=dropped_entries_shape, context_shape=context_shape,
#                                     generator_function=generator_function)
