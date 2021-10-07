import json

from surVAE.data.base import load_num_batches
from surVAE.data.image_data import get_image_data, Preprocess
import torch
from nflows import distributions, flows, transforms
from nflows.utils import create_mid_split_binary_mask
import torch.nn as nn
from nflows.nn.nets import ConvResidualNet
from nflows.utils import get_num_parameters

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

# def config():
# Saving
from surVAE.models.sur_flows import NByOneConv, surRqNSF, NByOneSlice, NByOneStandardConv, TanhLayer, LeakyRelu, \
    NByOneInnConv
from surVAE.utils.io import save_object
from surVAE.utils import autils
import time
import matplotlib.pyplot as plt
import os
import numpy as np

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Saving
    parser.add_argument('-d', '--outputdir', type=str, default='plane_images_local',
                        help='Choose the base output directory')
    parser.add_argument('-n', '--outputname', type=str, default='local', help='Set the output name directory')
    parser.add_argument('--load', type=int, default=0, help='Load a model?')
    parser.add_argument('--train_flow', type=int, default=1, help='Train the flow?')

    # Model set up
    parser.add_argument('--model', type=str, default='funnel_conv',
                        help='The dimension of the input data.')
    parser.add_argument('--slice', type=int, default=2,
                        help='Use a funnel or slice the tensor?')
    parser.add_argument('--funnel_first', type=int, default=0,
                        help='The dimension of the input data.')
    parser.add_argument('--n_funnels', type=int, default=3,
                        help='The number of funnel layers to apply.')
    parser.add_argument('--conv_width', type=int, default=2,
                        help='The width of the convolutional kernel to apply.')
    parser.add_argument('--steps_per_level', type=int, default=7,
                        help='The number of steps per GLOW type level.')
    parser.add_argument('--levels', type=int, default=3,
                        help='The number of levels to apply to the data.')
    parser.add_argument('--multi_scale', type=int, default=0,
                        help='Multi scale architecture?')
    parser.add_argument('--actnorm', type=int, default=1,
                        help='Use actnorm?')
    parser.add_argument('--coupling_layer_type', type=str, default='rational_quadratic_spline',
                        help='The type of coupling layer to apply to the data.')
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='The number of hidden channels in the nets that learn flow param.')
    parser.add_argument('--use_resnet', type=int, default=1,
                        help='Use resnet layers to learn flow param.?')
    parser.add_argument('--num_res_blocks', type=int, default=3,
                        help='Number of resnet blocks if using resnet.')
    parser.add_argument('--resnet_batchnorm', type=int, default=1,
                        help='Use batchnorm in resnet layers?')
    parser.add_argument('--dropout_prob', type=float, default=0.2,
                        help='Dropout prob in net for learning flow param.')
    parser.add_argument('--apply_unconditional_transform', type=int, default=0,
                        help='Spline conditional transformation?')
    parser.add_argument('--min_bin_height', type=float, default=0.001,
                        help='Minimum spline bin height.')
    parser.add_argument('--min_bin_width', type=float, default=0.001,
                        help='Minimum spline bin width.')
    parser.add_argument('--min_derivative', type=float, default=0.001,
                        help='Minimum spline derivative.')
    parser.add_argument('--num_bins', type=int, default=2,
                        help='Number of bins in the spline.')
    parser.add_argument('--tail_bound', type=float, default=3.0,
                        help='Spline tail bound.')
    parser.add_argument('--activation_funnel', type=str, default='none',
                        help='Spline tail bound.')
    parser.add_argument('--gauss_decoder', type=int, default=1,
                        help='Spline tail bound.')

    # Dataset and training parameters
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='The name of the plane dataset on which to train.')
    # parser.add_argument('--dataset', type=str, default='imagenet-64-fast',
    #                     help='The name of the plane dataset on which to train.')
    parser.add_argument('--valid_frac', type=float, default=0.01,
                        help='The fraction of samples to take for validation.')
    parser.add_argument('--num_bits', type=int, default=8,
                        help='The number of bits to take in the image.')
    parser.add_argument('--pad', type=int, default=2,
                        help='The amount of padding to apply.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Learning rate.')
    parser.add_argument('--cosine_annealing', type=int, default=1,
                        help='Use cosine annealing?')
    parser.add_argument('--eta_min', type=float, default=0.0,
                        help='The min eta in cosine annealing schedule.')
    parser.add_argument('--warmup_fraction', type=float, default=0.0,
                        help='The warmp up fractionof the training steps.')
    parser.add_argument('--num_steps', type=int, default=200000,
                        help='The number of training steps to run for.')

    # reproducibility
    parser.add_argument('--seed', type=int, default=656693568,
                        help='Random seed for PyTorch and NumPy.')

    return parser.parse_args()


args = parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

svo = save_object(f'{args.outputdir}_{args.dataset}_{args.model}', exp_name=args.outputname, args=args)
directory = svo.image_dir
# Dataset
dataset = args.dataset
num_workers = 0

# Pre-processing
preprocessing = 'glow'
num_bits = args.num_bits
pad = args.pad  # For mnist-like datasets

# Model architecture
flow_type = args.model
n_funnels = args.n_funnels

conv_width = args.conv_width
# steps_per_level = 10
steps_per_level = args.steps_per_level
levels = args.levels
# For imagenet experiments this is set to true, but have removed option for public code, not used for funnel experiments
multi_scale = args.multi_scale
actnorm = args.actnorm

# Coupling transform
coupling_layer_type = args.coupling_layer_type
# spline_params = args.spline_params[0]
# spline_params = json.loads(args.spline_params)
spline_params = {
    "apply_unconditional_transform": args.apply_unconditional_transform,
    "min_bin_height": args.min_bin_height,
    "min_bin_width": args.min_bin_width,
    "min_derivative": args.min_derivative,
    "num_bins": args.num_bins,
    "tail_bound": args.tail_bound
}

# Coupling transform net
# hidden_channels = int(args.hidden_channels / 1.3) if flow_type[:6] == 'funnel' else args.hidden_channels
hidden_channels = args.hidden_channels
if not isinstance(hidden_channels, list):
    hidden_channels = [hidden_channels] * levels

if flow_type == 'glow':
    n_funnels = 0
    conv_width = 1

use_resnet = args.use_resnet
num_res_blocks = args.num_res_blocks  # If using resnet
resnet_batchnorm = args.resnet_batchnorm
dropout_prob = args.dropout_prob

# Optimization
batch_size = args.batch_size
learning_rate = args.learning_rate
cosine_annealing = args.cosine_annealing
eta_min = args.eta_min
warmup_fraction = args.warmup_fraction
num_steps = args.num_steps
temperatures = [0.5, 0.75, 1.]

# Training logistics
use_gpu = True
multi_gpu = False
run_descr = ''
flow_checkpoint = None
optimizer_checkpoint = None
start_step = 0

intervals = {
    'save': 1000,
    'sample': 1000,
    'eval': 1000,
    'reconstruct': 1000,
    'log': 10  # Very cheap.
}

# For evaluation
num_samples = 64
samples_per_row = 8
num_reconstruct_batches = 10


class Conv2dSameSize(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size):
        same_padding = kernel_size // 2  # Padding that would keep the spatial dims the same
        super().__init__(in_channels, out_channels, kernel_size,
                         padding=same_padding)


class ConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.net = nn.Sequential(
            Conv2dSameSize(in_channels, hidden_channels, kernel_size=3),
            nn.ReLU(),
            Conv2dSameSize(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            Conv2dSameSize(hidden_channels, out_channels, kernel_size=3),
        )

    def forward(self, inputs, context=None):
        return self.net.forward(inputs)


def create_transform_step(num_channels, hidden_channels, coupling_layer_type='rational_quadratic_spline', size_in=None,
                          size_context=None, context_channels=None):
    if use_resnet:
        def create_convnet(in_channels, out_channels):
            net = ConvResidualNet(in_channels=in_channels,
                                  out_channels=out_channels,
                                  hidden_channels=hidden_channels,
                                  num_blocks=num_res_blocks,
                                  use_batch_norm=resnet_batchnorm,
                                  dropout_probability=dropout_prob,
                                  context_channels=context_channels)
            return net
    else:
        if dropout_prob != 0.:
            raise ValueError()

        def create_convnet(in_channels, out_channels):
            return ConvNet(in_channels, hidden_channels, out_channels)

    mask = create_mid_split_binary_mask(num_channels)

    if coupling_layer_type == 'sur':
        coupling_layer = surRqNSF(size_in,
                                  size_context,
                                  mask,
                                  create_convnet,
                                  tails='linear',
                                  tail_bound=spline_params['tail_bound'],
                                  num_bins=spline_params['num_bins'],
                                  apply_unconditional_transform=spline_params['apply_unconditional_transform'],
                                  min_bin_width=spline_params['min_bin_width'],
                                  min_bin_height=spline_params['min_bin_height']
                                  )
    elif coupling_layer_type == 'cubic_spline':
        coupling_layer = transforms.PiecewiseCubicCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails='linear',
            tail_bound=spline_params['tail_bound'],
            num_bins=spline_params['num_bins'],
            apply_unconditional_transform=spline_params['apply_unconditional_transform'],
            min_bin_width=spline_params['min_bin_width'],
            min_bin_height=spline_params['min_bin_height']
        )
    elif coupling_layer_type == 'quadratic_spline':
        coupling_layer = transforms.PiecewiseQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails='linear',
            tail_bound=spline_params['tail_bound'],
            num_bins=spline_params['num_bins'],
            apply_unconditional_transform=spline_params['apply_unconditional_transform'],
            min_bin_width=spline_params['min_bin_width'],
            min_bin_height=spline_params['min_bin_height']
        )
    elif coupling_layer_type == 'rational_quadratic_spline':
        coupling_layer = transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails='linear',
            tail_bound=spline_params['tail_bound'],
            num_bins=spline_params['num_bins'],
            apply_unconditional_transform=spline_params['apply_unconditional_transform'],
            min_bin_width=spline_params['min_bin_width'],
            min_bin_height=spline_params['min_bin_height'],
            min_derivative=spline_params['min_derivative']
        )
    elif coupling_layer_type == 'affine':
        coupling_layer = transforms.AffineCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet
        )
    elif coupling_layer_type == 'additive':
        coupling_layer = transforms.AdditiveCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet
        )
    else:
        raise RuntimeError('Unknown coupling_layer_type')

    step_transforms = []

    if actnorm:
        step_transforms.append(transforms.ActNorm(num_channels))

    step_transforms.extend([
        transforms.OneByOneConvolution(num_channels),
        coupling_layer
    ])

    return transforms.CompositeTransform(step_transforms)


class ReshapeTransform(transforms.Transform):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, inputs, context=None):
        if tuple(inputs.shape[1:]) != self.input_shape:
            raise RuntimeError('Unexpected inputs shape ({}, but expecting {})'
                               .format(tuple(inputs.shape[1:]), self.input_shape))
        return inputs.reshape(-1, *self.output_shape), torch.zeros(inputs.shape[0]).to(inputs.device)

    def inverse(self, inputs, context=None):
        if tuple(inputs.shape[1:]) != self.output_shape:
            raise RuntimeError('Unexpected inputs shape ({}, but expecting {})'
                               .format(tuple(inputs.shape[1:]), self.output_shape))
        return inputs.reshape(-1, *self.input_shape), torch.zeros(inputs.shape[0]).to(inputs.device)


class RotateImageTransform(transforms.Transform):
    def forward(self, inputs, context=None):
        if inputs.dim() != 4:
            raise Exception('Expected an image as input.')
        return inputs.permute(0, 1, 3, 2), torch.zeros(inputs.shape[0]).to(inputs.device)

    def inverse(self, inputs, context=None):
        if inputs.dim() != 4:
            raise Exception('Expected an image as input.')
        return inputs.permute(0, 1, 3, 2), torch.zeros(inputs.shape[0]).to(inputs.device)


def funnel_conv(num_channels, hidden_channels, image_width):
    step_transforms = []

    # # TODO: does this help with training funnels?
    if actnorm:
        step_transforms.append(transforms.ActNorm(num_channels))
    indx = 1
    # indx = 0
    if args.slice == 1:
        step_transforms.extend([
            NByOneSlice(num_channels,
                        hidden_features=hidden_channels,
                        width=conv_width,
                        num_blocks=num_res_blocks,
                        nstack=steps_per_level,
                        # tail_bound=spline_params['tail_bound'],
                        tail_bound=4.,
                        num_bins=spline_params['num_bins'],
                        spline=1)
        ])
    elif args.slice == 2:
        step_transforms.extend([
            NByOneStandardConv(num_channels, image_width,
                               hidden_features=hidden_channels,
                               width=conv_width,
                               num_blocks=num_res_blocks,
                               nstack=steps_per_level,
                               tail_bound=4.,
                               num_bins=spline_params['num_bins'],
                               spline=1,
                               gauss=args.gauss_decoder
                               )
        ])
    elif args.slice == 3:
        step_transforms.extend([
            NByOneInnConv(num_channels, image_width,
                          hidden_features=hidden_channels,
                          width=conv_width,
                          num_blocks=num_res_blocks,
                          nstack=steps_per_level,
                          tail_bound=4.,
                          num_bins=spline_params['num_bins'],
                          spline=1,
                          gauss=args.gauss_decoder
                          )
        ])
    else:
        step_transforms.extend([
            NByOneConv(num_channels,
                       hidden_features=hidden_channels,
                       width=conv_width,
                       num_blocks=num_res_blocks,
                       nstack=steps_per_level,
                       # tail_bound=spline_params['tail_bound'],
                       tail_bound=4.,
                       num_bins=spline_params['num_bins'],
                       spline=1)
        ])

    if args.activation_funnel == 'tanh':
        step_transforms.extend([TanhLayer()])
    elif args.activation_funnel == 'leaky_relu':
        step_transforms.extend([LeakyRelu()])

    return transforms.CompositeTransform(step_transforms), step_transforms[indx].output_image_size


def add_glow(size_in, context_channels=None):
    c, h, w = size_in
    all_transforms = []
    for level, level_hidden_channels in zip(range(levels), hidden_channels):
        image_size = c * h * w
        squeeze_transform = transforms.SqueezeTransform()
        c_t, h_t, w_t = squeeze_transform.get_output_shape(c, h, w)
        if c_t * h_t * w_t == image_size:
            squeeze = 1
            c, h, w = c_t, h_t, w_t
        else:
            print(f'No more squeezing after level {level + n_funnels}')
            squeeze = 0

        layer_transform = [create_transform_step(c, level_hidden_channels, context_channels=context_channels) for _ in
                           range(steps_per_level)] + [transforms.OneByOneConvolution(c)]
        if squeeze:
            layer_transform = [squeeze_transform] + layer_transform
        all_transforms += [transforms.CompositeTransform(layer_transform)]
        print(c, h, w)

    all_transforms.append(ReshapeTransform(
        input_shape=(c, h, w),
        output_shape=(c * h * w,)
    ))

    return all_transforms


def create_transform(flow_type, size_in, size_out):
    # if not isinstance(hidden_channels, list):
    #     hidden_channels = [hidden_channels] * levels

    c, h, w = size_in
    c_out, h_out, w_out = size_out

    all_transforms = []

    if flow_type == 'glow':
        if multi_scale:
            mct = transforms.MultiscaleCompositeTransform(num_transforms=levels)
            for level, level_hidden_channels in zip(range(levels), hidden_channels):

                squeeze_transform = transforms.SqueezeTransform()
                c, h, w = squeeze_transform.get_output_shape(c, h, w)

                transform_level = transforms.CompositeTransform(
                    [squeeze_transform]
                    + [create_transform_step(c, level_hidden_channels) for _ in range(steps_per_level)]
                    + [transforms.OneByOneConvolution(c)]  # End each level with a linear transformation.
                )

                new_shape = mct.add_transform(transform_level, (c, h, w))
                if new_shape:  # If not last layer
                    c, h, w = new_shape

            # So the correct shape can be inferred
            c, h, w = size_in
        else:
            all_transforms = add_glow(size_in)

    elif flow_type == 'funnel_conv':
        for level, level_hidden_channels in zip(range(levels), hidden_channels):

            if args.funnel_first == 1:
                if level == 0:
                    funnel_model, width = funnel_conv(c, level_hidden_channels, w)
                    all_transforms += [funnel_model]
                    # w = int((w - w % conv_width) * (conv_width - 1) / conv_width)
                    # w = 16
                    # h = 16
                    w = width
                    h = w

            image_size = c * h * w
            squeeze_factor = 2
            squeeze_transform = transforms.SqueezeTransform(factor=squeeze_factor)
            c_t, h_t, w_t = squeeze_transform.get_output_shape(c, h, w)
            if c_t * h_t * w_t == image_size:
                squeeze = 1
                c, h, w = c_t, h_t, w_t
            else:
                print(f'No more squeezing after level {level + n_funnels}')
                squeeze = 0

            layer_transform = [create_transform_step(c, level_hidden_channels) for _ in range(steps_per_level)] + [
                transforms.OneByOneConvolution(c)]
            if squeeze:
                if (args.funnel_first == 2) and (level == 0):
                    # funnel_model, width = funnel_conv(c, level_hidden_channels, w)
                    # layer_transform = [squeeze_transform, transforms.OneByOneConvolution(c),
                    #                    funnel_model] + layer_transform
                    # # w = int((w - w % conv_width) * (conv_width - 1) / conv_width)
                    # w = width
                    # h = w
                    funnel_model, width = funnel_conv(c, level_hidden_channels, w)
                    funnel_model_2, width = funnel_conv(c, level_hidden_channels, width)
                    w = width
                    h = w
                    layer_transform = [squeeze_transform, transforms.OneByOneConvolution(c),
                                       funnel_model, transforms.OneByOneConvolution(c),
                                       funnel_model_2] + layer_transform
                else:
                    layer_transform = [squeeze_transform] + layer_transform
            all_transforms += [transforms.CompositeTransform(layer_transform)]

            if args.funnel_first == 0:
                if level == 0:
                # if level < 2:
                    funnel_model, width = funnel_conv(c, level_hidden_channels, w)
                    all_transforms += [funnel_model]
                    w = width
                    h = w

            print(c, h, w)

        all_transforms.append(ReshapeTransform(
            input_shape=(c, h, w),
            output_shape=(c * h * w,)
        ))

    elif flow_type == 'funnel':
        # image_size = c * h * w
        for level, level_hidden_channels in zip(range(levels), hidden_channels):
            image_size = c * h * w
            squeeze_factor = 2
            squeeze_transform = transforms.SqueezeTransform(factor=squeeze_factor)
            c_t, h_t, w_t = squeeze_transform.get_output_shape(c, h, w)
            if (c_t * h_t * w_t == image_size) and (level < 1):
                squeeze = 1
                c, h, w = c_t, h_t, w_t
            else:
                print(f'No more squeezing after level {level + n_funnels}')
                squeeze = 0

            layer_transform = [create_transform_step(c, level_hidden_channels, coupling_layer_type=coupling_layer_type)
                               for _ in range(steps_per_level)] + [transforms.OneByOneConvolution(c)]

            if squeeze:
                layer_transform = [squeeze_transform] + layer_transform

            # if level >= levels - n_funnels:
            if level < n_funnels:
                layer_transform += [create_transform_step(c, level_hidden_channels, size_in=(1, h, w),
                                                          size_context=(c - 1, h, w), coupling_layer_type='sur')]
                c -= 1
                image_size -= h * w
            print(c, h, w)

            all_transforms += [transforms.CompositeTransform(layer_transform)]

        all_transforms.append(ReshapeTransform(
            input_shape=(c, h, w),
            output_shape=(c * h * w,)
        ))

    else:
        raise RuntimeError('Unknown type of flow')

    if not multi_scale:
        mct = transforms.CompositeTransform(all_transforms)

    # Inputs to the model in [0, 2 ** num_bits]
    # Only ever going to use glow preprocessing to follow prescription of NSF paper
    if preprocessing == 'glow':
        # Map to [-0.5,0.5]
        preprocess_transform = transforms.AffineScalarTransform(scale=1. / 2 ** num_bits,
                                                                shift=-0.5)

    else:
        raise RuntimeError('Unknown preprocessing type: {}'.format(preprocessing))

    return transforms.CompositeTransform([preprocess_transform, mct]), (c, h, w)


def create_flow(size_in, size_out, flow_checkpoint=None, flow_type=flow_type):
    c_out, h_out, w_out = size_out
    transform, (c_out, h_out, w_out) = create_transform(flow_type, size_in, size_out)
    distribution = distributions.StandardNormal((c_out * h_out * w_out,))

    flow = flows.Flow(transform, distribution)

    if flow_checkpoint is not None:
        flow.load_state_dict(torch.load(flow_checkpoint))

    if args.load:
        flow.load_state_dict(torch.load(os.path.join(directory, f'{args.outputname}_flow_last.pt')))

    return flow


def train_flow(flow, train_dataset, val_dataset, dataset_dims, device):
    flow = flow.to(device)

    run_dir = directory
    summary_writer = SummaryWriter(run_dir, max_queue=100)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers)

    if val_dataset:
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers)
    else:
        val_loader = None

    # Random batch and identity transform for reconstruction evaluation.
    random_batch, _ = next(iter(DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=0  # Faster than starting all workers just to get a single batch.
    )))
    identity_transform = transforms.CompositeTransform([
        flow._transform,
        transforms.InverseTransform(flow._transform)
    ])

    optimizer = torch.optim.Adam(flow.parameters(), lr=learning_rate)

    if optimizer_checkpoint is not None:
        optimizer.load_state_dict(torch.load(optimizer_checkpoint))

    if cosine_annealing:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=num_steps,
            last_epoch=-1 if start_step == 0 else start_step,
            eta_min=eta_min
        )
    else:
        scheduler = None

    def nats_to_bits_per_dim(x):
        c, h, w = dataset_dims
        return autils.nats_to_bits_per_dim(x, c, h, w)

    print('Starting training...')

    best_val_log_prob = None
    start_time = None
    num_batches = num_steps - start_step

    for step, (batch, _) in enumerate(load_num_batches(loader=train_loader,
                                                       num_batches=num_batches),
                                      start=start_step):
        if step == 0:
            start_time = time.time()  # Runtime estimate will be more accurate if set here.

        flow.train()

        optimizer.zero_grad()

        batch = batch.to(device)

        log_density = flow.log_prob(batch)

        loss = -nats_to_bits_per_dim(torch.mean(log_density))

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
            summary_writer.add_scalar('learning_rate', scheduler.get_last_lr(), step)

        summary_writer.add_scalar('loss', loss.item(), step)

        if best_val_log_prob:
            summary_writer.add_scalar('best_val_log_prob', best_val_log_prob, step)

        flow.eval()  # Everything beyond this point is evaluation.

        if step % intervals['log'] == 0:
            elapsed_time = time.time() - start_time
            progress = autils.progress_string(elapsed_time, step, num_steps)
            print("It: {}/{} loss: {:.3f} [{}]".format(step, num_steps, loss, progress))

        if step % intervals['sample'] == 0:
            fig, axs = plt.subplots(1, len(temperatures), figsize=(4 * len(temperatures), 4))
            for temperature, ax in zip(temperatures, axs.flat):
                with torch.no_grad():
                    noise = flow._distribution.sample(64) * temperature
                    samples, _ = flow._transform.inverse(noise)
                    samples = Preprocess(num_bits).inverse(samples)

                autils.imshow(make_grid(samples, nrow=8), ax)
                # n_conv = NByOneStandardConv(3)
                # autils.imshow(make_grid(n_conv.test_inverse(batch[:64])[0], nrow=8), axs[0])
                # autils.imshow(make_grid(batch[:64], nrow=8), axs[1])

                ax.set_title('T={:.2f}'.format(temperature))

            summary_writer.add_figure(tag='samples', figure=fig, global_step=step)
            fig.savefig(svo.save_name(f'samples_{step}.png'))
            plt.close(fig)

            # fig, axs = plt.subplots(1, 2, figsize=(4, 4))
            # autils.imshow(make_grid((batch[:64] / 2 ** num_bits - 0.5) * 2, nrow=8), axs[0])
            # n_conv = NByOneStandardConv(3)
            # autils.imshow(make_grid((n_conv.test_inverse(batch[:64])[0] / 2 ** num_bits - 0.5) * 2, nrow=8), axs[1])
            # fig.savefig(svo.save_name(f'training_data{step}.png'))
            # fig.tight_layout()
            # plt.close(fig)

        if step > 0 and step % intervals['eval'] == 0 and (val_loader is not None):
            def log_prob_fn(batch):
                return flow.log_prob(batch.to(device))

            val_log_prob = autils.eval_log_density(log_prob_fn=log_prob_fn,
                                                   data_loader=val_loader)
            val_log_prob = nats_to_bits_per_dim(val_log_prob).item()

            print("It: {}/{} val_log_prob: {:.3f}".format(step, num_steps, val_log_prob))
            summary_writer.add_scalar('val_log_prob', val_log_prob, step)

            if best_val_log_prob is None or val_log_prob > best_val_log_prob:
                best_val_log_prob = val_log_prob

                torch.save(flow.state_dict(), os.path.join(run_dir, 'flow_best.pt'))
                print('It: {}/{} best val_log_prob improved, saved flow_best.pt'
                      .format(step, num_steps))

        if step > 0 and (step % intervals['save'] == 0 or step == (num_steps - 1)):
            torch.save(optimizer.state_dict(), os.path.join(run_dir, f'{args.outputname}_optimizer_last.pt'))
            torch.save(flow.state_dict(), os.path.join(run_dir, f'{args.outputname}_flow_last.pt'))
            print('It: {}/{} saved optimizer_last.pt and flow_last.pt'.format(step, num_steps))

        # TODO: this is actually a useful measure, but breaks everything in it's current formulation
        # if step > 0 and step % intervals['reconstruct'] == 0:
        #     with torch.no_grad():
        #         random_batch_ = random_batch.to(device)
        #         random_batch_rec, logabsdet = identity_transform(random_batch_)
        #
        #         max_abs_diff = torch.max(torch.abs(random_batch_rec - random_batch_))
        #         max_logabsdet = torch.max(logabsdet)
        #
        #     summary_writer.add_scalar(tag='max_reconstr_abs_diff',
        #                               scalar_value=max_abs_diff.item(),
        #                               global_step=step)
        #     summary_writer.add_scalar(tag='max_reconstr_logabsdet',
        #                               scalar_value=max_logabsdet.item(),
        #                               global_step=step)


def evaluate_flow(flow, val_dataset, dataset_dims, device, anomaly_dataset=None):
    flow = flow.to(device)
    flow.eval()
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers)

    if anomaly_dataset is not None:
        anomaly_loader = DataLoader(dataset=val_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers)
    else:
        anomaly_loader = None

    def nats_to_bits_per_dim(x):
        c, h, w = dataset_dims
        return autils.nats_to_bits_per_dim(x, c, h, w)

    def log_prob_fn(batch):
        return flow.log_prob(batch.to(device))

    val_log_prob = autils.eval_log_density(log_prob_fn=log_prob_fn,
                                           data_loader=val_loader)
    val_log_prob = nats_to_bits_per_dim(val_log_prob)
    print(f'Bits per dim {val_log_prob}')

    if anomaly_loader is not None:
        anomaly_log_prob = autils.eval_log_density(log_prob_fn=log_prob_fn,
                                                   data_loader=val_loader)
        anomaly_log_prob = nats_to_bits_per_dim(anomaly_log_prob)
        print(anomaly_log_prob)


def train_and_generate_images():
    train_dataset, val_dataset, (c, h, w) = get_image_data(dataset, num_bits, valid_frac=args.valid_frac)

    if flow_type == 'funnel_conv':
        # c_out, h_out, w_out = c, int(h / conv_width ** n_funnels), int(w / conv_width ** n_funnels)
        # TODO: when you fix this it should be ceil not floor, for now you are dropping a row of pixels to make it even
        # for the squeezing operation to work how it should
        c_out, h_out, w_out = c, \
                              h, \
                              int((w - w % conv_width) * ((conv_width - 1) / conv_width) ** n_funnels)
        # int(np.floor(w * ((conv_width - 1) / conv_width) ** n_funnels))
    elif flow_type == 'funnel_non_conv':
        c_out, h_out, w_out = 256, 1, 1
    elif flow_type == 'funnel':
        c_out, h_out, w_out = c - n_funnels, h, w
    else:
        c_out, h_out, w_out = c, h, w

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    flow = create_flow((c, h, w), (c_out, h_out, w_out))
    print(f'There are {get_num_parameters(flow)} params')
    # Can't set default back without messing with the nflows package directly, the problem is the zeros likelihoods
    # torch.set_default_tensor_type('torch.FloatTensor')

    if args.train_flow:
        train_flow(flow, train_dataset, val_dataset, (c, h, w), device)
    #
    if dataset == 'mnist':
        anomaly_dataset, val_dataset, _ = get_image_data('fashion-mnist', num_bits, valid_frac=0.1)
    else:
        anomaly_dataset = None
    evaluate_flow(flow, val_dataset, (c, h, w), device, anomaly_dataset=anomaly_dataset)


if __name__ == '__main__':
    train_and_generate_images()
