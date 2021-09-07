from nflows import transforms
from nflows import flows
import nflows

import torch
import numpy as np
import matplotlib.pyplot as plt

import time

from surVAE.models.sur_flows import SurNSF
from surVAE.data.hyper_dim import HyperCheckerboardDataset

import argparse

from surVAE.utils.io import save_object
from surVAE.utils.plotting import getCrossFeaturePlot
from surVAE.utils.torch_utils import tensor2numpy


def parse_args():
    parser = argparse.ArgumentParser()

    # Saving
    parser.add_argument('-d', '--outputdir', type=str, default='checkerboard_local',
                        help='Choose the base output directory')
    parser.add_argument('-n', '--outputname', type=str, default='local',
                        help='Set the output name directory')
    parser.add_argument('--load', type=int, default=0,
                        help='Load a model?')

    # Model set up
    parser.add_argument('--inp_dim', type=int, default=2,
                        help='The dimension of the input data.')
    parser.add_argument('--nodes', type=int, default=64,
                        help='The number of nodes in each layer used to learn the flow parameters.')
    parser.add_argument('--num_blocks', type=int, default=2,
                        help='The number of layers used to learn the flow placements.')
    parser.add_argument('--nstack', type=int, default=2,
                        help='The number of flow layers.')
    parser.add_argument('--tails', type=str, default='linear',
                        help='The tail function to apply.')
    parser.add_argument('--tail_bound', type=float, default=4.,
                        help='The tail bound.')
    parser.add_argument('--num_bins', type=int, default=10,
                        help='The number of bins to use in the RQ-NSF.')
    parser.add_argument('--num_add', type=int, default=1,
                        help='The number of additional layers to add.')
    parser.add_argument('--add_sur', type=int, default=1,
                        help='Whether to make the additional layers surVAE layers.')

    # Dataset and training parameters
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Whether to make the additional layers surVAE layers.')
    parser.add_argument('--n_epochs', type=int, default=5,
                        help='Whether to make the additional layers surVAE layers.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Whether to make the additional layers surVAE layers.')
    parser.add_argument('--ndata', type=int, default=int(1e5),
                        help='Whether to make the additional layers surVAE layers.')
    parser.add_argument('--n_val', type=int, default=int(1e3),
                        help='Whether to make the additional layers surVAE layers.')
    parser.add_argument('--n_test', type=int, default=int(1e4),
                        help='Whether to make the additional layers surVAE layers.')
    parser.add_argument('--gclip', type=float, default=5.,
                        help='Whether to make the additional layers surVAE layers.')
    parser.add_argument('--monitor_interval', type=int, default=100,
                        help='Whether to make the additional layers surVAE layers.')
    parser.add_argument('--bnorm', type=int, default=1,
                        help='Apply batch normalisation?')

    return parser.parse_args()


def checkerboard_test():
    # Parse args
    args = parse_args()

    # Set up saving and change the name of some args
    svo = save_object(args.outputdir, args.outputname, args=args)
    inp_dim = args.inp_dim
    out_dim = inp_dim - args.num_add * args.add_sur
    nstack = args.nstack + args.num_add * (1 - args.add_sur)

    # Set up and define the model
    transform_list = []
    for i in range(nstack):
        transform_list += [
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(inp_dim, args.nodes,
                                                                               num_blocks=args.num_blocks,
                                                                               tail_bound=args.tail_bound,
                                                                               num_bins=args.num_bins,
                                                                               tails=args.tails)]
        if args.bnorm:
            transform_list += [
                transforms.BatchNorm(inp_dim)
            ]

        # transform_list += [transforms.ReversePermutation(inp_dim)]
        transform_list += [transforms.LULinear(inp_dim)]

    # Add the surVAE layers
    dim = inp_dim
    for _ in range(args.num_add * args.add_sur):
        transform_list += [SurNSF(dim, args.nodes,
                                  num_blocks=args.num_blocks,
                                  tail_bound=args.tail_bound,
                                  num_bins=args.num_bins,
                                  tails=args.tails)
                           ]
        dim -= 1
        if args.bnorm:
            transform_list += [
                transforms.BatchNorm(dim)
            ]
        # transform_list += [transforms.ReversePermutation(dim)]
        transform_list += [transforms.LULinear(dim)]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Running on {device}')
    transform = transforms.CompositeTransform(transform_list[:-1])
    base_dist = nflows.distributions.StandardNormal([out_dim])
    flow = flows.Flow(transform, base_dist).to(device)

    # Set up the dataset and training parameters
    val_batch_size = 1000

    testset = HyperCheckerboardDataset(args.n_test, inp_dim)

    optimizer = torch.optim.Adam(flow.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.ndata / args.batch_size * args.n_epochs, 0)

    # Train the model
    train_save = []
    val_save = []

    if args.load:
        transform.load_state_dict(torch.load(svo.save_name('model', extension='')))
    else:
        for epoch in range(args.n_epochs):
            start_time = time.time()
            trainset = HyperCheckerboardDataset(args.ndata, inp_dim)
            training_data = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                        num_workers=0)
            validset = HyperCheckerboardDataset(args.n_val, inp_dim)
            valid_data = torch.utils.data.DataLoader(validset, batch_size=val_batch_size, shuffle=True, num_workers=0)

            # Training
            running_loss = []
            for i, lo_data in enumerate(training_data, 0):
                data = lo_data.to(device)
                # zero the parameter gradients before calculating the losses
                optimizer.zero_grad()
                loss = -flow.log_prob(data).mean()
                loss.backward()
                if args.gclip > 0:
                    torch.nn.utils.clip_grad_value_(flow.parameters(), args.gclip)
                optimizer.step()
                scheduler.step()

                if i % args.monitor_interval == 0:
                    losses = -flow.log_prob(data).mean()
                    running_loss += [losses.item()]
                    s = '[{}, {}] {}'.format(epoch + 1, i + 1, running_loss[-1])

            # Update training loss trackers
            train_save += [np.mean(running_loss, 0)]

            # Validation
            val_loss = np.zeros((int(args.n_val / val_batch_size)))
            for i, data in enumerate(valid_data):
                val_loss[i] = -flow.log_prob(data.to(device)).mean().item()

            val_save += [np.mean(val_loss, 0)]

            with open(svo.save_name('timing', extension='txt'), 'w') as f:
                f.write('{}\n'.format(time.time() - start_time))

        torch.save(transform.state_dict(), svo.save_name('model', extension=''))

        print('Finished Training')

        # Plot the training information
        fig, ax = plt.subplots(1, 1, figsize=(20, 5))
        ax.plot(train_save, label='Train')
        ax.plot(val_save, '--', label='validation')
        ax.legend()
        ax.set_ylabel("loss")
        ax.set_xlabel("epoch")
        plt.savefig(svo.save_name('training'))

    flow.eval()

    # Plot the distribution of the encoding
    with torch.no_grad():
        encoding = flow.transform_to_noise(testset.data.to(device))
    plt.figure()
    getCrossFeaturePlot(encoding, svo.save_name('encoding'))

    # Look at the likelihoods of in and out of distribution samples
    if inp_dim < 16:
        test_bs = int(1e5)
    else:
        test_bs = int(1e4)
    n_batches = int(np.ceil(args.n_test / test_bs))
    scores_uniform = torch.empty((n_batches, test_bs))
    scores_data = torch.empty((n_batches, test_bs))
    uniform_encoding = torch.empty((n_batches, test_bs, dim))
    cpu = torch.device("cpu")

    def get_samples():
        uniform_sample = torch.distributions.uniform.Uniform(torch.zeros(inp_dim) - 4.,
                                                             torch.ones(inp_dim) * 4.,
                                                             validate_args=None).sample([test_bs])
        # uniform_sample = torch.distributions.multivariate_normal.MultivariateNormal(
        #     torch.zeros(inp_dim),
        #     torch.eye(inp_dim),
        #     validate_args=None
        # ).sample([test_bs])
        inlier_sample = HyperCheckerboardDataset(test_bs, inp_dim).data
        return uniform_sample.to(device), inlier_sample.to(device)

    with torch.no_grad():
        for i in range(n_batches):
            uniform_sample, inlier_sample = get_samples()
            scores_uniform[i] = flow.log_prob(uniform_sample).to(cpu)
            scores_data[i] = flow.log_prob(inlier_sample).to(cpu)
            uniform_encoding[i] = flow.transform_to_noise(uniform_sample)

    scores_uniform = scores_uniform.view(-1)
    scores_data = scores_data.view(-1)
    # Plot the distribution of the encoding
    plt.figure()
    getCrossFeaturePlot(encoding, svo.save_name('encoding_anomalies'), anomalies=uniform_encoding.view(-1, dim))

    print(f'p(ood)/p(in_dist) {2 * scores_uniform.mean() / scores_data.mean() - 1}')
    print(f'p(in_dist) - p(ood) {2 * (scores_data.mean() - scores_uniform.mean())}')

    with open(svo.save_name('', extension='npy'), 'wb') as f:
        np.save(f, tensor2numpy(scores_uniform))
        np.save(f, tensor2numpy(scores_data))


if __name__ == '__main__':
    checkerboard_test()
