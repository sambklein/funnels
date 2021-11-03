import glob

import numpy.lib.recfunctions as rfn

from nflows.utils import get_num_parameters
from nflows import transforms
from nflows import flows
import nflows

import torch
import numpy as np
import matplotlib.pyplot as plt

import time

from surVAE.data.base import BasicData
from surVAE.data.plane import load_plane_dataset
from surVAE.models.flows import get_transform
from surVAE.models.sur_flows import SurNSF
from surVAE.data.hyper_dim import HyperCheckerboardDataset

import argparse

from surVAE.utils.io import save_object, get_top_dir, on_cluster
from surVAE.utils.physics_utils import calculate_bmjj
from surVAE.utils.plotting import getCrossFeaturePlot, plot2Dhist, plot_likelihood, get_bins, get_weights
from surVAE.utils.torch_utils import tensor2numpy


def parse_args():
    parser = argparse.ArgumentParser()

    # Saving
    parser.add_argument('-d', '--outputdir', type=str, default='ttbar_local',
                        help='Choose the base output directory')
    parser.add_argument('-n', '--outputname', type=str, default='local',
                        help='Set the output name directory')
    parser.add_argument('--load', type=int, default=1,
                        help='Load a model?')

    # Model set up
    parser.add_argument('--nodes', type=int, default=16,
                        help='The number of nodes in each layer used to learn the flow parameters.')
    parser.add_argument('--num_blocks', type=int, default=2,
                        help='The number of layers used to learn the flow placements.')
    parser.add_argument('--nstack', type=int, default=2,
                        help='The number of flow layers.')
    parser.add_argument('--tails', type=str, default='linear',
                        help='The tail function to apply.')
    parser.add_argument('--tail_bound', type=float, default=1.2,
                        help='The tail bound.')
    parser.add_argument('--num_bins', type=int, default=10,
                        help='The number of bins to use in the RQ-NSF.')
    parser.add_argument('--num_add', type=int, default=1,
                        help='The number of additional layers to add.')
    parser.add_argument('--add_sur', type=int, default=1,
                        help='Whether to make the additional layers surVAE layers.')
    parser.add_argument('--splines', type=int, default=1,
                        help='Use RQ-NSF if true, else Real NVP.')

    # Dataset and training parameters
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Whether to make the additional layers surVAE layers.')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Whether to make the additional layers surVAE layers.')
    parser.add_argument('--lr', type=float, default=0.001,
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


def read_ttbar_file(file):
    with open(file, 'rb') as f:
        array = np.load(f)
    unstruct_array = rfn.structured_to_unstructured(array)
    four_vector_mask = np.zeros(unstruct_array.shape[1], dtype='bool')
    four_vector_mask[:25] = np.mod(np.arange(25) + 1, 5) != 0
    four_vectors = unstruct_array[:, four_vector_mask]
    tags = unstruct_array[:, ~four_vector_mask][:, :-2]
    et_info = unstruct_array[:, -2:]
    names = np.array(array.dtype.names)
    four_vector_mask[-2:] = True
    return four_vectors, tags, et_info, names[four_vector_mask]


def read_ttbar():
    if on_cluster():
        files = glob.glob(f'{get_top_dir()}/surVAE/data/downloads/ttbar/*.npy')
    else:
        files = [f'{get_top_dir()}/surVAE/data/downloads/ttbar/events_merged.npy']
    four_vectors = []
    tags = []
    et_info = []
    for file in files:
        fv, tg, et, names = read_ttbar_file(file)
        four_vectors += [fv]
        tags += [tg]
        et_info += [et]
    return np.concatenate(four_vectors), np.concatenate(tags), np.concatenate(et_info), names


def ttbar_experiment():
    # Parse args
    args = parse_args()

    # Set up the dataset and training parameters
    val_batch_size = 1000

    four_vectors, tags, et_info, names = read_ttbar()

    dataset = np.concatenate((four_vectors, et_info), 1)
    n_data = dataset.shape[0]
    train_split = 0.9
    n_train = int(n_data * train_split)
    train_data = BasicData(dataset[:n_train])
    valid_set = BasicData(dataset[n_train:])
    # As long as this isn't used for anything it is okay
    test_set = valid_set
    # Normalise the data
    valid_set.normalize(facts=train_data.get_and_set_norm_facts(normalize=True))

    training_data = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                num_workers=0)
    valid_data = torch.utils.data.DataLoader(valid_set, batch_size=val_batch_size, shuffle=True, num_workers=0)

    # Set up saving and change the name of some args
    svo = save_object(args.outputdir, args.outputname, args=args)
    inp_dim = dataset.shape[1]
    out_dim = inp_dim - args.num_add * args.add_sur
    nstack = args.nstack + args.num_add * (1 - args.add_sur)
    spline = args.splines
    nodes = args.nodes if args.add_sur else int(1.45 * args.nodes)

    # Set up and define the model
    transform_list = [get_transform(inp_dim, nodes=nodes,
                                    nstack=nstack,
                                    num_blocks=args.num_blocks,
                                    tail_bound=args.tail_bound,
                                    num_bins=args.num_bins,
                                    tails=args.tails,
                                    spline=spline)]

    # Add the surVAE layers
    dim = inp_dim
    sur_layers = []
    for _ in range(args.num_add * args.add_sur):
        sur_layers += [SurNSF(dim, args.nodes,
                              num_blocks=args.num_blocks,
                              tail_bound=args.tail_bound,
                              num_bins=args.num_bins,
                              tails=args.tails,
                              spline=spline)
                       ]
        dim -= 1
        if args.bnorm:
            sur_layers += [
                transforms.BatchNorm(dim)
            ]
        # transform_list += [transforms.ReversePermutation(dim)]
        sur_layers += [transforms.LULinear(dim)]

    transform_list += sur_layers[:-1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Running on {device}')
    transform = transforms.CompositeTransform(transform_list)
    base_dist = nflows.distributions.StandardNormal([out_dim])
    flow = flows.Flow(transform, base_dist).to(device)
    print(f'There are {get_num_parameters(flow)} params')

    optimizer = torch.optim.Adam(flow.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_data / args.batch_size * args.n_epochs, 0)

    # Train the model
    train_save = []
    val_save = []

    if args.load:
        transform.load_state_dict(torch.load(svo.save_name('model', extension='')))
    else:
        for epoch in range(args.n_epochs):
            flow.train()
            start_time = time.time()

            # Training
            running_loss = []
            for i, lo_data in enumerate(training_data, 0):
                if spline:
                    lo_data = lo_data * args.tail_bound
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

            # Update training loss trackers
            train_save += [np.mean(running_loss, 0)]
            flow.eval()

            # Validation
            mean_val_loss = 0
            for i, data in enumerate(valid_data):
                mean_val_loss += -flow.log_prob(data.to(device)).mean().item()
            val_save += [mean_val_loss / np.ceil(len(valid_set) / val_batch_size)]

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
    cpu = torch.device("cpu")
    with torch.no_grad():
        test_data = test_set.data.to(device)
        if spline:
            test_data = test_data * args.tail_bound
        encoding = flow.transform_to_noise(test_data).to(cpu)
    plt.figure()
    getCrossFeaturePlot(encoding, svo.save_name('encoding'))

    # Plot a selection of generated samples
    with torch.no_grad():
        samples = tensor2numpy(flow.sample(args.n_test))
    data_dim = test_set.data.shape[1]
    ncols = int(np.ceil(data_dim / 3))
    nrows = int(np.ceil(data_dim / ncols))
    fig, axs_ = plt.subplots(nrows, ncols, figsize=(5 * ncols + 8, 5 * nrows + 2))
    axs = fig.axes
    for i in range(data_dim):
        bins = get_bins(test_set[:, i])
        og_data = tensor2numpy(test_set[:, i])
        axs[i].hist(og_data, label='original', alpha=0.5, density=True, bins=bins,
                    weights=get_weights(og_data))
        # Plot samples drawn from the model
        axs[i].hist(samples[:, i], label='samples', alpha=0.5, density=True, bins=bins,
                    weights=get_weights(samples[:, i]))
        axs[i].set_title(names[i])
        # axs[i].legend()
    for j in range(i + 1, nrows * ncols):
        axs[j].set_visible(False)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.1, 0.88), frameon=False)
    fig.savefig(svo.save_name('samples'))

    def get_bmjj(data):
        return calculate_bmjj(data[:, 0:4], data[:, 4:8], data[:, 8:12])

    bmjj_samples = get_bmjj(tensor2numpy(train_data.unnormalize(torch.tensor(samples))))
    bmjj = get_bmjj(dataset)
    bmjj_samples = np.concatenate((bmjj_samples[:1500], bmjj))
    bins = get_bins(bmjj, nbins=60)
    fig, ax = plt.subplots(1, 1, figsize=(20, int(20 / 1.618)))
    ax.hist(bmjj, label='data', alpha=0.5, density=True, bins=bins, weights=get_weights(bmjj))
    # Plot samples drawn from the model
    ax.hist(bmjj_samples, label='samples', alpha=0.5, density=True, bins=bins, weights=get_weights(bmjj_samples))
    ax.legend(frameon=False, fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=30, length=12)
    ax.tick_params(axis='both', which='minor', labelsize=30)
    # plt.tick_params(
    #     axis='y',
    #     which='both',
    #     bottom=False,
    #     top=False,
    #     left=False,
    #     right=False,
    #     labelbottom=False,
    #     labelleft=False)
    ax.set_yticklabels([])
    ax.set_xlim(right=1200)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})
    ax.set_xlabel(r'm${}_{\mathrm{Top}, \mathrm{had}}$ [GeV]', fontsize=42)
    ax.set_ylabel(r'a.u', fontsize=42)
    fig.savefig(svo.save_name('bmjj'))

    # Plot the model density
    test_bs = int(1e5)
    n_batches = int(np.ceil(args.n_test / test_bs))
    scores_uniform = torch.empty((n_batches, test_bs))
    uniform_sample = torch.empty((n_batches, test_bs, inp_dim))

    with torch.no_grad():
        for i in range(n_batches):
            uniform_sample[i] = torch.distributions.uniform.Uniform(torch.zeros(inp_dim) - 1,
                                                                    torch.ones(inp_dim),
                                                                    validate_args=None).sample([test_bs])
            if spline:
                uniform_sample[i] *= args.tail_bound
            scores_uniform[i] = flow.log_prob(uniform_sample[i].to(device)).to(cpu)

    scores_uniform = scores_uniform.view(-1)
    uniform_sample = uniform_sample.view(-1, inp_dim)

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    plot_likelihood(uniform_sample, scores_uniform, ax)
    fig.tight_layout()
    plt.savefig(svo.save_name('likelihood'))
    plt.clf()

    with open(svo.save_name('', extension='npy'), 'wb') as f:
        np.save(f, encoding)
        np.save(f, samples)
        np.save(f, tensor2numpy(scores_uniform))
        np.save(f, tensor2numpy(uniform_sample))


if __name__ == '__main__':
    ttbar_experiment()
