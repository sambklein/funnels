import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from surVAE.utils.plotting import plot2Dhist, plot_likelihood


def plot_data(data_names, save_name, bound=1.):
    n_datasets = len(data_names)
    fig, axs = plt.subplots(n_datasets, 3, figsize=(18, 6 * n_datasets))

    for i, data in enumerate(data_names):
        with open(f'data/{data}.npy', 'rb') as f:
            encoding = np.load(f)
            samples = np.load(f) / bound
            scores_uniform = np.load(f)
            uniform_sample = np.load(f) / bound

        if encoding.shape[1] == 1:
            axs[i, 0].hist(encoding, bins=50, density=True, alpha=0.5)

            mu, std = norm.fit(encoding)
            xmin, xmax = axs[i, 0].get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, 0, 1)
            axs[i, 0].plot(x, p, 'k', linewidth=3)
            axs[i, 0].set_title(f'Mean {mu:.2f}, sigma {std:.2f}')

        else:
            plot2Dhist(encoding, axs[i, 0], bounds=[-4, 4])
        plot2Dhist(samples, axs[i, 1], bounds=[-1, 1])
        plot_likelihood(uniform_sample, scores_uniform, axs[i, 2])
        for j in range(3):
            axs[i, j].set_box_aspect(1)
        # TODO: plut a gaussian distribution over the top

    title_font = 20
    axs[0, 0].set_title('Model Encoding', fontsize=title_font)
    axs[0, 1].set_title('Model Samples', fontsize=title_font)
    axs[0, 2].set_title('Model Density', fontsize=title_font)
    fig.tight_layout()
    fig.savefig(f'{save_name}.png')
    plt.close(fig)


def plot_plane_grid():
    # plot_data([f'_realNVP_{i}' for i in range(4, 8)], 'Funnel_NVP')
    # plot_data(['_realNVP_8'] + [f'_realNVP_{i}' for i in range(1, 4)], 'Real_NVP')
    # plot_data([f'_rqNSF_{i}' for i in range(4, 8)], 'Funnel_RQnsf_bounded')
    # plot_data(['_rqNSF_8'] + [f'_rqNSF_{i}' for i in range(1, 4)], 'RQnsf_bounded')
    plot_data([f'_rqNSFfour_{i}' for i in range(4, 8)], 'Funnel_RQnsf', bound=4.)
    plot_data(['_rqNSFfour_8'] + [f'_rqNSFfour_{i}' for i in range(1, 4)], 'RQnsf', bound=4.)


if __name__ == '__main__':
    plot_plane_grid()
