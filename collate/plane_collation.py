import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from surVAE.utils.plotting import plot2Dhist, plot_likelihood


def plot_data(data_names, save_name, bound=1., bins=50):
    n_datasets = len(data_names)
    fig, axs = plt.subplots(n_datasets, 3, figsize=(18, 6 * n_datasets))

    if axs.ndim == 1:
        axs = axs[np.newaxis, :]

    for i, data in enumerate(data_names):
        with open(f'data/{data}.npy', 'rb') as f:
            encoding = np.load(f)
            samples = np.load(f) / bound
            scores_uniform = np.load(f)
            uniform_sample = np.load(f) / bound

        if encoding.shape[1] == 1:
            axs[i, 0].hist(encoding, bins=bins, density=True, alpha=0.5)

            mu, std = norm.fit(encoding)
            xmin, xmax = axs[i, 0].get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, 0, 1)
            axs[i, 0].plot(x, p, 'k', linewidth=3, alpha=0.5)
            axs[i, 0].set_title(f'Mean {mu:.2f}, sigma {std:.2f}')

        else:
            plot2Dhist(encoding, axs[i, 0], bounds=[-4, 4], bins=bins)
        plot2Dhist(samples, axs[i, 1], bounds=[-1, 1], bins=bins)
        plot_likelihood(uniform_sample, scores_uniform, axs[i, 2], n_bins=bins)
        for j in range(3):
            axs[i, j].set_box_aspect(1)
            axs[i, j].tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                left=False,
                labelbottom=False,
                labelleft=False
            )

    title_font = 42
    axs[0, 0].set_title('Model encoding', fontsize=title_font)
    axs[0, 1].set_title('Model samples', fontsize=title_font)
    axs[0, 2].set_title('Model density', fontsize=title_font)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f'{save_name}.png')
    plt.close(fig)


def plot_plane_grid():
    # plot_data([f'_realNVP_{i}' for i in range(4, 8)], 'Funnel_NVP')
    # plot_data(['_realNVP_8'] + [f'_realNVP_{i}' for i in range(1, 4)], 'Real_NVP')
    # plot_data([f'_rqNSF_{i}' for i in range(4, 8)], 'Funnel_RQnsf_bounded')
    # plot_data(['_rqNSF_8'] + [f'_rqNSF_{i}' for i in range(1, 4)], 'RQnsf_bounded')
    # plot_data([f'_rqNSFfour_{i}' for i in range(4, 8)], 'Funnel_RQnsf', bound=4.)
    # plot_data(['_rqNSFfour_8'] + [f'_rqNSFfour_{i}' for i in range(1, 4)], 'RQnsf', bound=4.)
    # plot_data([f'_rqNSFfour_{i}' for i in range(4, 5)], 'quantitative_image', bound=4.)

    # [plot_data([f'_plane_{i}'], f'quantitative_image_{i}', bound=4., bins=1000) for i in range(1, 5)]
    [plot_data([f'_plane_2_{i}'], f'quantitative_image_{i}', bound=4., bins=1000) for i in range(1, 3)]


if __name__ == '__main__':
    plot_plane_grid()
