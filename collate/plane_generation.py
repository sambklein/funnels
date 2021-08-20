import numpy as np
import matplotlib.pyplot as plt

from surVAE.utils.plotting import plot2Dhist, plot_likelihood


def plot_plane_grid():
    data_names = ['checkerboard', 'sine_wave']

    n_datasets = len(data_names)
    fig, axs = plt.subplots(n_datasets, 3, figsize=(18, 6 * n_datasets))

    for i, data in enumerate(data_names):
        with open(f'data/{data}.npy', 'rb') as f:
            encoding = np.load(f)
            samples = np.load(f)
            scores_uniform = np.load(f)
            uniform_sample = np.load(f)

        axs[i, 0].hist(encoding, bins=50)
        plot2Dhist(samples, axs[i, 1])
        plot_likelihood(uniform_sample, scores_uniform, axs[i, 2])

    title_font = 20
    axs[0, 0].set_title('Model Encoding', fontsize=title_font)
    axs[0, 1].set_title('Model Samples', fontsize=title_font)
    axs[0, 2].set_title('Model Density', fontsize=title_font)
    fig.tight_layout()
    fig.savefig('plane_figure.png')


if __name__ == '__main__':
    plot_plane_grid()
