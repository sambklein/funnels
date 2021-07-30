# Some plotting functions
import colorsys

import numpy
import numpy as np
import torch
from matplotlib import pyplot as plt

import seaborn as sns
from .torch_utils import tensor2numpy, shuffle_tensor


def get_bins(data, nbins=20):
    max_ent = data.max().item()
    min_ent = data.min().item()
    return np.linspace(min_ent, max_ent, num=nbins)


def get_mask(x, bound):
    return np.logical_and(x > bound[0], x < bound[1])


def apply_bound(data, bound):
    mask = np.logical_and(get_mask(data[:, 0], bound), get_mask(data[:, 1], bound))
    return data[mask, 0], data[mask, 1]


def plot2Dhist(data, ax, bins=50, bounds=None):
    if bounds:
        x, y = apply_bound(data, bounds)
    else:
        x = data[:, 0]
        y = data[:, 1]
    count, xbins, ybins = np.histogram2d(x, y, bins=bins)
    count[count == 0] = np.nan
    ax.imshow(count.T,
              origin='lower', aspect='auto',
              extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
              )

def getCrossFeaturePlot(data, nm):
    nfeatures = data.shape[1]
    fig, axes = plt.subplots(nfeatures, nfeatures,
                             figsize=(np.clip(5 * nfeatures + 2, 5, 22), np.clip(5 * nfeatures - 1, 5, 20)))
    if nfeatures ==1:
        axes.hist(tensor2numpy(data))
    else:
        for i in range(nfeatures):
            for j in range(nfeatures):
                if i == j:
                    axes[i, i].hist(tensor2numpy(data[:, i]))
                elif i < j:
                    bini = get_bins(data[:, i])
                    binj = get_bins(data[:, j])
                    axes[i, j].hist2d(tensor2numpy(data[:, i]), tensor2numpy(data[:, j]), bins=[bini, binj],
                                      density=True, cmap='Reds')
                else:
                    axes[i, j].set_visible(False)
    fig.tight_layout()
    plt.savefig(nm)
