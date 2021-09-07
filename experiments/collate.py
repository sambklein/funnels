import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle
import glob

from surVAE.utils.io import get_top_dir, save_object


class collater():

    def __init__(self, xvar, max_val=None):
        self.xvar = xvar
        self.max_val = max_val

    def add_dir(self, add_sur, directory, name, color):
        top_dir = get_top_dir()
        sv_dir = top_dir + '/images' + '/' + directory + '/'
        xvar = self.xvar
        files = glob.glob(sv_dir + '/*.json')
        train_info = np.zeros((len(files), 1))
        xs = []

        for i, info_name in enumerate(files):
            svo = save_object(directory)
            info_dict = svo.read_experiment(info_name)
            if info_dict['add_sur'] == add_sur:
                npy_file = svo.save_name('', extension='npy')
                try:
                    with open(npy_file, 'rb') as f:
                        scores_uniform = np.load(f)
                        scores_data = np.load(f)
                    train_info[i, 0] = (np.mean(scores_data) - np.mean(scores_uniform)) / abs(np.mean(scores_data))
                    xs += [info_dict[xvar]]
                except:
                    print(f'Missing {npy_file}')
                    xs += [-1]
            else:
                xs += [-1]

        xvals = sorted(list(set(xs)))
        n_exp = len(xvals)
        plt_arr = np.zeros((n_exp, 3))
        drp = False
        for i, x in enumerate(xvals):
            if x != -1:
                mx = [xx == x for xx in xs]
                inf = train_info[mx]
                plt_arr[i, 0] = np.mean(inf)
                plt_arr[i, 1] += np.std(inf)
                # plt_arr[i, 0] = np.percentile(inf, 50)
                # plt_arr[i, 1] = np.percentile(inf, 33)
                # plt_arr[i, 2] = np.percentile(inf, 69)
            else:
                drp = True
        if drp:
            xvals = xvals[1:]
            plt_arr = plt_arr[1:]

        if self.max_val is not None:
            mx = [i <= self.max_val for i in xvals]
            xvals = np.array(xvals)[mx]
            plt_arr = plt_arr[mx]
        plt.plot(xvals, plt_arr[:, 0], label=name, color=color)
        plt.fill_between(xvals, plt_arr[:, 0] - plt_arr[:, 1], plt_arr[:, 0] + plt_arr[:, 1],
                         color=color, alpha=0.3, linewidth=0.1)
        # plt.fill_between(xvals, plt_arr[:, 1], plt_arr[:, 2],
        #                  color=color, alpha=0.3, linewidth=0.1)
        # plt.plot(xvals, plt_arr[:, 0], label=name, color=color)
        return xvals, plt_arr


def likelihood_dim():
    plotter = collater('inp_dim', max_val=5)
    plt.figure()
    # add_dir(1, 'sur_test', 'SurVAE', 'indianred')
    # xvals, _ = add_dir(0, 'test_nsf', 'Flow', 'royalblue')
    # add_dir(1, 'sur1', 'SurVAE', 'indianred')
    # xvals, _ = plotter.add_dir(0, 'nsf1', 'Flow', 'royalblue')
    # xvals, _ = plotter.add_dir(1, 'sur1_no_pref', 'SurNoPref', 'green')
    xvals, _ = plotter.add_dir(1, 'sur_nsf_bn', 'SurVAE', 'indianred')
    xvals, _ = plotter.add_dir(0, 'sur_nsf_bn', 'Flow', 'royalblue')
    plt.legend()
    plt.xlabel('Data Dimension', fontsize=15)
    # plt.ylabel('Likelihood Difference', fontsize=15)
    plt.ylabel(r'$ \frac{ \left\langle \log \left(  p_\mathrm{Data} \right) \right\rangle - '
               r'\left\langle \log \left( p_{\mathrm{Data}} + p_{\mathrm{outliers}} \right) \right\rangle } '
               r'{ | \left\langle \log \left(  p_\mathrm{Data} \right) \right\rangle | }$',
               fontsize=15)
    # plt.xscale('log')
    # plt.xlim([min(x), max(x)])
    plt.xticks(xvals, fontsize=12)
    plt.yscale('symlog')
    plt.yticks(fontsize=12)
    top_dir = get_top_dir()
    sv_dir = top_dir + '/images/'
    plt.tight_layout()
    plt.savefig(sv_dir + 'summary_likelihood_dim.png')

def likelihood_depth():
    plotter = collater('num_add')
    plt.figure()
    # xvals, _ = plotter.add_dir(0, 'sur_nsf_4', 'Flow', 'royalblue')
    # xvals, _ = plotter.add_dir(1, 'sur_nsf_4', 'SurVAE', 'indianred')
    # xvals, _ = plotter.add_dir(0, 'sur_nsf_6', 'Flow', 'royalblue')
    # xvals, _ = plotter.add_dir(1, 'sur_nsf_6', 'SurVAE', 'indianred')
    # xvals, _ = plotter.add_dir(0, 'sur_nsf_6_normal', 'Flow', 'royalblue')
    # xvals, _ = plotter.add_dir(1, 'sur_nsf_6_normal', 'SurVAE', 'indianred')
    xvals, _ = plotter.add_dir(0, 'AD_FUNNEL', 'Flow', 'royalblue')
    xvals, _ = plotter.add_dir(1, 'AD_FUNNEL', 'SurVAE', 'indianred')
    plt.legend()
    plt.xlabel('additional layers', fontsize=15)
    # plt.ylabel('Likelihood Difference', fontsize=15)
    plt.ylabel(r'$ \frac{ \left\langle \log \left(  p_\mathrm{Data} \right) \right\rangle - '
               r'\left\langle \log \left( p_{\mathrm{Data}} + p_{\mathrm{outliers}} \right) \right\rangle } '
               r'{ | \left\langle \log \left(  p_\mathrm{Data} \right) \right\rangle | }$',
               fontsize=15)
    # plt.xscale('log')
    # plt.xlim([min(x), max(x)])
    plt.xticks(xvals, fontsize=12)
    plt.yscale('symlog')
    plt.yticks(fontsize=12)
    top_dir = get_top_dir()
    sv_dir = top_dir + '/images/'
    plt.tight_layout()
    plt.savefig(sv_dir + 'summary_likelihood_depth.png')


if __name__ == '__main__':
    likelihood_dim()
    likelihood_depth()
