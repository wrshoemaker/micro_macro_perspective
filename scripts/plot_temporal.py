import os
import numpy
import pickle
import sys
import math

import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scipy import stats, signal, special


import config
import data_utils
import plot_utils

target_dataset = 'caporaso_et_al'
target_host = 'M3'

c_blue='#1E90FF'
c_orange='#EB5900'
tick_labelsize=8

size_x, size_y = 4,4
lw=3
scatter_size=40
#target_


res_ret_dict = pickle.load(open('%sres_ret_dict.pickle' % config.data_directory, "rb"))
mle_dict = pickle.load(open('%smle_dict.pickle' % config.data_directory, "rb"))
dataset_all = ['david_et_al', 'poyet_et_al', 'caporaso_et_al']



def make_corr_phylo_dict():

    print('ww')


def plot_sojourn_time():

    sys.stderr.write("Plottting sojourn, residence, and return times.....\n")

    fig, ax = plt.subplots(figsize=(size_x, size_y))

    res_ret_ls = ['-', ':']
    for res_ret_bool_idx, res_ret_bool in enumerate([True, False]):

        res_ret_dict_subset = res_ret_dict[target_dataset][target_host][res_ret_bool]

        x_range = numpy.asarray(res_ret_dict_subset['x_range_pdf'])
        days = numpy.asarray(res_ret_dict_subset['days_pdf'])

        sort_idx = numpy.argsort(x_range)
        x_range = x_range[sort_idx]
        days = days[sort_idx]
        print(sum(days))
        #smf = 1 - numpy.cumsum(days)
        smf = numpy.cumsum(days[::-1])[::-1]
        ax.plot(x_range, smf, lw=lw, c=c_orange, ls=res_ret_ls[res_ret_bool_idx])

    
    # sojourn
    days_run_lengths_all = []
    for asv in  mle_dict[target_dataset][target_host].keys():
        days_run_lengths = numpy.asarray(mle_dict[target_dataset][target_host][asv]['days_run_lengths'])
        days_run_lengths_all.append(days_run_lengths)


    sojourn_range = numpy.unique(numpy.concatenate(days_run_lengths_all))
    mixture_pdf = numpy.zeros_like(sojourn_range, dtype=float)
    
    for arr in days_run_lengths_all:
        unique, counts = numpy.unique(arr, return_counts=True)
        pdf = counts/sum(counts)
        indices = numpy.searchsorted(sojourn_range, unique)
        mixture_pdf[indices] += len(arr) * pdf

    mixture_pdf = mixture_pdf/sum(mixture_pdf)
    #mixture_smf = 1 - numpy.cumsum(mixture_pdf)
    # >=
    mixture_smf = numpy.cumsum(mixture_pdf[::-1])[::-1]
    ax.plot(sojourn_range, mixture_smf, lw=lw, c=c_blue, ls='-')

    legend_elements = [Line2D([0], [0], color=c_blue, lw=4, ls='-', label='Sojourn time, ' + r'$T_{\mathrm{sojourn}}$'),
                       Line2D([0], [0], color=c_orange, lw=4, ls='-', label='Residence time, ' + r'$T_{\mathrm{residence}}$'),
                       Line2D([0], [0], color=c_orange, lw=4, ls=':', label='Return time, ' + r'$T_{\mathrm{return}}$')]

    ax.legend(handles=legend_elements, loc='lower left', fontsize=11)

    ax.xaxis.set_tick_params(labelsize=tick_labelsize)
    ax.yaxis.set_tick_params(labelsize=tick_labelsize)

    ax.set_xlim([1, 412])
    ax.set_ylim([min(mixture_smf), 1])

    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)

    ax.set_xlabel("Time (days), " + r'$t$', fontsize=14)
    ax.set_ylabel("Fraction observations " + r'$\geq \, t$', fontsize=14)

    
    fig.subplots_adjust(hspace=0.25, wspace=0.15)
    fig_name = "%sfig2_sojourn_time.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def plot_mean_vs_delta():

    sys.stderr.write("Plottting mean abundance vs. change in abundance.....\n")

    x_mean_all = []
    mean_abs_delta_all = []
    for asv in  mle_dict[target_dataset][target_host].keys():

        x_mean = mle_dict[target_dataset][target_host][asv]['x_mean']
        rel_abundance = numpy.asarray(mle_dict[target_dataset][target_host][asv]['rel_abundance'])
        mean_abs_delta = numpy.mean(numpy.abs(rel_abundance[1:] - rel_abundance[:-1]))

        x_mean_all.append(x_mean)
        mean_abs_delta_all.append(mean_abs_delta)

    
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(numpy.log10(x_mean_all), numpy.log10(mean_abs_delta_all))
    slope_env=1
    slope_demog=0.66

    x_range =  numpy.linspace(min(numpy.log10(x_mean_all)), max(numpy.log10(x_mean_all)), 10000)
    y_pred_env = slope_env*x_range + intercept
    y_pred_demog = slope_demog*x_range + intercept
    
    fig, ax = plt.subplots(figsize=(size_x, size_y))

    ax.plot(10**x_range, 10**y_pred_env, ls='-', lw=lw, c='k', zorder=1)
    ax.plot(10**x_range, 10**y_pred_demog, ls=':', lw=lw, c='k', zorder=1)


    ax.scatter(x_mean_all, mean_abs_delta_all, s=scatter_size, c=c_blue, zorder=2)
    #print(min(x_mean_all), max(x_mean_all))
    ax.xaxis.set_tick_params(labelsize=tick_labelsize)
    ax.yaxis.set_tick_params(labelsize=tick_labelsize)
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)

    ax.set_xlabel("Mean abundance, " + r'$\bar{x}_{i}$', fontsize=14)
    ax.set_ylabel("Mean change in abundance b/w\nobservations,  " + r'$\overline{|\Delta x_{i}|}$', fontsize=14)


    legend_elements = [Line2D([0], [0], color='k', lw=4, ls='-', label='Environmental'),
                       Line2D([0], [0], color='k', lw=4, ls=':', label='Demographic')]

    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, title = "Noise type")

    fig.subplots_adjust(hspace=0.25, wspace=0.15)
    fig_name = "%sfig2_mean_vs_delta.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()
        



def plot_corr_pdf(min_occupancy=0.8):

    for dataset_idx, dataset in enumerate(dataset_all):
        
        host_all = list(mle_dict[dataset].keys())
        host_all.sort()

        for host_idx, host in enumerate(host_all):

            print('www')

            #asv_all = list(mle_dict[dataset][host].keys())
            #asvs_all = []
            #print(asv_all)





def plot_hust(min_n_g=10):

    sys.stderr.write("Plottting time vs. variance in growth.....\n")

    days = mle_dict[target_dataset][target_host][list(mle_dict[target_dataset][target_host].keys())[0]]['days']

    # max(days) - min(days)
    delta_t_range = numpy.arange(1, 101)
    delta_t_all = []
    var_g_all = []

    for delta_t in delta_t_range:

        g_all = []

        for asv in  mle_dict[target_dataset][target_host].keys():

            rel_abundance = numpy.asarray(mle_dict[target_dataset][target_host][asv]['rel_abundance'])

            t_mid, g = data_utils.discretized_growth_rate(days, rel_abundance, delta_t, divide_by_delta_t=False)
            g_all.append(g)

        g_all = numpy.concatenate(g_all)
        if len(g) > min_n_g:

            delta_t_all.append(delta_t)
            var_g_all.append(numpy.var(g_all, ddof=1))


    slope, intercept, r_value, p_value, std_err = stats.linregress(numpy.log10(delta_t_all), numpy.log10(var_g_all))
    sys.stderr.write("Hurst exponent = %.6f \n" % slope)

    x_range =  numpy.linspace(min(numpy.log10(delta_t_all)), max(numpy.log10(delta_t_all)), 10000)
    y_pred = slope*x_range + intercept

    fig, ax = plt.subplots(figsize=(size_x, size_y))

    ax.plot(10**x_range, 10**y_pred, ls=':', lw=lw, c='k', zorder=2)
    ax.scatter(delta_t_all, var_g_all, s=scatter_size, c=c_blue, zorder=1)

    ax.xaxis.set_tick_params(labelsize=tick_labelsize)
    ax.yaxis.set_tick_params(labelsize=tick_labelsize)

    ax.set_ylim([0.1, 10 ])
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)


    ax.set_xlabel("Time b/w observations (days), " + r'$\Delta t$', fontsize=14)
    ax.set_ylabel("Variance of change\nin abundances, " + r'$\mathrm{Var}(\Delta x_{i} | \Delta t)$', fontsize=14)


    fig.subplots_adjust(hspace=0.25, wspace=0.15)
    fig_name = "%sfig2_hurst.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()
        


def plot_g_dist(min_n_g=200):

    sys.stderr.write("Plottting growth rate distribution.....\n")

    days = mle_dict[target_dataset][target_host][list(mle_dict[target_dataset][target_host].keys())[0]]['days']
    delta_t_range = numpy.arange(1, 101)

    fig, ax = plt.subplots(figsize=(size_x, size_y))

    c_g = ["#63a0c8", "#1f78b4", "#08306b"]

    for delta_t_idx, delta_t in enumerate([1, 5, 10]):

        g_all = []

        for asv in  mle_dict[target_dataset][target_host].keys():

            rel_abundance = numpy.asarray(mle_dict[target_dataset][target_host][asv]['rel_abundance'])

            t_mid, g = data_utils.discretized_growth_rate(days, rel_abundance, delta_t, divide_by_delta_t=True)
            g_all.append(g)


        g_all = numpy.concatenate(g_all)
        g_unique = numpy.sort(numpy.unique(g_all))
        n_g_all = len(g_all)

        if n_g_all <= min_n_g:
            continue

        print(delta_t, numpy.var(g_all)/numpy.abs(numpy.mean(g_all)))

        print(g_all)

        #survival = numpy.array([(g_all >= v).sum() / n_g_all for v in g_unique])

        hist, bin_edges = numpy.histogram(g_all, bins=15, density=False)
        hist = hist / hist.sum()
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

        #ax.bar(bin_midpoints, hist, width=bin_edges[1]-bin_edges[0], c='k', alpha=1)
        ax.step(bin_midpoints, hist, where='mid', lw=lw, c=c_g[delta_t_idx], label=r'$\Delta t = $' + str(delta_t))
        
        
    #ax.set_xscale('log', base=10)
    ax.set_xlim([-6, 6 ])
    ax.set_ylim([0.001, 1])
    ax.set_yscale('log', base=10)

    ax.xaxis.set_tick_params(labelsize=tick_labelsize)
    ax.yaxis.set_tick_params(labelsize=tick_labelsize)

    ax.set_xlabel("Discretized growth rate, " + r'$g_{i}(\Delta t)$', fontsize=14)
    ax.set_ylabel("Probability", fontsize=14)

    ax.axvline(x=0, lw=lw, ls=':', c='k', label='Stationarity', zorder=2)
    ax.legend(loc='upper left', fontsize=10)



    fig.subplots_adjust(hspace=0.25, wspace=0.15)
    fig_name = "%sfig2_g_dist.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



if __name__ == "__main__":
 
    sys.stderr.write("Plottting temporal patterns.....\n")


    plot_g_dist()


    #plot_corr_pdf()

    #make_corr_phylo_dict()