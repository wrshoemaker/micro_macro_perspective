import os
import numpy
import pickle
import sys
import math
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scipy import stats, signal, special


import config
import data_utils
import plot_utils
import stats_utils

target_dataset = 'caporaso_et_al'
target_host = 'M3'
min_n_autocorr_values = 20
target_asv = 'TACGGAGGATCCGAGCGTTATCCGGATTTATTGGGTTTAAAGGGAGCGTAGGTGGATTGTTAAGTCAGTTGTGAAAGTTTGCGGCTCAACCGTAAAATTGCAGTTGAAACTGGCAGTCTT'

c_blue=plot_utils.c_blue
c_orange=plot_utils.c_orange
tick_labelsize=plot_utils.tick_labelsize

size_x, size_y = plot_utils.size_x, plot_utils.size_y
lw=plot_utils.lw
scatter_size=plot_utils.scatter_size
#target_


res_ret_dict = pickle.load(open('%sres_ret_dict.pickle' % config.data_directory, "rb"))
mle_dict = pickle.load(open('%smle_dict.pickle' % config.data_directory, "rb"))
dataset_all = ['david_et_al', 'poyet_et_al', 'caporaso_et_al']




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

    ax.legend(handles=legend_elements, loc='lower left', fontsize=13)

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

    ax.set_xlabel("Mean relative abundance, " + r'$\bar{x}_{i}$', fontsize=14)
    ax.set_ylabel("Mean change in\nabundance b/w observations, " + r'$\overline{|\Delta x_{i}|}$', fontsize=14)


    legend_elements = [Line2D([0], [0], color='k', lw=4, ls='-', label='Environment'),
                       Line2D([0], [0], color='k', lw=4, ls=':', label='Demography')]

    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, title = "Noise source", title_fontsize=11)

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

            t_mid, x, g = data_utils.discretized_growth_rate(days, rel_abundance, delta_t, divide_by_delta_t=False)
            g_all.append(g)

        g_all = numpy.concatenate(g_all)
        if len(g) > min_n_g:

            delta_t_all.append(delta_t)
            var_g_all.append(numpy.var(g_all, ddof=1))


    slope, intercept, r_value, p_value, std_err = stats.linregress(numpy.log10(delta_t_all), numpy.log10(var_g_all))
    sys.stderr.write("Hurst exponent = %.6f \n" % slope)

    x_range =  numpy.linspace(min(numpy.log10(delta_t_all)), max(numpy.log10(delta_t_all)), 10000)
    y_pred = slope*x_range + intercept

    y_pred_diff = 0.5*x_range + intercept

    fig, ax = plt.subplots(figsize=(size_x, size_y))

    ax.plot(10**x_range, 10**y_pred_diff, ls=':', lw=lw, c='k', zorder=2, label=r'$H = 0.5$' + '   (diffusion)')
    ax.plot(10**x_range, 10**y_pred, ls='--', lw=lw, c='k', zorder=2, label=r'$H \approx $' +  str(round(slope, 2)) + ' (subdiffusion)')

    ax.scatter(delta_t_all, var_g_all, s=scatter_size, c=c_blue, zorder=1)

    ax.xaxis.set_tick_params(labelsize=tick_labelsize)
    ax.yaxis.set_tick_params(labelsize=tick_labelsize)

    ax.set_ylim([0.6, 10])
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)

    ax.legend(loc='upper left', fontsize=11 )
    ax.set_xlabel("Time b/w observations (days), " + r'$\Delta t$', fontsize=14)
    ax.set_ylabel("Variance of change in log-fold\nabundance, " + r'$\mathrm{Var}(\Delta \mathrm{ln} \, x_{i} | \Delta t)$', fontsize=14)


    fig.subplots_adjust(hspace=0.25, wspace=0.15)
    fig_name = "%sfig2_hurst.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()
        


def plot_g_dist(min_n_g=200):

    sys.stderr.write("Plottting growth rate distribution.....\n")

    days = mle_dict[target_dataset][target_host][list(mle_dict[target_dataset][target_host].keys())[0]]['days']
    delta_t_range = numpy.arange(1, 101)

    fig, ax = plt.subplots(figsize=(size_x, size_y))

    #c_g = ["#63a0c8", "#1f78b4", "#08306b"]

    #for delta_t_idx, delta_t in enumerate([1, 5, 10]):

    g_all = []

    for asv in  mle_dict[target_dataset][target_host].keys():

        rel_abundance = numpy.asarray(mle_dict[target_dataset][target_host][asv]['rel_abundance'])

        t_mid, x, g = data_utils.discretized_growth_rate(days, rel_abundance, 1, divide_by_delta_t=True)
        g_all.append(g)


    g_all = numpy.concatenate(g_all)
    g_unique = numpy.sort(numpy.unique(g_all))
    n_g_all = len(g_all)

    #if n_g_all <= min_n_g:
    #    continue

    #print(delta_t, numpy.var(g_all)/numpy.abs(numpy.mean(g_all)))

    #print(g_all)

    #survival = numpy.array([(g_all >= v).sum() / n_g_all for v in g_unique])

    hist, bin_edges = numpy.histogram(g_all, bins=15, density=False)
    hist = hist / hist.sum()
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    #ax.bar(bin_midpoints, hist, width=bin_edges[1]-bin_edges[0], c='k', alpha=1)
    ax.step(bin_midpoints, hist, where='mid', lw=lw, c=c_blue)#, label=r'$\Delta t = $' + str(delta_t))
    
    
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



def plot_autocorr():

    sys.stderr.write("Plottting temporal autocorrelation.....\n")

    days = mle_dict[target_dataset][target_host][list(mle_dict[target_dataset][target_host].keys())[0]]['days']

    fig, ax = plt.subplots(figsize=(size_x, size_y))

    #c_g = ["#63a0c8", "#1f78b4", "#08306b"]


    for asv in  mle_dict[target_dataset][target_host].keys():

        if asv != 'TACGGAGGATCCGAGCGTTATCCGGATTTATTGGGTTTAAAGGGAGCGTAGGTGGATTGTTAAGTCAGTTGTGAAAGTTTGCGGCTCAACCGTAAAATTGCAGTTGAAACTGGCAGTCTT':
            continue

        rel_abundance = numpy.asarray(mle_dict[target_dataset][target_host][asv]['rel_abundance'])

        delay_days_all, autocorr_all = stats_utils.autocorrelation_by_days(rel_abundance, days, min_n_autocorr_values=min_n_autocorr_values)

        autocorr_pos_idx = (autocorr_all >= -1) & (delay_days_all < 100)

        delay_days_all = delay_days_all[autocorr_pos_idx]
        autocorr_all = autocorr_all[autocorr_pos_idx]

        #autocorr_all = numpy.log(autocorr_all)

        if len(autocorr_all) < 4:
            continue

        if autocorr_all[-1] >= 0.8:
            continue

        ax.plot(delay_days_all, autocorr_all, lw=lw, c=c_blue, ls='-', alpha=1)


    ax.set_xlim([0, 5])
    ax.set_ylim([-0.2, 1])

    ax.set_xlabel("Time b/w observations (days), " + r'$\Delta t$', fontsize=14)
    ax.set_ylabel("Temporal autocorrelation", fontsize=14)

    ax.axhline(y=0, lw=lw, ls=':', c='k', label='No linear correlation', zorder=2)
    ax.legend(loc='upper right', fontsize=11)

    ax.xaxis.set_tick_params(labelsize=tick_labelsize)
    ax.yaxis.set_tick_params(labelsize=tick_labelsize)


    fig.subplots_adjust(hspace=0.25, wspace=0.15)
    fig_name = "%sfig2_autocorr.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def plot_psd():


    sys.stderr.write("Plottting power spectral density.....\n")

    f = numpy.linspace(0.01, 20, 1000)
    omega = 2 * numpy.pi * f

    days = numpy.asarray(mle_dict[target_dataset][target_host][list(mle_dict[target_dataset][target_host].keys())[0]]['days'])
    days = days - min(days)

    fig, ax = plt.subplots(figsize=(size_x, size_y))

    for asv in  mle_dict[target_dataset][target_host].keys():

        # TACGGAGGATCCGAGCGTTATCCGGATTTATTGGGTTTAAAGGGAGCGTAGATGGATGTTTAAGTCAGTTGTGAAAGTTTGCGGCTCAACCGTAAAATTGCAGTTGATACTGGATGTCTT

        if asv != 'TACGGAGGATCCGAGCGTTATCCGGATTTATTGGGTTTAAAGGGAGCGTAGGTGGATTGTTAAGTCAGTTGTGAAAGTTTGCGGCTCAACCGTAAAATTGCAGTTGAAACTGGCAGTCTT':
            continue

        #print(mle_dict[target_dataset][target_host][asv]['x_mean'])

        rel_abundance = numpy.asarray(mle_dict[target_dataset][target_host][asv]['rel_abundance'])
        x_mean = mle_dict[target_dataset][target_host][asv]['x_mean']
        #psd = signal.lombscargle(days, rel_abundance - x_mean, omega)
        #psd = numpy.sqrt(4*(psd/len(days)))

        # 128
        f, psd = signal.welch(rel_abundance - x_mean, fs=1, nperseg=80, scaling='density')
        to_keep_idx = f>0
        f = f[to_keep_idx]
        psd = psd[to_keep_idx]

        ax.plot(f, psd, lw=lw, c=c_blue, zorder=1)

        # slope
        slope, intercept, r_value, p_value, std_err = stats.linregress(numpy.log10(f), numpy.log10(psd))
        sys.stderr.write("Slope = %.6f \n" % slope)

        x_range =  numpy.linspace(min(numpy.log10(f)), max(numpy.log10(f)), 10000)
        y_pred_env = slope*x_range + intercept
        #ax.plot(10**x_range, 10**y_pred_env, ls=':', lw=lw, c='k', zorder=2, label='Exponent = ' + str(round(slope, 3)))

        ax.axhline(y=10**intercept, ls=':', lw=lw, c='k', zorder=2, label='White')
        ax.plot(10**x_range, 10**(-1*x_range + intercept), ls='--', lw=lw, c='k', zorder=2, label='Pink')

        ax.plot(10**x_range, 10**(-2*x_range + intercept), ls='-', lw=lw, c='k', zorder=2, label='Brownian')

    
    # peak at day = 1 means daily oscillation
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)

    ax.set_xlabel("Frequency", fontsize=14)
    ax.set_ylabel("Power Spectral Density (PSD)", fontsize=14)

    ax.xaxis.set_tick_params(labelsize=tick_labelsize)
    ax.yaxis.set_tick_params(labelsize=tick_labelsize)

    ax.legend(loc='upper right', fontsize=11, title='Noise color', title_fontsize=11)


    fig.subplots_adjust(hspace=0.25, wspace=0.15)
    fig_name = "%sfig2_psd.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


def plot_crosscorr_old():

    sys.stderr.write("Plottting cross-correlation.....\n")

    days = mle_dict[target_dataset][target_host][list(mle_dict[target_dataset][target_host].keys())[0]]['days']
    asv_all = list(mle_dict[target_dataset][target_host].keys())
    asv_pair_all = list(combinations(asv_all, 2))

    fig, ax = plt.subplots(figsize=(size_x, size_y))

    for asv_pair in asv_pair_all:

        rel_abundance_i = numpy.asarray(mle_dict[target_dataset][target_host][asv_pair[0]]['rel_abundance'])
        rel_abundance_j = numpy.asarray(mle_dict[target_dataset][target_host][asv_pair[1]]['rel_abundance'])

        lag_list, corr_list = stats_utils.crosscorrelation_by_days(rel_abundance_i, rel_abundance_j, days, min_n_corr_values=25)

        #if (sum(numpy.abs(corr_list) == 1.0) / len(corr_list)) > 0.2:
        if sum(numpy.abs(corr_list) == 1.0) > 1:
            continue

        #print(lag_list)

        ax.plot(lag_list, corr_list, lw=lw, c=c_blue, ls='-', alpha=0.7)


    ax.set_xlim([-5, 5])
    ax.set_ylim([-1, 1])

    ax.set_xlabel("Time b/w observations (days), " + r'$\Delta t$', fontsize=14)
    ax.set_ylabel("Temporal cross-correlation", fontsize=14)

    ax.axhline(y=0, lw=lw, ls=':', c='k', zorder=2)
    

    fig.subplots_adjust(hspace=0.25, wspace=0.15)
    fig_name = "%sfig2_crosscorr.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




def plot_crosscorr():

    sys.stderr.write("Plottting cross-correlation.....\n")

    days = mle_dict[target_dataset][target_host][list(mle_dict[target_dataset][target_host].keys())[0]]['days']
    asv_all = list(mle_dict[target_dataset][target_host].keys())
    asv_pair_all = list(combinations(asv_all, 2))

    fig, ax = plt.subplots(figsize=(size_x, size_y))

    asv_i = 'TACGTAGGGGGCAAGCGTTATCCGGATTTACTGGGTGTAAAGGGAGCGTAGACGGTGTGGCAAGTCTGATGTGAAAGGCATGGGCTCAACCTGTGGACTGCATTGGAAACTGTCATACTT'
    asv_j = 'TACGTAGGGGGCAAGCGTTATCCGGATTTACTGGGTGTAAAGGGAGCGTAGACGGACTGGCAAGTCTGATGTGAAAGGCGGGGGCTCAACCCCTGGACTGCATTGGAAACTGTTAGTCTT'

    rel_abundance_i = numpy.asarray(mle_dict[target_dataset][target_host][asv_i]['rel_abundance'])
    rel_abundance_j = numpy.asarray(mle_dict[target_dataset][target_host][asv_j]['rel_abundance'])

    lag_list, corr_list = stats_utils.crosscorrelation_by_days(rel_abundance_i, rel_abundance_j, days, min_n_corr_values=25)

    ax.plot(lag_list, corr_list, lw=lw, c=c_blue, ls='-', alpha=1, zorder=2)

    ax.set_xlim([-5, 5])
    ax.set_ylim([-1, 1])

    ax.set_xlabel("Time b/w observations (days), " + r'$\Delta t$', fontsize=14)
    ax.set_ylabel("Temporal cross-correlation", fontsize=14)

    ax.xaxis.set_tick_params(labelsize=tick_labelsize)
    ax.yaxis.set_tick_params(labelsize=tick_labelsize)

    ax.axhline(y=0, lw=lw, ls=':', c='k', zorder=1)
    

    fig.subplots_adjust(hspace=0.25, wspace=0.15)
    fig_name = "%sfig2_crosscorr.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




def plot_mean_vs_logfold():

    target_asv_ = 'TACGTAGGGGGCAAGCGTTATCCGGATTTACTGGGTGTAAAGGGAGCGTAGACGGTGTGGCAAGTCTGATGTGAAAGGCATGGGCTCAACCTGTGGACTGCATTGGAAACTGTCATACTT'

    days = numpy.asarray(mle_dict[target_dataset][target_host][list(mle_dict[target_dataset][target_host].keys())[0]]['days'])
    days = days - min(days)

    fig, ax = plt.subplots(figsize=(size_x, size_y))

    #for asv in  mle_dict[target_dataset][target_host].keys():
    #    print(mle_dict[target_dataset][target_host][asv]['x_mean'], asv)

    rel_abundance = numpy.asarray(mle_dict[target_dataset][target_host][target_asv_]['rel_abundance'])
    t_mid, x, g = data_utils.discretized_growth_rate(days, rel_abundance, 1, divide_by_delta_t=False)   
 
    # scatter_size=40
    ax.scatter(x, g, s=35, c=c_blue, zorder=1)

    slope, intercept, r_value, p_value, std_err = stats.linregress(numpy.log10(x), g)
    x_range =  numpy.linspace(min(numpy.log10(x)), max(numpy.log10(x)), 10000)
    y_pred = slope*x_range + intercept

    ax.plot(10**x_range, y_pred, ls='--', lw=lw, c='k', zorder=2)

    ax.xaxis.set_tick_params(labelsize=tick_labelsize)
    ax.yaxis.set_tick_params(labelsize=tick_labelsize)
    ax.set_xscale('log', base=10)

    ax.set_xlabel("Relative abundance, " + r'$x_{i}(t)$', fontsize=14)
    ax.set_ylabel("Log-fold change in abundance, " + r'$\Delta \mathrm{ln} \, x_{i}$', fontsize=14)


    fig.subplots_adjust(hspace=0.25, wspace=0.15)
    fig_name = "%sfig2_mean_vs_logfold.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def plot_mean_vs_logfold_slopes():

    
    #asv_all = list(mle_dict[target_dataset][target_host].keys())

    slope_all = []

    for host in list(mle_dict[target_dataset].keys()):

        days = numpy.asarray(mle_dict[target_dataset][host][list(mle_dict[target_dataset][host].keys())[0]]['days'])
        days = days - min(days)
        
        for asv in list(mle_dict[target_dataset][host].keys()):
            
            rel_abundance = numpy.asarray(mle_dict[target_dataset][host][asv]['rel_abundance'])
            t_mid, x, g = data_utils.discretized_growth_rate(days, rel_abundance, 1, divide_by_delta_t=False) 

            slope, intercept, r_value, p_value, std_err = stats.linregress(numpy.log10(x), g)
            slope_all.append(slope)


    fig, ax = plt.subplots(figsize=(size_x, size_y))

    hist, bin_edges = numpy.histogram(slope_all, bins=7, density=True)
    hist = hist / hist.sum()
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    #ax.axvline(x=0, lw=lw, ls=':', c='k', zorder=3)
    ax.step(bin_midpoints, hist, where='mid', lw=lw, c=c_blue, zorder=2, label='Observed')#, label=r'$\Delta t = $' + str(delta_t))

    ax.set_xlabel("Pairwise correlation", fontsize=14)
    ax.set_ylabel("Scaled probability density", fontsize=14)
    ax.xaxis.set_tick_params(labelsize=tick_labelsize)
    ax.yaxis.set_tick_params(labelsize=tick_labelsize)

    fig.subplots_adjust(hspace=0.25, wspace=0.15)
    fig_name = "%sfig2_mean_vs_logfold_slopes.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def plot_diss():


    # TACGGAGGATCCGAGCGTTATCCGGATTTATTGGGTTTAAAGGGAGCGTAGGCGGACGCTTAAGTCAGTTGTGAAAGTTTGCGGCTCAACCGTAAAATTGCAGTTGATACTGGGTGTCTT
    # TACGTATGGTGCAAGCGTTATCCGGATTTACTGGGTGTAAAGGGAGCGCAGGCGGTGCGGCAAGTCTGATGTGAAAGCCCGGGGCTCAACCCCGGTACTGCATTGGAAACTGTCGTACTA
    # TACGTAGGGGGCAAGCGTTATCCGGATTTACTGGGTGTAAAGGGAGCGTAGACGGTGTGGCAAGTCTGATGTGAAAGGCATGGGCTCAACCTGTGGACTGCATTGGAAACTGTCATACTT
    # TACGTAGGGGGCAAGCGTTATCCGGAATTACTGGGTGTAAAGGGTGCGTAGGTGGTATGGCAAGTCAGAAGTGAAAACCCAGGGCTTAACTCTGGGACTGCTTTTGAAACTGTCAGACTG
    # TACGTAGGTGGCGAGCGTTATCCGGATTTACTGGGTGTAAAGGGCGCGTAGGCGGGAATGCAAGTCAGATGTGAAATCCAAGGGCTCAACCCTTGAACTGCATTTGAAACTGTATTTCTT
    target_asv_1 = 'TACGTATGGTGCAAGCGTTATCCGGATTTACTGGGTGTAAAGGGAGCGCAGGCGGTGCGGCAAGTCTGATGTGAAAGCCCGGGGCTCAACCCCGGTACTGCATTGGAAACTGTCGTACTA'
    
    target_asv_2 = 'AACGTAGGTCACAAGCGTTGTCCGGAATTACTGGGTGTAAAGGGAGCGCAGGCGGGAAGACAAGTTGGAAGTGAAATCTATGGGCTCAACCCATAAACTGCTTTCAAAACTGTTTTTCTT'
    
    days = numpy.asarray(mle_dict[target_dataset][target_host][list(mle_dict[target_dataset][target_host].keys())[0]]['days'])

    delta_t_plot_1, diss_plot_1, diss_inf_1 = data_utils.temporal_dissimilarity_all_delta(days, mle_dict[target_dataset][target_host][target_asv_1]['abundance'], min_n=10)
    delta_t_plot_2, diss_plot_2, diss_inf_2 = data_utils.temporal_dissimilarity_all_delta(days, mle_dict[target_dataset][target_host][target_asv_2]['abundance'], min_n=10)


    fig, ax = plt.subplots(figsize=(size_x, size_y))
    ax.plot(delta_t_plot_1, diss_plot_1/diss_inf_1, ls='-', lw=2, alpha=1, c=c_blue, zorder=2)
    ax.plot(delta_t_plot_2, diss_plot_2/diss_inf_2, ls='-', lw=2, alpha=1, c=c_orange, zorder=2)

    ax.set_xlim([0, max(delta_t_plot_2)])
    ax.set_ylim([0, 2])

    ax.xaxis.set_tick_params(labelsize=tick_labelsize)
    ax.yaxis.set_tick_params(labelsize=tick_labelsize)

    ax.set_xlabel("Time b/w observations (days), " + r'$\Delta t$', fontsize=14)
    ax.set_ylabel("Temporal dissimilarity, " + r'$\Phi_{i}(\Delta t)$', fontsize=14)

    fig.subplots_adjust(hspace=0.25, wspace=0.15)
    fig_name = "%sfig2_diss.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def plot_corr_dist():

    sys.stderr.write("Plottting correlation distribution.....\n")

    #days = mle_dict[target_dataset][target_host][list(mle_dict[target_dataset][target_host].keys())[0]]['days']
    asv_all = list(mle_dict[target_dataset][target_host].keys())
    asv_pair_all = list(combinations(asv_all, 2))

    fig, ax = plt.subplots(figsize=(size_x, size_y))

    rho_all = []
    rho_null_all = []
    for asv_pair in asv_pair_all:

        rel_abundance_i = numpy.asarray(mle_dict[target_dataset][target_host][asv_pair[0]]['rel_abundance'])
        rel_abundance_j = numpy.asarray(mle_dict[target_dataset][target_host][asv_pair[1]]['rel_abundance'])
        rho_all.append(numpy.corrcoef(rel_abundance_i, rel_abundance_j)[0,1])

        for n in range(100):
            rho_null_all.append(numpy.corrcoef(numpy.random.permutation(rel_abundance_i), numpy.random.permutation(rel_abundance_j))[0,1])

    rho_all = numpy.asarray(rho_all)
    rho_null_all = numpy.asarray(rho_null_all)


    #bins = numpy.linspace(-1, 1, 30)
    hist, bin_edges = numpy.histogram(rho_all, bins=10, density=True)
    hist = hist / hist.sum()
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    hist_null, bin_edges_null = numpy.histogram(rho_null_all, bins=12, density=True)
    hist_null = hist_null / hist_null.sum()
    bin_midpoints_null = (bin_edges_null[:-1] + bin_edges_null[1:]) / 2


    scale = hist.max() / hist_null.max()
    hist_null = hist_null*scale

    ax.axvline(x=0, lw=lw, ls=':', c='k', zorder=3)
    ax.step(bin_midpoints, hist, where='mid', lw=lw, c=c_blue, zorder=2, label='Observed')#, label=r'$\Delta t = $' + str(delta_t))
    ax.step(bin_midpoints_null, hist_null, where='mid', lw=lw, ls='-', c='k', zorder=1, label='Null')#, label=r'$\Delta t = $' + str(delta_t))

    ax.set_xlabel("Pairwise correlation", fontsize=14)
    ax.set_ylabel("Scaled probability density", fontsize=14)
    ax.xaxis.set_tick_params(labelsize=tick_labelsize)
    ax.yaxis.set_tick_params(labelsize=tick_labelsize)
    
    ax.set_xlim([-1* max(numpy.absolute(bin_midpoints)), max(numpy.absolute(bin_midpoints))])
    ax.set_ylim([min(hist), max(hist_null)*1.1])
    #ax.set_yscale('log', base=10)

    ax.legend(loc='upper left', fontsize=10)

    
    fig.subplots_adjust(hspace=0.25, wspace=0.15)
    fig_name = "%sfig2_corr_dist.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



if __name__ == "__main__":
 
    sys.stderr.write("Plottting temporal patterns.....\n")

    #plot_sojourn_time()

    #plot_mean_vs_logfold()

    #plot_diss()

    #plot_corr_dist()

    #plot_autocorr()


    #plot_hust()

    #plot_g_dist()

    #plot_hust()
    #plot_mean_vs_delta()
    #plot_mean_vs_logfold()

    plot_mean_vs_logfold_slopes()

    #plot_mean_vs_delta()
    #plot_crosscorr()

    #plot_psd()

    #plot_autocorr()

    # to-do
    #temporal dissimilarity
    # growth vs. log x
    
    # pairwise plots

    #S = tsdata_to_cpsd.cpsd_welch_matlab(rna_dna, n=n, h=h, nfft=nfft, window=window, noverlap=noverlap, fs=1.0)
    #S_xy = S[0,1,:]

    # Phase spectrum (radians)
    #phase_xy = numpy.angle(S_xy)
    # Avoid division by zero at DC
    #freqs = numpy.linspace(0, fs/2, h)
    #freqs_nonzero = freqs.copy()
    # ignore 0 Hz for lag calculation
    #freqs_nonzero[0] = numpy.nan  
    # Time lag at each frequency (same units as 1/fs, e.g., weeks)
    #time_lag = phase_xy / (2 * numpy.pi * freqs_nonzero)

    # magnitude-squared coherence
    #coh_xy = numpy.abs(S_xy)**2 / (S[0,0,:] * S[1,1,:])
    #mask = coh_xy > min_coh_xy
    #avg_lag = numpy.nanmean(time_lag[mask])