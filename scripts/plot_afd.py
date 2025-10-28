
import os
import numpy
import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker
from scipy.stats import loggamma, mode
import scipy.special as special


import config
import data_utils
import plot_utils


fig, ax = plt.subplots(figsize=(4,4))

min_occupancy = 1
min_rescaled = -4
max_rescaled = 4

counts_all = []
flat_rescaled_afd_ln_all_environments = []
for environment in data_utils.datasets_crossectional:

    print(plot_utils.environment_name_dict[environment])
    #environment = 'activatedsludge'
    environment_counts_np = data_utils.get_read_counts(environment, longitudinal_bool=False)
    rel_environment_counts_np = environment_counts_np/numpy.sum(environment_counts_np, axis=0)

    # otus detected in all samples
    #rel_environment_counts_np = rel_environment_counts_np[numpy.all(rel_environment_counts_np, axis=1)]
    occupancy = numpy.sum(rel_environment_counts_np > 0, axis=1)/rel_environment_counts_np.shape[1]
    rel_environment_counts_np = rel_environment_counts_np[(occupancy>=min_occupancy),:]

    rescaled_afd_ln_all = []
    for afd in rel_environment_counts_np:
        # rescale individual AFDs
        afd_ln = numpy.log(afd[afd>0])
        rescaled_afd_ln = (afd_ln - numpy.mean(afd_ln))/numpy.std(afd_ln)
        #rescaled_afd_log10 = rescaled_afd_log10[ (rescaled_afd_log10 >= min_rescaled) & (rescaled_afd_log10 <= max_rescaled) ]
        rescaled_afd_ln_all.append(rescaled_afd_ln)

    if len(rescaled_afd_ln_all) == 0:
        continue

    flat_rescaled_afd_ln_all = numpy.concatenate(rescaled_afd_ln_all).ravel()
    #print(numpy.median(flat_rescaled_afd_log10_all))

    flat_rescaled_afd_ln_all_environments.append(flat_rescaled_afd_ln_all)

    # bin
    counts, bin_edges = data_utils.bin_x(flat_rescaled_afd_ln_all, n_bins=20, log10=False, normalize_counts=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    to_plot_idx = (counts>0)
    bin_centers = bin_centers[to_plot_idx]
    counts = counts[to_plot_idx]
    counts_all.append(counts)

    ax.scatter(bin_centers, counts, marker=plot_utils.environment_shape_dict[environment], facecolors=plot_utils.environment_facecolor_dict[environment], edgecolors=plot_utils.environment_cmap_dict[environment], label=plot_utils.environment_name_dict[environment])



flat_rescaled_afd_ln_all_environments = numpy.concatenate(flat_rescaled_afd_ln_all_environments).ravel()
to_fit_flat_rescaled_afd_ln_all_environments = flat_rescaled_afd_ln_all_environments[(flat_rescaled_afd_ln_all_environments > -3) & (flat_rescaled_afd_ln_all_environments < 3)]
loggamma_fit = loggamma.fit(to_fit_flat_rescaled_afd_ln_all_environments)

#k = 1.7
#k_digamma = special.digamma(k)
#k_trigamma = special.polygamma(1,k)

x_range = numpy.linspace(min(flat_rescaled_afd_ln_all_environments), max(flat_rescaled_afd_ln_all_environments), 10000)
#loggamma_pdf = k*k_trigamma*x_range - numpy.exp(numpy.sqrt(k_trigamma)*x_range + k_digamma) - numpy.log(special.gamma(k)) + k*k_digamma + numpy.log10(numpy.exp(1))
#loggamma_pdf = loggamma.pdf(x_range, loggamma_fit[0], loggamma_fit[1], loggamma_fit[2])

flat_counts_all = numpy.concatenate(counts_all).ravel()

#print(min(flat_counts_all))

#gammalog = k*k_trigamma*x_range - np.exp(np.sqrt(k_trigamma)*x_range + k_digamma) - np.log(special.gamma(k)) + k*k_digamma + np.log10(np.exp(1))
ax.plot(x_range, loggamma(loggamma_fit[0], loggamma_fit[1], loggamma_fit[2]).pdf(x_range), 'k', label='Gamma', lw=2, ls='-', zorder=2)
#ax.plot(x_range, 10**pdf_gammalog, 'k', label='Gamma', lw=2, ls='-', zorder=2)


# fit the loggamma
ax.legend(loc='upper left', fontsize=9)
ax.set_yscale('log', base=10)


ax.xaxis.set_tick_params(labelsize=8)
ax.yaxis.set_tick_params(labelsize=8)

ax.set_ylim([min(flat_counts_all), 1])

ax.set_xlabel("Rescaled " + r'$\mathrm{ln}$' + " relative abundance", fontsize=14)
ax.set_ylabel("Probability density", fontsize=14)



fig.subplots_adjust(hspace=0.25, wspace=0.15)
fig_name = "%safd.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()