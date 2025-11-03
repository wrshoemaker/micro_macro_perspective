
import os
import numpy
import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker

import config
import data_utils
import plot_utils

fig, ax = plt.subplots(figsize=(4,4))

min_occupancy = 0.2

means_all = []
vars_all = []

flat_rescaled_afd_log10_all_environments = []
for environment in data_utils.datasets_crossectional:

    print(plot_utils.environment_name_dict[environment])
    environment_counts_np = data_utils.get_read_counts(environment, longitudinal_bool=False)
    rel_environment_counts_np = environment_counts_np/numpy.sum(environment_counts_np, axis=0)
    occupancy = numpy.sum(rel_environment_counts_np > 0, axis=1)/rel_environment_counts_np.shape[1]
    rel_environment_counts_np = rel_environment_counts_np[(occupancy>=min_occupancy),:]

    means = numpy.mean(rel_environment_counts_np, axis=1)
    vars = numpy.var(rel_environment_counts_np, axis=1)

    means_all.append(means)
    vars_all.append(vars)

    # bin on x and y
    bin_centers, x_means, y_means = data_utils.bin_mean_xy(means, vars, nbins=20, log10_x=True, log10_y=True)

    ax.scatter(10**x_means, 10**y_means, marker=plot_utils.environment_shape_dict[environment], facecolors=plot_utils.environment_facecolor_dict[environment], edgecolors=plot_utils.environment_cmap_dict[environment])



# plot Taylor's Law
means_all = numpy.concatenate(means_all).ravel()
vars_all = numpy.concatenate(vars_all).ravel()
log10_means_all = numpy.log10(means_all)
log10_vars_all = numpy.log10(means_all)

log10_x_range = numpy.linspace(min(log10_means_all), max(log10_means_all), 10000)
intercept = numpy.mean(numpy.log10(vars_all)) - 2*numpy.mean(numpy.log10(means_all))
log10_y_pred = intercept + 2*log10_x_range


ax.plot(10**log10_x_range, 10**log10_y_pred, 'k', lw=2, ls='-', zorder=2, label=r'$\sigma^{2}_{x_{i}} \propto \bar{x}_{i}^{2}$')
ax.legend(loc='upper left', fontsize=9)


ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)

ax.xaxis.set_tick_params(labelsize=8)
ax.yaxis.set_tick_params(labelsize=8)

#ax.set_ylim([min(flat_counts_all), 1])

ax.set_xlabel("Mean relative abundance, " + r'$\bar{x}_{i}$', fontsize=14)
ax.set_ylabel("Variance of relative abundance, " + r'$\sigma^{2}_{x_{i}}$', fontsize=14)

#ax.set_title("Taylor's Law", fontsize=16, fontweight='bold')



fig.subplots_adjust(hspace=0.25, wspace=0.15)
fig_name = "%staylors_law.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()