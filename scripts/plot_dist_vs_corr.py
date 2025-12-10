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

size_x, size_y = 4,4
tick_labelsize=8
c_blue='#1E90FF'

min_occupancy = 0.8
data_dict = pickle.load(open('%scaporaso_et_al_dict.pickle' % config.data_directory, "rb"))
phylo_dict = pickle.load(open('%sphylo_dist_dict.pickle' % config.data_directory, "rb"))


asv_all = list(data_dict['asv'].keys())
total_reads = numpy.asarray(data_dict['total_reads'])

asv_final = []

for i in range(len(asv_all)):

    asv_i = asv_all[i]
    afd_i = numpy.asarray(data_dict['asv'][asv_i])
    occupancy_i = sum(afd_i>0)/len(afd_i)

    if occupancy_i >= min_occupancy:
        asv_final.append(asv_i)


asv_pair_all = list(combinations(asv_final, 2))

dist_final = []
rho_final = []
for asv_pair in asv_pair_all:

    asv_pair = tuple(sorted(asv_pair))

    asv_i = asv_pair[0]
    asv_j = asv_pair[1]

    afd_i = numpy.asarray(data_dict['asv'][asv_i])/total_reads
    afd_j = numpy.asarray(data_dict['asv'][asv_j])/total_reads

    dist_final.append(phylo_dict['caporaso_et_al'][asv_pair])
    rho_final.append(numpy.corrcoef(afd_i, afd_j)[0,1])


dist_final = numpy.asarray(dist_final)
rho_final = numpy.asarray(rho_final)


fig, ax = plt.subplots(figsize=(size_x, size_y))

x_bins = numpy.logspace(numpy.log10(dist_final.min()), numpy.log10(dist_final.max()), 15)
mean_y, bin_edges, _ = stats.binned_statistic(dist_final, rho_final, statistic='mean', bins=x_bins)
counts, _, _ = stats.binned_statistic(dist_final, rho_final, statistic='count', bins=x_bins)

to_plot = counts >= 10

bin_centers = numpy.sqrt(bin_edges[:-1] * bin_edges[1:])

print(len(asv_pair_all))

ax.plot(bin_centers[to_plot], mean_y[to_plot],  ls='-', lw=4, alpha=1, c=c_blue, zorder=2)
ax.set_xscale('log', base=10)
ax.axhline(y=0, lw=4, ls=':', c='k', zorder=3)

ax.set_xlabel("Pairwise phylogenetic distance", fontsize=14)
ax.set_ylabel("Mean pairwise correlation", fontsize=14)
ax.xaxis.set_tick_params(labelsize=tick_labelsize)
ax.yaxis.set_tick_params(labelsize=tick_labelsize)


fig.subplots_adjust(hspace=0.25, wspace=0.15)
fig_name = "%sfig2_dist_vs_corr.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()