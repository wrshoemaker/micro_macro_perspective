import os
import numpy
import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker
from scipy.stats import loggamma, mode, nbinom
import scipy.special as special
import pandas

import stats_utils
import config
import data_utils
import plot_utils

from scipy import stats, signal, special



df = pandas.read_csv("%sVirMic_data.csv" % config.data_directory, sep=",")



size_x, size_y = plot_utils.size_x, plot_utils.size_y
scatter_size=plot_utils.scatter_size
c_blue = plot_utils.c_blue
tick_labelsize = plot_utils.tick_labelsize
lw = plot_utils.lw


phage = df.VIRUS.values
cells = df.BACTERIA.values 

fig, ax = plt.subplots(figsize=(size_x,size_y))

# make scatterplot density map..
ax.scatter(cells, phage, s=10, c=c_blue, zorder=1, alpha=0.4)

ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)

ax.xaxis.set_tick_params(labelsize=tick_labelsize)
ax.yaxis.set_tick_params(labelsize=tick_labelsize)


fig.subplots_adjust(hspace=0.25, wspace=0.15)
fig_name = "%scells_vs_phage.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()