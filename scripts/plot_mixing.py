import os
import numpy
import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker
from scipy.stats import loggamma, mode, nbinom
import scipy.special as special
import pandas

from matplotlib.ticker import LogLocator, LogFormatter
from matplotlib.ticker import LogFormatterMathtext


import stats_utils
import config
import data_utils
import plot_utils

from scipy import stats, signal, special




size_x, size_y = plot_utils.size_x, plot_utils.size_y
scatter_size=plot_utils.scatter_size
c_blue = plot_utils.c_blue
tick_labelsize = plot_utils.tick_labelsize
lw = plot_utils.lw


df = pandas.read_csv('%sFig4cd_data.txt' % config.data_directory, sep='\t')
df_subset = df[(df["Group"] == 'prot')]
df_subset = df_subset.dropna()

evenness = 1 - df_subset['Gini Factor'].values
intermixing = df_subset.Intermixing.values


fig, ax = plt.subplots(figsize=(size_x, size_y))

ax.scatter(intermixing, evenness, s=scatter_size, c=c_blue, zorder=1, alpha=1)
slope, intercept, r_value, p_value, std_err = stats.linregress(intermixing, evenness)

x_range =  numpy.linspace(min(intermixing), max(intermixing), 10000)
ax.plot(x_range, slope*x_range + intercept, ls='--', lw=lw, c='k', zorder=2)


ax.set_xlabel("Spatial mixing [" + r'$\mu \mathrm{m}^{-1}$' + "]", fontsize=14)
ax.set_ylabel("Evenness of birth events", fontsize=14)
ax.set_ylim([0,0.8])
ax.xaxis.set_tick_params(labelsize=tick_labelsize)
ax.yaxis.set_tick_params(labelsize=tick_labelsize)




fig.subplots_adjust(hspace=0.25, wspace=0.15)
fig_name = "%smixing.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()