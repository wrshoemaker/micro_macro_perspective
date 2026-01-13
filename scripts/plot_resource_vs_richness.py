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



size_x, size_y = plot_utils.size_x, plot_utils.size_y
scatter_size=plot_utils.scatter_size
c_blue = plot_utils.c_blue
tick_labelsize = plot_utils.tick_labelsize
lw = plot_utils.lw

x_positions = [1, 2, 4, 8, 15, 16]


df = pandas.read_csv('%sDiversity_data.csv' % config.data_directory)
meanRich_Csource = (df.groupby("C_number").agg(mean_rich=("D0", "mean"), SE_rich=("D0", lambda x: x.std(ddof=1) / numpy.sqrt(len(x)))).reset_index())

meanRich_tib = (
    df
    .groupby(["medium", "FC_number", "C_number"])
    .agg(
        mean_rich=("D0", "mean"),
        SE_rich=("D0", lambda x: x.std(ddof=1) / numpy.sqrt(len(x)))
    )
    .reset_index()
)

# create numeric variable for medium (R factor-like behavior)
meanRich_tib["medium"] = meanRich_tib["medium"].astype("category")
meanRich_tib["medium_id"] = meanRich_tib["medium"].cat.codes + 1



resource_n = meanRich_tib['C_number'].values
diversity = meanRich_tib['mean_rich'].values

slope, intercept, r_value, p_value, std_err = stats.linregress(resource_n, diversity)

x_range =  numpy.linspace(min(resource_n), max(resource_n)+1, 10000)

fig, ax = plt.subplots(figsize=(size_x,size_y))
ax.scatter(resource_n, diversity, s=scatter_size, c=c_blue, zorder=1, alpha=0.8)

#ax.axhline(y=10**intercept, ls=':', lw=lw, c='k', zorder=2, label='White')
ax.plot(x_range, slope*x_range + intercept, ls='--', lw=lw, c='k', zorder=2, label='Regression')


ax.plot(x_range, x_range, ls=':', lw=lw, c='k', zorder=2, label='Competitive exclusion (prediction)')


ax.set_ylim([0, 50])


ax.set_xticks(x_positions, x_positions)
ax.xaxis.set_tick_params(labelsize=tick_labelsize)
ax.yaxis.set_tick_params(labelsize=tick_labelsize)


ax.set_xlabel("# supplied carbon sources", fontsize=14)
ax.set_ylabel("Community diversity (# ASVs)", fontsize=14)


fig.subplots_adjust(hspace=0.25, wspace=0.15)
fig_name = "%sresource_vs_richness.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()