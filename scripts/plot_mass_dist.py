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


file_dict = {'glu': 'Fig3ac_SF4a_mopsEZglu_withmetadata', 'suc': 'Fig3bd_SF4b_mopsEZsuc_withmetadata'}

c_sources = ['glu', 'suc']
target_time = {'glu':'3.56666666666667', 'suc': '3.56666666666667'}
rep_dict = {'glu': 4, 'suc': 4}
c_color = {'glu': '#F2B342', 'suc': '#EA573D'}
#c = 'glu'
c_label = {'glu': 'Glucose', 'suc': 'Succinate'}



fig, ax = plt.subplots(figsize=(size_x,size_y))


for c in c_sources:

    df = pandas.read_csv('%s%s.csv' % (config.data_directory, file_dict[c]))
    df[['CumulativeTime_hrs']] = df[['CumulativeTime_hrs']].astype(str)
    hours = numpy.unique(df.CumulativeTime_hrs.values)

    # subset
    df_subset = df[(df["CumulativeTime_hrs"] == target_time[c]) & (df["Rep"] == rep_dict[c])]

    mass = df_subset.Mass.values
    mass = mass[mass > 0]

    bins = numpy.logspace(numpy.log10(mass.min()), numpy.log10(mass.max()), 40)
    hist, bin_edges = numpy.histogram(mass, bins=bins, density=True)
    hist = hist / hist.sum()
    bin_midpoints = numpy.sqrt(bin_edges[:-1] * bin_edges[1:])

    #hist, bin_edges = numpy.histogram(mass, bins=40, density=True)
    #hist = hist / hist.sum()
    #bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    #scale = hist.max() / hist_null.max()
    #hist_null = hist_null*scale

    ax.step(bin_midpoints, hist, where='mid', lw=lw, c=c_color[c], zorder=2, label=c_label[c])
    #ax.step(bin_midpoints_null, hist_null, where='mid', lw=lw, ls='-', c='k', zorder=1, label='Null')#, label=r'$\Delta t = $' + str(delta_t))


ax.set_xlim([100, 1000])
ax.set_ylim([0, 0.175])

ax.set_xscale("log", base=10)
#ax.set_yscale("log", base=10)


#ax.xaxis.set_major_locator(LogLocator(base=10))
#ax.xaxis.set_minor_locator(LogLocator(base=10, subs='auto'))
#ax.xaxis.set_major_formatter(LogFormatter())


ax.set_xlabel("Cell mass [fg]", fontsize=14)
ax.set_ylabel("Probability density", fontsize=14)

#ax.xaxis.set_major_formatter(LogFormatterMathtext())
#ax.tick_params(axis='x', which='both', labelsize=tick_labelsize)


ax.set_xticks([1e2, 1e3])
ax.xaxis.set_major_formatter(LogFormatterMathtext())
ax.minorticks_off()
ax.tick_params(axis='x', labelsize=tick_labelsize)
#ax.xaxis.set_tick_params(labelsize=tick_labelsize)
ax.yaxis.set_tick_params(labelsize=tick_labelsize)


#ax.legend(loc='upper left')

fig.subplots_adjust(hspace=0.25, wspace=0.15)
fig_name = "%smass_dist.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()