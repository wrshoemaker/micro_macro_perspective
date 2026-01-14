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


depth_cutoff = 100

size_x, size_y = plot_utils.size_x, plot_utils.size_y
scatter_size=plot_utils.scatter_size
c_blue = plot_utils.c_blue
tick_labelsize = plot_utils.tick_labelsize
lw = plot_utils.lw


phage = df.VIRUS.values
cells = df.BACTERIA.values 
depth = df.DEPTH.values

fig, ax = plt.subplots(figsize=(size_x,size_y))


# make scatterplot density map..
#ax.scatter(cells, phage, s=10, c=c_blue, zorder=1, alpha=0.4)

#ax.set_xscale('log', base=10)
#ax.set_yscale('log', base=10)

cutoff_cmap_all = ['Blues', 'Blues']

#for cutoff_idx, cutoff in enumerate([depth <= depth_cutoff, depth > depth_cutoff]):
for cutoff_idx, cutoff in enumerate([depth > depth_cutoff]):

    cells_c = cells[cutoff]
    phage_c = phage[cutoff]

    plot_utils.plot_color_by_pt_dens(cells_c, phage_c, 3, loglog=1, plot_obj=ax, cmap=cutoff_cmap_all[cutoff_idx], alpha=0.7, size=6)


    cells_c_log10 = numpy.log10(cells_c)
    phage_c_log10 = numpy.log10(phage_c)

    slope, intercept, r_value, p_value, std_err = stats.linregress(cells_c_log10, phage_c_log10)
    
    x_range =  numpy.linspace(min(cells_c_log10), max(cells_c_log10), 10000)
    ax.plot(10**x_range, 10**(slope*x_range + intercept), ls='--', lw=lw, c='k', zorder=2)

    

ax.set_ylim([100000, 20000000])

ax.xaxis.set_tick_params(labelsize=tick_labelsize)
ax.yaxis.set_tick_params(labelsize=tick_labelsize)

#cell_density_notation = '[(cells)' + r'$ \cdot  \mathrm{mL}^{-1}$'  +  ']' 

ax.set_xlabel("Cell density " + '[(cells)' + r'$ \cdot  \mathrm{mL}^{-1}$'  +  ']' , fontsize=14)
ax.set_ylabel("Phage density " + '[(particles)' + r'$ \cdot  \mathrm{mL}^{-1}$'  +  ']' , fontsize=14)


fig.subplots_adjust(hspace=0.25, wspace=0.15)
fig_name = "%scells_vs_phage.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()