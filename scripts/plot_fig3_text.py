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


fig, ax = plt.subplots(figsize=(size_x, size_y))


ax.text(0.03, 0.9, "Supplied C sources", fontsize=14)

ax.text(0.03, 0.8, "Excreted Carbon sources", fontsize=14)
ax.text(0.03, 0.7, "Species", fontsize=14)

#ax.text(0.03, 0.6, "Slow      Fast", fontsize=14)
ax.text(0.03, 0.5, "Growth rate", fontsize=14)

ax.text(0.03, 0.4, "Spatial distance", fontsize=14)


ax.text(0.03, 0.3, "Resource diffusion", fontsize=14, color='#9B5C97')#, fontweight='bold')


ax.text(0.03, 0.2, "Infection     Growth      Lysis ", fontsize=14, color='k')#, fontweight='bold')



#ax.text(0.03, 0.1, "Slow (succinate)", fontsize=14, color='#EA573D')#, fontweight='bold')
#ax.text(0.5, 0.1, "Fast (glucose)", fontsize=14, color='#F2B342')#, fontweight='bold')

ax.text(0.1, 0.1, "Observed, " + r'$y = 1.4 \cdot x$', fontsize=14, color='k')#, fontweight='bold')

#ax.text(0.1, 0.1, "Time, " + r'$t$', fontsize=14, color='k')#, fontweight='bold')
#ax.text(0.1, 0.1, "Relative abundance, " + r'$x_{i}(t)$', fontsize=14, color='k')#, fontweight='bold')

#ax.set_xlabel("Relative\nabundance, " + r'$x_{i}(t)$', fontsize=14)


fig.subplots_adjust(hspace=0.25, wspace=0.15)
fig_name = "%sfig3_text.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()