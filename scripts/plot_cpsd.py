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

import tsdata_to_cpsd


import config
import data_utils
import plot_utils
import stats_utils

target_dataset = 'caporaso_et_al'
target_host = 'M3'
min_n_autocorr_values = 20
#target_asv = 'TACGGAGGATCCGAGCGTTATCCGGATTTATTGGGTTTAAAGGGAGCGTAGGTGGATTGTTAAGTCAGTTGTGAAAGTTTGCGGCTCAACCGTAAAATTGCAGTTGAAACTGGCAGTCTT'

c_blue='#1E90FF'
c_orange='#EB5900'
tick_labelsize=8

size_x, size_y = 4,4
lw=3
scatter_size=40
#target_







res_ret_dict = pickle.load(open('%sres_ret_dict.pickle' % config.data_directory, "rb"))
mle_dict = pickle.load(open('%smle_dict.pickle' % config.data_directory, "rb"))
dataset_all = ['david_et_al', 'poyet_et_al', 'caporaso_et_al']

asv_all = list(mle_dict[target_dataset][target_host].keys())
asv_pair_all = list(combinations(asv_all, 2))


n = 2
# frequency resolution
fres = 128
# MATLAB’s h = fres + 1              
h = fres + 1 
fs=1.0
#nfft = 2*(h-1)
nfft = 256
# 251 observations
window = int(len(mle_dict[target_dataset][target_host][asv_all[0]]['rel_abundance']) / 2)
noverlap = 30
min_coh_xy = 0.3
n_surr=1000

df = 1.0 / nfft



pair_info = {}

for asv_pair in asv_pair_all:

    afd_i = numpy.asarray(mle_dict[target_dataset][target_host][asv_pair[0]]['rel_abundance'])
    afd_j = numpy.asarray(mle_dict[target_dataset][target_host][asv_pair[1]]['rel_abundance'])

    afd_pair = numpy.column_stack((afd_i, afd_j))

    S = tsdata_to_cpsd.cpsd_welch_matlab(afd_pair, n=n, h=h, nfft=nfft, window=window, noverlap=noverlap, fs=1.0)

    S_xx = S[0,0,:]
    S_yy = S[1,1,:]
    S_xy = S[0,1,:]

    coherence = numpy.abs(S_xy)**2 / (S_xx * S_yy)
    coherence = numpy.clip(coherence.real, 0.0, 1.0)


    I_ij = -0.5 * numpy.sum(numpy.log(1.0 - coherence)) * df

    print(I_ij)

    pair_info[asv_pair] = I_ij

    #freq = numpy.fft.rfftfreq(nfft, d=1/fs)

    #print(coherence)

    #ax.plot(freq, coherence, lw=1, c=c_blue, ls='-', alpha=0.4)


best_pair = max(pair_info, key=pair_info.get)
best_value = pair_info[best_pair]
#print(best_value)
print(best_pair)


afd_i = numpy.asarray(mle_dict[target_dataset][target_host][best_pair[0]]['rel_abundance'])
afd_j = numpy.asarray(mle_dict[target_dataset][target_host][best_pair[1]]['rel_abundance'])
afd_pair = numpy.column_stack((afd_i, afd_j))

S = tsdata_to_cpsd.cpsd_welch_matlab(afd_pair, n=n, h=h, nfft=nfft, window=window, noverlap=noverlap, fs=1.0)

S_xx = S[0,0,:]
S_yy = S[1,1,:]
S_xy = S[0,1,:]

coherence = numpy.abs(S_xy)**2 / (S_xx * S_yy)
coherence = numpy.clip(coherence.real, 0.0, 1.0)

freq = numpy.fft.rfftfreq(nfft, d=1.0)  # since fs=1.0

fig, ax = plt.subplots(figsize=(size_x, size_y))

ax.plot(freq, coherence, lw=2, c=c_blue, ls='-', alpha=1)

ax.set_xlim([0, max(freq)])
ax.set_ylim([0, 1])

ax.set_xlabel("Frequency", fontsize=14)
ax.set_ylabel("Cross-Power Spectral density", fontsize=14)

ax.xaxis.set_tick_params(labelsize=tick_labelsize)
ax.yaxis.set_tick_params(labelsize=tick_labelsize)


fig.subplots_adjust(hspace=0.25, wspace=0.15)
fig_name = "%sfig2_cpsd.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()