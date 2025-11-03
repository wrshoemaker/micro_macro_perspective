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



id_val = plot_utils.id_val_example
environment = plot_utils.environment_example
proj, gamma_pars, mean_pars = data_utils.get_processed_data()



simulates_sp = pandas.DataFrame()
npoints = 5

sname = mean_pars.loc[mean_pars["idall"] == id_val, "sname"].iloc[0]

proj_id = proj[proj["idall"] == id_val]
nmin = proj_id["nreads"].astype(int).min()
nmax = proj_id["nreads"].astype(int).max()

dn = (numpy.log10(nmax) - numpy.log10(nmin)) / npoints

sobs = proj_id["otu_id"].nunique()

row = mean_pars.loc[mean_pars["idall"] == id_val].iloc[0]
beta = row["mbeta"]
mu = row["mu"]
sigma = row["sigma"]
stot = row["stot"]

for i in range(-2, npoints + 3):  # R’s -2:(npoints+2)
    n = nmin * 10 ** (i * dn)
    
    pred_sp = stats_utils.predicted_sp(n, mu, sigma, beta, stot)
    pred_nr = stats_utils.predicted_reads(n, mu, sigma, beta, stot)
    pred_shannon = stats_utils.predicted_shannonindex(n, mu, sigma, beta, stot)
    
    d = pandas.DataFrame({
        "idall": [id_val],
        "sname": [sname],
        "pred_sp": [pred_sp],
        "pred_si": [pred_shannon],
        "tot_sp": [stot],
        "nreads": [pred_nr],
        "sigma": [sigma],
        "mu": [mu],
        "sobs": [sobs],
        "beta": [beta],
        "n0": [n]
    })
    
    simulates_sp = pandas.concat([simulates_sp, d], ignore_index=True)


#psp = (proj.groupby(["project_id", "sample_id", "run_id", "sname", "scat", "nreads"]).agg(nsp=("otu_id", pandas.Series.nunique)).reset_index())

proj_environment = proj[proj["sname"] == plot_utils.sname_example]

for col in ["project_id", "sample_id", "run_id"]:
    proj_environment[col] = proj_environment[col].astype("category")


#pspm = (proj.groupby(["project_id", "sample_id", "run_id", "sname", "scat", "nreads"]).agg(nsp=("otu_id", "nunique")).reset_index())
pspm = (proj_environment.groupby(["project_id", "sample_id", "run_id", "sname", "scat", "nreads"]).agg(nsp=("otu_id", "nunique")).reset_index())


df_sub = pspm[pspm["sname"] == plot_utils.sname_example]
sim_sub = simulates_sp[simulates_sp["sname"] == plot_utils.sname_example]


fig, ax = plt.subplots(figsize=(4,4))

ax.scatter(df_sub["nreads"], df_sub["nsp"], color="gray", s=10, alpha=1, edgecolor="none", zorder=1)
ax.plot(sim_sub["n0"], sim_sub["pred_sp"], color="black", linewidth=2, label='Prediction', ls='-', zorder=2)


bin_centers, x_means, y_means = data_utils.bin_mean_xy(df_sub['nreads'].values, df_sub['nsp'].values, nbins=5, log10_x=True, log10_y=True)
ax.scatter(10**x_means, 10**y_means, zorder=3, linewidth=3, marker=plot_utils.environment_shape_dict[environment], facecolors=plot_utils.environment_facecolor_dict[environment], edgecolors=plot_utils.environment_cmap_dict[environment])

ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)

#ax.xaxis.set_tick_params(labelsize=8)
#ax.yaxis.set_tick_params(labelsize=8)

#ax.tick_params(axis='both', which='both', labelsize=12)
ax.tick_params(axis='both', which='both', labelsize=8)
ax.legend(loc='upper left', fontsize=9)


ax.set_xlabel("Total # of reads, " + r'$N$', fontsize=14)
ax.set_ylabel("# observed community members", fontsize=14)
ax.set_title("Abundance-Richness Scaling", fontsize=16, fontweight='bold')




fig.subplots_adjust(hspace=0.25, wspace=0.15)
fig_name = "%sabundance_richness.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()