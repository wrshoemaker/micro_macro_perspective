import os
import numpy
import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker
import pandas


import config
import data_utils
import plot_utils
import stats_utils



id_val = 'SRP056641 GUT'
environment = 'GUT'
# gut1
nbins = 12
min_mean_abund = 10**-7

proj, gamma_pars, mean_pars = data_utils.get_processed_data()
#subset_mean_pars = mean_pars[mean_pars["sname"] == target_env]


expected_occ = pandas.DataFrame()
#for id_val in mean_pars["idall"].unique():

row = mean_pars.loc[mean_pars["idall"] == id_val].iloc[0]
sname = row["sname"]
beta = row["mbeta"]
mu = row["mu"]
sigma = row["sigma"]
stot = int(row["stot"])

# Get nreads for this idall
nreads = proj.loc[proj["idall"] == id_val, ["nreads", "run_id"]].drop_duplicates()["nreads"].astype(int).values

etas = numpy.random.normal(loc=mu, scale=sigma, size=stot)

occupancys = numpy.outer(nreads, etas)
occupancys = numpy.vectorize(stats_utils.predict_occupancys)(nreads[:, None], etas[None, :], beta)

weight_eta = 1 - numpy.exp(numpy.sum(numpy.log(1 - occupancys), axis=0))
occ_eta = numpy.mean(occupancys, axis=0)

rnd_matrix = numpy.random.rand(*occupancys.shape)
rnd_occupancys = (rnd_matrix < occupancys).astype(float)
rndocc_eta = numpy.mean(rnd_occupancys, axis=0)


d = pandas.DataFrame({
    "idall": [id_val] * len(etas),
    "sname": [sname] * len(etas),
    "o": occ_eta,
    "ornd": rndocc_eta,
    "f": numpy.exp(etas),
    "pobs": weight_eta,
    "sigma": sigma,
    "mu": mu,
    "beta": beta
})


expected_occ = pandas.concat([expected_occ, d], ignore_index=True)



def bin_group(df, n_bins):
    df = df.copy()
    df["l"] = numpy.log10(df["f"])
    df["dl"] = (df["l"].max() - df["l"].min()) / n_bins
    df["b"] = ((df["l"] - df["l"].min()) / df["dl"]).astype(int)
    df = df.groupby(["idall", "sname", "b"], as_index=False).agg({
        "o": "mean",
        "f": "mean"})
    
    return df


def bin_group_expected(df, n_bins=20):
    df = df.copy()
    df["l"] = numpy.log10(df["f"])
    df["dl"] = (df["l"].max() - df["l"].min()) / n_bins
    df["b"] = ((df["l"] - df["l"].min()) / df["dl"]).astype(int)
    df = df.groupby(["idall", "sname", "b"], as_index=False).agg({
        "o": "mean",
        "f": "mean",
        "pobs": "mean" })
    return df

#expected_occ_binned = expected_occ.groupby(["idall", "sname"], group_keys=False).apply(bin_group_expected)

#occ_freq_binned = expected_occ.groupby(["idall", "sname"], group_keys=False).apply(lambda x: bin_group(x, nbins))

obs_occ = expected_occ['o'].values
pred_occ = expected_occ['ornd'].values
mean_abund = expected_occ['f'].values

to_keep = (obs_occ>0)&(pred_occ>0)&(mean_abund>=min_mean_abund)
obs_occ = obs_occ[to_keep]
pred_occ = pred_occ[to_keep]
mean_abund = mean_abund[to_keep]

log10_obs_occ = numpy.log10(obs_occ)
log10_pred_occ = numpy.log10(pred_occ)
log10_mean_abund = numpy.log10(mean_abund)


bin_centers_log10_obs, x_means_log10_obs, y_means_log10_obs = data_utils.bin_mean_xy(log10_mean_abund, log10_obs_occ, nbins=nbins, log10_x=False, log10_y=False)
bin_centers_log10_pred, x_means_log10_pred, y_means_log10_pred = data_utils.bin_mean_xy(log10_mean_abund, log10_pred_occ, nbins=nbins, log10_x=False, log10_y=False)


#x_means_log10_pred = numpy.argsort(x_means_log10_pred)
#x_means_log10_pred = x_means_log10_pred[x_means_log10_pred]
#y_means_log10_pred = y_means_log10_pred[x_means_log10_pred]


fig, ax = plt.subplots(figsize=(4,4))

ax.scatter(10**x_means_log10_obs, 10**y_means_log10_obs, marker=plot_utils.environment_shape_dict[environment], facecolors=plot_utils.environment_facecolor_dict[environment], edgecolors=plot_utils.environment_cmap_dict[environment])
ax.plot(10**x_means_log10_pred, 10**y_means_log10_pred, 'k', lw=2, ls='-', zorder=2)


ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)

ax.xaxis.set_tick_params(labelsize=8)
ax.yaxis.set_tick_params(labelsize=8)

#ax.set_ylim([min(flat_counts_all), 1])

ax.set_xlabel("Mean relative abundance, " + r'$\bar{x}_{i}$', fontsize=14)
ax.set_ylabel("Occupancy", fontsize=14)


fig.subplots_adjust(hspace=0.25, wspace=0.15)
fig_name = "%sabundance_occupancy.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()



#numpy.array(bin_centers), numpy.array(x_means), numpy.array(y_means) = bin_mean_xy(x, y, nbins=20, log10_x=False, log10_y=False)

