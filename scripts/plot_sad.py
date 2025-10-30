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
# gut1
npoints = 10


proj, gamma_pars, mean_pars = data_utils.get_processed_data()



def predicted_cumsad(N, mu, sigma, beta, sp_tot, klist):
    eta = numpy.random.normal(loc=mu, scale=sigma, size=100000)    
    p0 = numpy.mean((beta / (beta + N * numpy.exp(eta))) ** beta)
    
    results = []
    
    for k in klist:
        prob = beta / (beta + N * numpy.exp(eta))
        
        # R: pnbinom(q, size, prob) = P(X ≤ q)
        # 1 - pnbinom(k-1, ...) = P(X ≥ k)
        sk = numpy.mean(1 - nbinom.cdf(k - 1, n=beta, p=prob)) / (1 - p0)
        
        results.append({"k": k, "sk": sk})
    
    return pandas.DataFrame(results)




sads_expected = pandas.DataFrame()

#for id_ in mean_pars["idall"].unique():
#    print(id_)
    
# Extract parameters
row = mean_pars[mean_pars["idall"] == id_val].iloc[0]
sname = row["sname"]
beta = row["mbeta"]
mu = row["mu"]
sigma = row["sigma"]
stot = row["stot"]

# Subset proj
proj_sub = proj[proj["idall"] == id_val]
nmin = int(proj_sub["nreads"].min())
nmax = int(proj_sub["nreads"].max())

dn = (numpy.log10(nmax) - numpy.log10(nmin)) / npoints

# nmin
klist = (10 ** (numpy.arange(0, int(numpy.log10(nmin / 2) * 5) + 1) / 5.)).astype(int)
sp_pred = stats_utils.predicted_sp(nmin, mu, sigma, beta, stot)
res = predicted_cumsad(nmin, mu, sigma, beta, stot, klist)
res = res.assign(idall=id_val, sname=sname, nreads=nmin, sp=sp_pred)
sads_expected = pandas.concat([res, sads_expected], ignore_index=True)

# nmax
klist = (10 ** (numpy.arange(0, int(numpy.log10(nmax / 2) * 5) + 1) / 5.)).astype(int)
sp_pred = stats_utils.predicted_sp(nmax, mu, sigma, beta, stot)
res = predicted_cumsad(nmax, mu, sigma, beta, stot, klist)
res = res.assign(idall=id_val, sname=sname, nreads=nmax, sp=sp_pred)
sads_expected = pandas.concat([res, sads_expected], ignore_index=True)


df = proj_sub.copy()

# make strings categorical
cat_cols = ["idall", "run_id", "sname"]
for c in cat_cols:
    if c in df.columns and df[c].dtype != "category":
        df[c] = df[c].astype("category")

# calculate sp
df["sp"] = df.groupby(["nreads", "idall", "run_id", "sname"], observed=True)["otu_id"].transform("nunique")

# aggregate counts => sk
group_cols = ["count", "nreads", "idall", "run_id", "sname", "sp"]
sads = df.groupby(group_cols, observed=True).size().reset_index(name="sk")

# make counts numeric
sads["count"] = pandas.to_numeric(sads["count"], errors="coerce")

# Sort and then calculate pcum
sads = sads.sort_values(["idall", "sname", "run_id", "count"], ascending=[True, True, True, False])
group_for_pcum = ["idall", "sname", "run_id"]
sads["pcum"] = sads.groupby(group_for_pcum)["sk"].cumsum() / sads.groupby(group_for_pcum)["sk"].transform("sum")





fig, ax = plt.subplots(figsize=(4,4))


for run_id, df_run in sads.groupby("run_id"):
    x = df_run["count"].values
    y = df_run["pcum"].values
    ax.plot(x, y, c=plot_utils.environment_cmap_dict[environment], lw=1, ls='-', alpha=0.3)


count = 0
for nreads, df_pred in sads_expected.groupby("nreads"):        
    x = df_pred["k"].values
    y = df_pred["sk"].values
    ax.plot(x, y, c='k', lw=2, ls='-', alpha=1)

    # lazy
    if count == 0:
        ax.plot(x, y, c='k', lw=2, ls='-', alpha=1, label='Prediction')

    else:
        ax.plot(x, y, c='k', lw=2, ls='-', alpha=1)

    count += 1



ax.xaxis.set_tick_params(labelsize=8)
ax.yaxis.set_tick_params(labelsize=8)

#ax.set_ylim([min(flat_density_all), 1])
ax.set_xlim([0.8, 20000])
ax.set_ylim([0.0008, 1.4])

ax.set_title('Species Abundance\nDistribution (SAD)', fontsize=16, fontweight='bold')

ax.legend(loc='upper right', fontsize=9)



ax.set_xlabel("# reads, " + r'$n$', fontsize=14)
ax.set_ylabel("Fraction community\nmembers with " + r'$\geq n$' + ' reads', fontsize=14)

ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)


fig.subplots_adjust(hspace=0.25, wspace=0.15)
fig_name = "%ssad.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()