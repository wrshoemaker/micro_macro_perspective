
import os
from collections import Counter
import numpy

import pyreadr
import config


import stats_utils
from scipy.special import erfc


#dataset_name_in_rdata = {'oral': 'oralcavity', 'Glacier': 'Glacier', 'Human gut': 'feces', 'River': 'River', 'Lake': 'Lake', 'Sludge': 'activatedsludge', 'Seawater':'seawater', 'Soil':'Environmental Terrestrial Soil'}
# ['feces', 'Glacier', 'oralcavity', 'Environmental Aquatic Marine Hydrothermal vents', 'skin', 'River', 'Lake', 'GUT', 'activatedsludge', 'Environmental Aquatic Marine', 'Environmental Terrestrial Soil', 'ORAL', ' seawater', 'VAGINAL']


#datasets = [('Oral', 'crosssectional'), ('Glacier', 'crossectional'), ('Human gut', 'crossectional'), 
#                ('River', 'crossectional'), ('Lake', 'crossectional'), ('Sludge', 'crossectional'),
#                ('Seawater', 'Crossectional'), ('Soil', 'crossectional')]


datasets_longitudinal = ['oralcavity', 'skin', 'feces']
datasets_crossectional = ['Glacier', 'GUT', 'ORAL', 'Lake', 'Environmental Aquatic Marine', 'River', 'activatedsludge',  'Environmental Terrestrial Soil']




def bin_x(x, n_bins=10, log10=True):

    x = numpy.asarray(x)
    
    if log10:
        x = x[x > 0]
        bin_edges = numpy.logspace(numpy.log10(x.min()), numpy.log10(x.max()), n_bins + 1)
    else:
        bin_edges = numpy.linspace(x.min(), x.max(), n_bins + 1)
    
    counts, edges = numpy.histogram(x, bins=bin_edges)

    bin_widths = numpy.diff(edges)
    density = counts / numpy.sum(counts * bin_widths) 

    #if normalize_counts == True:
    #    counts = counts/sum(counts)
    
    return counts, density, bin_edges


def bin_xy(x, y, n_bins=10, log10_x=True, log10_y=True):

    x = numpy.asarray(x)
    y = numpy.asarray(y)

    if log10_x:
        x = x[x > 0]
        x_edges = numpy.logspace(numpy.log10(x.min()), numpy.log10(x.max()), n_bins + 1)
    else:
        x_edges = numpy.linspace(x.min(), x.max(), n_bins + 1)

    if log10_y:
        y = y[y > 0]
        y_edges = numpy.logspace(numpy.log10(y.min()), numpy.log10(y.max()), n_bins + 1)
    else:
        y_edges = numpy.linspace(y.min(), y.max(), n_bins + 1)

    hist, _, _ = numpy.histogram2d(x, y, bins=[x_edges, y_edges])
    
    return hist, x_edges, y_edges


# binning strategy for mean of the x and y values along the x axis

def bin_mean_xy(x, y, nbins=20, log10_x=False, log10_y=False):
    x = numpy.asarray(x)
    y = numpy.asarray(y)

    if log10_x:
        mask = x > 0
        x = numpy.log10(x[mask])
        y = y[mask]

    # Log-transform y if requested
    if log10_y:
        mask = y > 0
        y = numpy.log10(y[mask])
        x = x[mask]

    # Define bin edges based on x
    bins = numpy.linspace(numpy.nanmin(x), numpy.nanmax(x), nbins + 1)
    bin_indices = numpy.digitize(x, bins) - 1 

    x_means = []
    y_means = []
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for i in range(nbins):
        in_bin = bin_indices == i
        if numpy.any(in_bin):
            x_means.append(numpy.mean(x[in_bin]))
            y_means.append(numpy.mean(y[in_bin]))
        else:
            x_means.append(numpy.nan)
            y_means.append(numpy.nan)

    # If x was log-transformed, invert bin centers back to linear scale (optional)
    if log10_x:
        bin_centers = 10 ** bin_centers

    return numpy.array(bin_centers), numpy.array(x_means), numpy.array(y_means)




#x_centers = 0.5 * (x_edges[:-1] + x_edges[-1:])
#y_centers = 0.5 * (y_edges[:-1] + y_edges[-1:])



def get_read_counts(environment, longitudinal_bool=False):

    if longitudinal_bool == True:
        rdatapath = '%slongitudinal.RData' % config.data_directory
        df_name = 'proj_time'
        datasets = datasets_longitudinal

    else:
        rdatapath = '%scrosssecdata.RData' % config.data_directory
        df_name = 'datatax'
        datasets = datasets_crossectional


    result = pyreadr.read_r(rdatapath)
    result_df = result[df_name]

    # subset
    environment_df = result_df[result_df["classification"] == environment]
    #print(environment_df.shape)
    # check for duplicate rows (same OTU + site combo more than once)
    if environment_df.duplicated(subset=['sample_id', 'otu_id']).sum() > 0:
        environment_df = environment_df.drop_duplicates(subset=['sample_id', 'otu_id'])

    environment_counts_df = environment_df.pivot(index='sample_id', columns='otu_id', values='count')
    environment_counts_df = environment_counts_df.fillna(0)
    # (OTUs, sites)
    environment_counts_np = environment_counts_df.to_numpy().T

    #environment_counts_np = environment_counts_np/numpy.sum(environment_counts_np, axis=0)
    
    return environment_counts_np




def get_processed_data():

    rdatapath = '%sdataestimate.RData' % config.data_directory
    result = pyreadr.read_r(rdatapath)

    proj = result['proj']
    gamma_pars = result['gamma_pars']
    mean_pars = result['mean_pars']

    #dh = result[result['o'] > 0.999].copy()

    return proj, gamma_pars, mean_pars



def get_mad_grilli_data():

    proj, gamma_pars, mean_pars = get_processed_data()

    statp = gamma_pars.copy()

    # Calculate log(f) and cutoff
    statp["lf"] = numpy.log(statp["f"])
    statp["cutoff"] = -100

    nbin = 20

    # Calculate df and bin index within each idall group
    def compute_bins(g):
        lf_min = g["lf"].min()
        lf_max = g["lf"].max()
        df = (lf_max - lf_min) / nbin
        g["df"] = df
        g["b"] = ((g["lf"] - lf_min) / df).astype(int)
        return g

    statp = statp.groupby("idall", group_keys=False).apply(compute_bins)

    # Summarise idall, b, df
    statp = (statp.groupby(["idall", "b", "df"], as_index=False).agg(lf=("lf", "mean"),cutoff=("cutoff", "mean"),n=("lf", "size")))

    # Calculate p = n / sum(n) / df within each idall
    statp["p"] = (statp.groupby("idall", group_keys=False).apply(lambda g: g["n"] / g["n"].sum() / g["df"]).reset_index(drop=True))

    print(statp)

    statp["xx"] = numpy.random.rand(len(statp))
    merged = statp.merge(mean_pars, on="idall", how="left")
    filtered = merged.query("lf > c + 0.15 and n > 10")

    filtered = filtered.sort_values(["xx", "sname"], ignore_index=True)

    # x and y vectors
    x = (filtered["lf"] - filtered["mu"]) / numpy.sqrt(2 * filtered["sigma"]**2)
    y = 10 ** (numpy.log10(filtered["p"]) - numpy.log10(0.5 * erfc((filtered["mu"] - filtered["c"]) / numpy.sqrt(2) / filtered["sigma"])) + 0.5 * numpy.log10(2 * numpy.pi))

    x = x.to_numpy()
    y = y.to_numpy()

    df_xy = filtered.assign(x=x, y=y)
    #df_xy[df_xy['sname'] == 'seawater'][['x', 'y']]


    return df_xy



def matching_pairs(t, delta_t):
    
    t = numpy.asarray(t)
    
    index_of = {time: idx for idx, time in enumerate(t)}
    
    pairs = []
    for i, ti in enumerate(t):
        target = ti + delta_t
        if target in index_of:
            j = index_of[target]
            if j > i:
                pairs.append((i, j))
    
    return pairs


def discretized_growth_rate(t, x, delta_t, divide_by_delta_t=True):
    
    t = numpy.asarray(t)
    x = numpy.asarray(x)

    pairs = matching_pairs(t, delta_t)

    t_mid_all = []
    g_all = []
    for i, j in pairs:
        
        if x[i] <= 0 or x[j] <= 0:
            raise ValueError(f"X contains non-positive value at index {i} or {j}; log undefined.")
        
        #dt = t[j] - t[i]
        g = (numpy.log(x[j]) - numpy.log(x[i]))

        if divide_by_delta_t == True:
            g = g / delta_t
        
        t_mid_all.append(0.5 * (t[i] + t[j]))
        g_all.append(g)

    
    return numpy.asarray([t_mid_all]), numpy.array(g_all)




if __name__ == "__main__":


    df_xy = get_mad_grilli_data()

    print(set(df_xy['sname'].tolist()))

    pairs = df_xy[['idall', 'sname']].drop_duplicates()
    idall_sname = list(zip(pairs['idall'], pairs['sname']))

    print(idall_sname)
    print('Hmmmm')
