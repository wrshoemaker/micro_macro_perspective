import os
import numpy
import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker
import sympy
import math
from scipy import special


import config
import data_utils
import plot_utils


def Klogn(emp_mad, c, mu0=-19,s0=5):
    # This function estimates the parameters (mu, s) of the lognormal distribution of K
    m1 = numpy.mean(numpy.log(emp_mad[emp_mad>c]))
    m2 = numpy.mean(numpy.log(emp_mad[emp_mad>c])**2)
    xmu = sympy.symbols('xmu')
    xs = sympy.symbols('xs')
    eq1 =- m1+xmu + numpy.sqrt(2/math.pi)*xs*sympy.exp(-((numpy.log(c)-xmu)**2)/2/(xs**2))/(sympy.erfc((numpy.log(c)-xmu)/numpy.sqrt(2)/xs))
    eq2 =- m2+xs**2+m1*xmu+numpy.log(c)*m1-xmu*numpy.log(c)
    sol = sympy.nsolve([eq1,eq2],[xmu,xs],[mu0,s0])

    return(float(sol[0]),float(sol[1]))


def get_lognorma_mad_prediction(x, mu, sigma, c):
    # x is the LOG of the mean abundance
    return numpy.sqrt(2/math.pi)/sigma*numpy.exp(-(x-mu)**2 /2/(sigma**2))/special.erfc((numpy.log(c)-mu)/numpy.sqrt(2)/sigma)



#min_occupancy = 0.1


fig, ax = plt.subplots(figsize=(4,4))


rescaled_ln_means_all = []
counts_start_all = []
for environment in data_utils.datasets_crossectional:

    environment_counts_np = data_utils.get_read_counts(environment, longitudinal_bool=False)
    n_reads_otus_np = numpy.sum(environment_counts_np, axis=1)
    #print(len(n_reads_otus_np))
    rel_environment_counts_np = environment_counts_np/numpy.sum(environment_counts_np, axis=0)
    means = numpy.mean(rel_environment_counts_np, axis=1)
    occupancy = numpy.sum(rel_environment_counts_np > 0, axis=1)/rel_environment_counts_np.shape[1]

    #means = means[occupancy >= min_occupancy]

    #mu_env, s_env = Klogn(means, 0.0001, mu0=-19,s0=5)
    #print( mu_env, s_env )


    ln_means = numpy.log(means)
    rescaled_ln_means = (ln_means - numpy.mean(ln_means))/numpy.std(ln_means)

    #rescaled_ln_means_all.append(rescaled_ln_means)

    rescaled_ln_means_to_plot = rescaled_ln_means[rescaled_ln_means>=0]

    counts, bin_edges = data_utils.bin_x(rescaled_ln_means_to_plot, n_bins=20, log10=False, normalize_counts=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    to_plot_idx = (counts>0)
    bin_centers = bin_centers[to_plot_idx]
    counts = counts[to_plot_idx]
    ax.scatter(bin_centers, counts, marker=plot_utils.environment_shape_dict[environment], facecolors=plot_utils.environment_facecolor_dict[environment], edgecolors=plot_utils.environment_cmap_dict[environment], zorder=1)

    counts_start_all.append(counts[0])



#flat_rescaled_ln_means_all = numpy.concatenate(rescaled_ln_means_all).ravel()

# code to fit MAD

#c = min(flat_rescaled_ln_means_all)*10
#mu, sigma = Klogn(flat_rescaled_ln_means_all, c)

x_mean_range = numpy.linspace(-3, 3, num=1000)
mu = 0
s = 1.2
#c = 0.001
offset = 0.1

rescaled_lognorm_pdf = numpy.mean(counts_start_all)*numpy.exp(-1*(x_mean_range**2)/2)


lognorm_pdf = get_lognorma_mad_prediction(x_mean_range, mu, 1, 0.001) 
lognorm_pdf_ = get_lognorma_mad_prediction(x_mean_range, mu, 1.5, 0.001) 
lognorm_pdf__ = get_lognorma_mad_prediction(x_mean_range, mu, 2, 0.001)



#lognorm_pdf_text = -1*(x_mean_range**2)

#ax.axhline(y=0.08, lw=1, ls='-')

#ax.plot(x_mean_range, lognorm_pdf, 'k', label='Lognormal', lw=2, ls='-', zorder=2)
#ax.plot(x_mean_range, lognorm_pdf_, 'k', label='Lognormal', lw=2, ls='--', zorder=2)
#ax.plot(x_mean_range, lognorm_pdf__, 'k', label='Lognormal', lw=2, ls=':', zorder=2)


ax.plot(x_mean_range, rescaled_lognorm_pdf, 'k', label='Lognormal', lw=2, ls='-', zorder=2)




#print(flat_rescaled_ln_means_all)

#ax.legend(loc='upper left', fontsize=9)
ax.set_yscale('log', base=10)

ax.xaxis.set_tick_params(labelsize=8)
ax.yaxis.set_tick_params(labelsize=8)

ax.set_xlim([-2.5, 2.5])
#ax.set_ylim([min(flat_rescaled_ln_means_all), 0.7])

ax.set_ylim([0.006, 0.2])


ax.set_xlabel("Rescaled " + r'$\mathrm{ln}$' + " mean relative abundance", fontsize=14)
ax.set_ylabel("Probability density", fontsize=14)







fig.subplots_adjust(hspace=0.25, wspace=0.15)
fig_name = "%smad.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()