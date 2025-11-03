import os, math
from collections import Counter
import numpy
from scipy.special import digamma, polygamma, gamma
from scipy import special
from scipy.stats import nbinom


import pyreadr
import config
import sympy




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



def standardized_loggamma_pdf(z, k):
    
    # z = rescaled log gamma rv

    psi = digamma(k)
    psi1 = polygamma(1, k)  # trigamma
    sigma = numpy.sqrt(psi1)

    # compute standardized log-gamma pdf
    pdf = (sigma / gamma(k)) * numpy.exp(k * (psi + sigma * z) - numpy.exp(psi + sigma * z))
    return pdf




def predicted_sp(N, mu, sigma, beta, sp_tot):

    eta = numpy.random.normal(loc=mu, scale=sigma, size=int(1e6))
    p0 = numpy.mean((1 + N * numpy.exp(eta) / beta) ** (-beta))
    return sp_tot * (1 - p0)



def predicted_shannonindex(N, mu, sigma, beta, sp_tot):

    s = []
    sp_tot = int(round(sp_tot)) 
    for _ in range(100):
        eta = numpy.random.normal(loc=mu, scale=sigma, size=sp_tot)
        prob = beta / (beta + N * numpy.exp(eta))
        
        # Negative binomial with parameters (n=beta, p=prob)
        k = nbinom.rvs(n=beta, p=prob)
        
        # Normalize nonzero counts
        k = k[k > 0].astype(float) / N
        
        if len(k) > 0:
            shannon_val = -numpy.sum(k * numpy.log(k))
            if shannon_val > 0:
                s.append(shannon_val)
    
    return numpy.mean(s) if len(s) > 0 else numpy.nan


def predicted_reads(N, mu, sigma, beta, sp_tot):
    eta = numpy.random.normal(loc=mu, scale=sigma, size=int(1e6))
    p0 = numpy.mean(N * numpy.exp(eta))
    return sp_tot * p0


def predict_occupancys(N, eta, beta):

    p0 = (beta / (beta + N * numpy.exp(eta))) ** beta
    return 1 - p0