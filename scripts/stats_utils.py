import os, math
from collections import Counter
import numpy
from scipy.special import digamma, polygamma, gamma
from scipy import special
from scipy.stats import nbinom
from scipy.signal import welch, csd, windows



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



def autocorrelation_by_days_old(data, days, min_n_autocorr_values=5):

    delay_days_range = numpy.arange(1, max(days) - min(days) - min_n_autocorr_values)
    # Rescale data by mean
    data = data - numpy.mean(data)

    delay_days_all = []
    autocorr_all = []

    for delay_days in delay_days_range:

        autocorr = []
        
        # Loop through all possible lags
        for i in range(len(data) - delay_days):
            current_day = days[i]
            lagged_day = days[i + delay_days]
            
            # Check if the difference in days equals delay_days
            if lagged_day - current_day == delay_days:
                current_value = data[i]
                lagged_value = data[i + delay_days]
                # Save product of current and lagged value
                autocorr.append((current_value) * (lagged_value))

        # Sufficient number of datapoints to calcualte autocorrelation
        if len(autocorr) >= min_n_autocorr_values:
            # Normalize covariance by sum of covariance terms divided by # of covariance terms
            # and the variance  
            delay_days_all.append(delay_days)  
            #autocorr_all.append(numpy.sum(autocorr) / (len(autocorr) * numpy.var(data)))
            autocorr_all.append(numpy.mean(autocorr) / numpy.var(data))


    delay_days_all = numpy.asarray(delay_days_all)
    autocorr_all = numpy.asarray(autocorr_all)

    delay_days_all = numpy.insert(delay_days_all, 0, 0, axis=0)
    autocorr_all = numpy.insert(autocorr_all, 0, 1, axis=0)


    return delay_days_all, autocorr_all



def autocorrelation_by_days(data, days, min_n_autocorr_values=5):

    data = numpy.asarray(data, dtype=float)
    days = numpy.asarray(days, dtype=float)
    
    data_centered = data - numpy.mean(data)
    
    delay_days_range = numpy.arange(1, int(days.max() - days.min() - min_n_autocorr_values) + 1)
    
    delay_days_all = [0]
    autocorr_all = [1.0]
    
    for lag in delay_days_range:
        idx_current = []
        idx_lagged = []
        for i in range(len(days) - lag):
            if days[i + lag] - days[i] == lag:
                idx_current.append(i)
                idx_lagged.append(i + lag)
        
        if len(idx_current) >= min_n_autocorr_values:
            x = data_centered[idx_current]
            y = data_centered[idx_lagged]
            acf_value = numpy.mean(x * y) / numpy.var(x)
            acf_value = numpy.clip(acf_value, -1, 1)
            
            delay_days_all.append(lag)
            autocorr_all.append(acf_value)
    
    return numpy.array(delay_days_all), numpy.array(autocorr_all)



def crosscorrelation_by_days(data_i, data_j, days, min_n_corr_values=5):
    
    data_i = numpy.asarray(data_i, dtype=float)
    data_j = numpy.asarray(data_j, dtype=float)
    days = numpy.asarray(days, dtype=float)

    i_center = data_i - numpy.mean(data_i)
    j_center = data_j - numpy.mean(data_j)

    global_denom = numpy.sqrt(numpy.var(i_center) * numpy.var(j_center))
    if global_denom == 0:
        raise ValueError("One of the signals has zero variance.")

    max_lag = int(days.max() - days.min() - min_n_corr_values)

    lags = numpy.arange(-max_lag, max_lag + 1)

    lag_list = []
    corr_list = []

    for lag in lags:
        idx_t = []
        idx_tlag = []

        if lag == 0:
            xi = i_center
            xj = j_center

        else:
            for k in range(len(days)):
                k_lag = k + lag
                if 0 <= k_lag < len(days):
                    if days[k_lag] - days[k] == lag:
                        idx_t.append(k)
                        idx_tlag.append(k_lag)

            if len(idx_t) < min_n_corr_values:
                continue

            xi = i_center[idx_t]
            xj = j_center[idx_tlag]

        corr = numpy.mean(xi * xj) / global_denom
        corr = float(numpy.clip(corr, -1, 1))

        lag_list.append(lag)
        corr_list.append(corr)

    lag_list = numpy.array(lag_list)
    corr_list = numpy.array(corr_list)

    zero_idx = numpy.where(lag_list == 0)[0]
    if len(zero_idx) > 0:
        c0 = corr_list[zero_idx[0]]
        if c0 != 0:
            corr_list /= c0

    corr_list = numpy.clip(corr_list, -1, 1)


    return lag_list, corr_list


def cpsd_welch_matlab(X, n, h, nfft, window, noverlap, fs=1.0):
    '''
    MATLAB-style cross-power spectral density using Welch method.

    # matlab cpsd computes cpsd using a window, overlap, and FFT length.
    # scipy.signal.csd in Python does same, with similar arguments (nperseg, noverlap, nfft, window)

    ### Scaling
    #MATLAB cpsd returns values normalized by sampling frequency and the window so that integrating the PSD gives signal variance.
    #SciPy csd normalizes, but scaling can differ depending on the scaling argument ('density' or 'spectrum').

    #density = power per Hz (like MATLAB’s default).
    #spectrum = power, but not per Hz (slightly different).

    ### Windowing 
    # MATLAB: default hamming window.
    #SciPy: default hann window. Set window='hamming'

    # same:
    # window type and length
    # nfft / frequency resolution
    # scaling (usually 'density').
    # sampling frequency matches (fs in Python).
    
    Input
    X: ndarray, shape (time, n_channels), single-trial time series data
    n: int, number of channels
    h: int, number of frequency bins (MATLAB uses nfft=2*(h-1), I think...) 
    window : int, window length
    noverlap : int, overlap between windows
    fs: float, sampling frequency (default 1), necessary for exact frequency axis

    Output
    S: ndarray, shape (n, n, h), cross-power spectral density matrix
    '''

    # definition used by MATLAB
    #nfft = 2*(h-1)
    #window = min([X.shape[0], nfft])
    #window = int(X.shape[0] / 2)
    #noverlap = window // 2    # 50% overlap
    #noverlap = 30

    S = numpy.zeros((n, n, h), dtype=complex)
    
    # MATLAB uses 'hamming' window by default
    # or (pwelch)?
    win = windows.hamming(window, sym=False)
    
    for i in range(n):
        # scaling='density' ==> matches MATLAB’s PSD units
        # return_onesided=True ==> MATLAB uses one-sided PSD by default
        f, Pxx = welch(X[:,i], fs=fs, window=win, nperseg=window, noverlap=noverlap, nfft=nfft, return_onesided=True, scaling='density')
        S[i,i,:] = Pxx
        for j in range(i+1, n):
            f, Pxy = csd(X[:,i], X[:,j], fs=fs, window=win, nperseg=window, noverlap=noverlap, nfft=nfft, return_onesided=True, scaling='density')
            S[i,j,:] = Pxy
    
    # lower triangle with complex conjugates ==>  matches MATLAB CPSD output
    # [i, j, :] slice of S is CPSD between i and j
    for i in range(n):
        for j in range(i+1, n):
            S[j,i,:] = numpy.conj(S[i,j,:])
    
    return S
