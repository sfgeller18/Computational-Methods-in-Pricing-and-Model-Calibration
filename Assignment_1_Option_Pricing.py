# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 21:45:59 2023

@author: simon
"""

import numpy as np # for fast vector computations
from time import time # to obtain the running time of some functions

def logNormal(S, r, q, sig, S0, T):
    ''' Computes the log normal density function for the stock price. '''
    f = np.exp(-0.5*((np.log(S/S0)-(r-q-sig**2/2)*T)/(sig*np.sqrt(T)))**2) / (sig*S*np.sqrt(2*np.pi*T))
    return f

# Parameters for put
S0 = 100
K = 90
r = 0.04
q = 0.02
sig = 0.25
T = 1.0

def numerical_integral_put(r, q, S0, K, sig, T, N):
    ''' Numerical integration for puts. '''
    
    # temporary value for output
    df = np.exp(-r * T)
    eta = K / N
    S = np.arange(0, N) * eta
    S[0] = 1e-8
    w = np.ones(N) * eta
    w[0] = eta / 2
    logN = np.zeros(N)
    logN = logNormal(S, r, q, sig, S0, T)
    sumP = np.sum((K - S) * logN * w)
    priceP = df * sumP 
    print("*")
    return eta, priceP

# vector with all values of n
n_min = 1
n_max = 15
n_vec = np.arange(n_min, n_max + 1, dtype=int)

# compute the numerical integration for each value of n
start_time = time()
# will store the results in vectors
eta_vec = np.zeros(n_max)
put_vec = np.zeros(n_max)
for i in range(n_max):
    N = 2** n_vec[i]
    [eta_vec[i], put_vec[i]] = numerical_integral_put(r, q, S0, K, sig, T, N)
end_time = time()
print('Elapsed time (seconds): ' + str(end_time - start_time))

# print a table with the numerical integration for each value of n
print('N\teta\tP_0')
for i in range(n_max):
    print('2^%i\t%.3f\t%.4f' % (n_vec[i], eta_vec[i], put_vec[i]))
    


def generic_CF(u, params, S0, r, q, T, model):
    ''' Computes the characteristic function for different models (BMS, Heston, VG). '''   
    
    if (model == 'BMS'):
        # unpack parameters
        sig = params[0]
        mu = np.log(S0) + (r-q-sig**2/2)*T
        a = sig*np.sqrt(T)
        phi = np.exp(1j*mu*u-(a*u)**2/2)
########################################    
    elif(model == 'Heston'):  
        # unpack parameters
        kappa  = params[0]
        theta  = params[1]
        sigma  = params[2]
        rho    = params[3]
        v0     = params[4]
        # cf
        tmp = (kappa-1j*rho*sigma*u)
        g = np.sqrt((sigma**2)*(u**2+1j*u)+tmp**2)        
        pow1 = 2*kappa*theta/(sigma**2)
        numer1 = (kappa*theta*T*tmp)/(sigma**2) + 1j*u*T*r + 1j*u*np.log(S0)
        log_denum1 = pow1 * np.log(np.cosh(g*T/2)+(tmp/g)*np.sinh(g*T/2))
        tmp2 = ((u*u+1j*u)*v0)/(g/np.tanh(g*T/2)+tmp)
        log_phi = numer1 - log_denum1 - tmp2
        phi = np.exp(log_phi)

    elif (model == 'VG'):
        # unpack parameters
        sigma  = params[0];
        nu     = params[1];
        theta  = params[2];
        # cf
        if (nu == 0):
            mu = np.log(S0) + (r-q - theta -0.5*sigma**2)*T
            phi  = np.exp(1j*u*mu) * np.exp((1j*theta*u-0.5*sigma**2*u**2)*T)
        else:
            mu  = np.log(S0) + (r-q + np.log(1-theta*nu-0.5*sigma**2*nu)/nu)*T
            phi = np.exp(1j*u*mu)*((1-1j*nu*theta*u+0.5*nu*sigma**2*u**2)**(-T/nu))

    return phi

def genericFFT(params, S0, K, r, q, T, alpha, eta, n, model):
    ''' Option pricing using FFT (model-free). '''
    
    N = 2**n
    
    # step-size in log strike space
    lda = (2 * np.pi / N) / eta
    
    # choice of beta
    #beta = np.log(S0)-N*lda/2 # the log strike we want is in the middle of the array
    beta = np.log(K) # the log strike we want is the first element of the array
    
    # forming vector x and strikes km for m=1,...,N
    km = np.zeros(N)
    xX = np.zeros(N)
    
    # discount factor
    df = np.exp(-r*T)
    
    nuJ = np.arange(N) * eta
    psi_nuJ = generic_CF(nuJ - (alpha + 1) * 1j, params, S0, r, q, T, model) / ((alpha + 1j*nuJ)*(alpha+1+1j*nuJ))
    
    km = beta + lda * np.arange(N)
    w = eta * np.ones(N)
    w[0] = eta / 2
    xX = np.exp(-1j * beta * nuJ) * df * psi_nuJ * w
     
    yY = np.fft.fft(xX)
    cT_km = np.zeros(N) 
    multiplier = np.exp(-alpha * km) / np.pi
    cT_km = multiplier * np.real(yY)
    
    return km, cT_km

# parameters
S0 = 100
K = 80
r = 0.05
q = 0.01
T = 1.0

# parameters for fft
eta_vec = np.array([0.1, 0.25])
n_vec = np.array([6, 10])
alpha_vec = np.array([-1.01, -1.25, -1.5, -1.75, -2., -5.])
num_prices = len(eta_vec) * len(n_vec) * len(alpha_vec)

def price_all_puts(params, S0, K, r, q, T, model, alpha_vec, eta_vec, n_vec):
    num_prices = len(eta_vec) * len(n_vec) * len(alpha_vec)
    # output is a matrix, the columns correspond to eta, n, alpha, and put price
    put_matrix = np.zeros([num_prices, 4])
    i = 0
    for eta in eta_vec:
        for n in n_vec:
            for alpha in alpha_vec:
                # pricing via fft
                put = 0 # store here the put price
                k_vec, option_vec = genericFFT(params, S0, K, r, q, T, alpha, eta, n, model)
                put = option_vec[0] # only interested in one strike
                put_matrix[i] = np.array([eta, n, alpha, put])
                i += 1
    return put_matrix

# model-specific parameters
mod = 'BMS'
sig = 0.3
params = [sig]

# model-specific parameters
mod = 'Heston'
kappa = 2.
theta = 0.05
lda = 0.3
rho = -0.7
v0 = 0.04
params = [kappa, theta, lda, rho, v0]

# model-specific parameters
mod = 'VG'
sigma = 0.3
nu = 0.5
theta = -0.4
params = [sigma, nu, theta]

start_time = time()
bms_matrix = price_all_puts(params, S0, K, r, q, T, mod, alpha_vec, eta_vec, n_vec)
end_time = time()
print('Elapsed time (seconds): ' + str(end_time - start_time))

# print results in table form
print('Model = ' + mod)
print('eta\tN\talpha\tput')
for i in range(num_prices):
    print('%.2f\t2^%i\t%.2f\t%.4f' % (bms_matrix[i,0], bms_matrix[i,1], bms_matrix[i,2], bms_matrix[i,3]))