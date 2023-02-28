# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 21:53:46 2023

@author: simon
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
%matplotlib inline
# for interactive figures
#%matplotlib notebook
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from scipy.stats import norm
from scipy import optimize
import cmath
import math
import readPlotOptionSurface 
import modulesForCalibration as mfc

# file name
excel_file = 'data_apple.xlsx'
# read data into a data frame
df = pd.read_excel(excel_file)
# create the 'Mid' variable
df['Mid'] = df[['Bid','Ask']].mean(axis=1)
# see some of the data
df.head()

# define strikes and maturities
#all_strikes = np.sort(df_calls.Strike.unique())
all_strikes = np.arange(170., 210. + 2.5, 2.5)
all_maturities = np.sort(df.Maturity_days.unique())
print(all_strikes)
print(all_maturities)
df_puts = df[df['Option_type'] == 'Put'][['Maturity_days', 'Strike', 'Mid']]
df_puts.head()

# define a grid for the surface
X, Y = np.meshgrid(all_strikes, all_maturities)
Z_p = np.empty([len(all_maturities), len(all_strikes)])

# Use linear interpolation to fill missing strikes
for i in range(len(all_maturities)):
    f = interpolate.interp1d(s, price, bounds_error=False, fill_value=0)
    Z_p[i, :] = f(all_strikes) 
    
# plot the surface
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z_p, cmap=cm.coolwarm)
ax.set_ylabel('Maturity (days)') 
ax.set_xlabel('Strike') 
ax.set_zlabel('P(K, T)')
ax.set_title('Apple Puts')
plt.savefig('fig3.png')
plt.show()

# select strike and maturity
K = 190.
T_days = 151
T_years = 1.* T_days / 365

# dividend rate
q = 0.005

# risk free rate
r = 0.0245

# spot price
S_0 = 190.3

# price
P_call = 10.875
P_put = 9.625

# utilize the following black scholes calculator to get the price from BS model
def BS_d1(S, K, r, q, sigma, tau):
    ''' Computes d1 for the Black Scholes formula '''
    d1 = 1.0*(np.log(1.0 * S/K) + (r - q + sigma**2/2) * tau) / (sigma * np.sqrt(tau))
    return d1

def BS_d2(S, K, r, q, sigma, tau):
    ''' Computes d2 for the Black Scholes formula '''
    d2 = 1.0*(np.log(1.0 * S/K) + (r - q - sigma**2/2) * tau) / (sigma * np.sqrt(tau))
    return d2

def BS_price(type_option, S, K, r, q, sigma, T, t=0):
    ''' Computes the Black Scholes price for a 'call' or 'put' option '''
    tau = T - t
    d1 = BS_d1(S, K, r, q, sigma, tau)
    d2 = BS_d2(S, K, r, q, sigma, tau)
    if type_option == 'call':
        price = S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    elif type_option == 'put':
        price = K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1) 
    return price

# auxiliary function for computing implied vol
def aux_imp_vol(sigma, P, type_option, S, K, r, q, T, t=0):
    s=0.0
    for sig in np.arange(0.01,0.40,0.001):
        if BS_price(type_option, S,K,r,q,sig,T,t=0)<P:
            s=sig
            break
    return s

imp_vol_call = optimize.brentq(aux_imp_vol, 0.01, 0.4, args=(P_call, 'call', S_0, K, r, q, T_years))
imp_vol_put = optimize.brentq(aux_imp_vol, 0.01, 0.4, args=(P_put, 'put', S_0, K, r, q, T_years))

# Parameters
alpha = 1.5
eta = 0.2
    
n = 12

# Model
model = 'Heston' 
# risk free rate
r = 0.0245
# dividend rate
q = 0.005
# spot price
S0 = 190.3

params1 = (1.0, 0.02, 0.05, -0.4, 0.08)
params2 = (3.0, 0.06, 0.10, -0.6, 0.04)

iArray = []
rmseArray = []
rmseMin = 1e10

maturities, strikes, callPrices = readPlotOptionSurface.readNPlot()

marketPrices = callPrices
maturities_years = maturities/365.0

#Model Initial Value Calibration
for i in mfc.myRange(0, 1, 0.05):
    params = i*np.array(params1) + (1.0-i)*np.array(params2)
    iArray.append(i)
     
    rmse = mfc.eValue(params, marketPrices, maturities_years, strikes, r, q, S0, alpha, eta, n, model)
    rmseArray.append(rmse)
    if (rmse < rmseMin):
        rmseMin = rmse
        optimParams = params
        
# use the following fixed Parameters to calibrate the model
S0 = 100
K = 80
k = np.log(K)
r = 0.05
q = 0.015

# Parameters setting in fourier transform
alpha = 1.5
eta = 0.2

n = 12
N = 2**n
# step-size in log strike space
lda = (2*np.pi/N)/eta

#Choice of beta
beta = np.log(S0)-N*lda/2

# Model
model = 'Heston'

# calculate characteristic function of different models
def generic_CF(u, params, T, model):
    
    if (model == 'GBM'):
        
        sig = params[0];
        mu = np.log(S0) + (r-q-sig**2/2)*T;
        a = sig*np.sqrt(T);
        phi = np.exp(1j*mu*u-(a*u)**2/2);
        
    elif(model == 'Heston'):
        
        kappa  = params[0];
        theta  = params[1];
        sigma  = params[2];
        rho    = params[3];
        v0     = params[4];
        tmp = (kappa-1j*rho*sigma*u);
        g = np.sqrt((sigma**2)*(u**2+1j*u)+tmp**2);
        
        pow1 = 2*kappa*theta/(sigma**2);

        numer1 = (kappa*theta*T*tmp)/(sigma**2) + 1j*u*T*r + 1j*u*math.log(S0);
        log_denum1 = pow1 * np.log(np.cosh(g*T/2)+(tmp/g)*np.sinh(g*T/2));
        tmp2 = ((u*u+1j*u)*v0)/(g/np.tanh(g*T/2)+tmp);
        log_phi = numer1 - log_denum1 - tmp2;
        phi = np.exp(log_phi);

    elif (model == 'VG'):
        
        sigma  = params[0];
        nu     = params[1];
        theta  = params[2];

        if (nu == 0):
            mu = math.log(S0) + (r-q - theta -0.5*sigma**2)*T;
            phi  = math.exp(1j*u*mu) * math.exp((1j*theta*u-0.5*sigma**2*u**2)*T);
        else:
            mu  = math.log(S0) + (r-q + math.log(1-theta*nu-0.5*sigma**2*nu)/nu)*T;
            phi = cmath.exp(1j*u*mu)*((1-1j*nu*theta*u+0.5*nu*sigma**2*u**2)**(-T/nu));

    return phi

# calculate option price by inverse fourier transform
def genericFFT(params, T):
    
    # forming vector x and strikes km for m=1,...,N
    km = []
    xX = []
    
    # discount factor
    df = math.exp(-r*T)
    
    for j in range(N):
        
        nuJ=j*eta
        km.append(beta+j*lda)
        
        psi_nuJ = df*generic_CF(nuJ-(alpha+1)*1j, params, T, model)/((alpha + 1j*nuJ)*(alpha+1+1j*nuJ))
        if j == 0:
            wJ = (eta/2)
        else:
            wJ = eta
        
        xX.append(cmath.exp(-1j*beta*nuJ)*psi_nuJ*wJ)
     
    yY = np.fft.fft(xX)
    
    cT_km = []    
    for i in range(N):
        multiplier = math.exp(-alpha*km[i])/math.pi
        cT_km.append(multiplier*np.real(yY[i]))
    
    return km, cT_km

# myRange(a, b) return a generator [a, a+1, ..., b], which is different from built-in generator Range that returns [a, a+1,..., b-1]. 
# You may use it to perform brute force
def myRange(start, finish, increment):
    while (start <= finish):
        yield start
        start += increment
        
# load virtual option data
data = pd.read_csv("Virtual Option Data.csv", index_col=0)

# generate strike and maturity array
strikes = np.array(data.index, dtype=float)
maturities = np.array(data.columns, dtype=float)
marketPrices = data.values

modelPrices = np.zeros_like(marketPrices)

rmseMin = 1.0e6
### part need to be finished ###
for kappa in myRange(2.5,3.0,0.25):
    for theta in myRange(0.06,0.065,0.025):
        for sig in myRange(0.1,0.3,0.05):
            for rho in myRange(-0.675,-0.625,0.025):
                for v0 in myRange(0.04,0.06,0.01):
                    params = []
                    params.append(kappa)
                    params.append(theta)
                    params.append(sig)
                    params.append(rho)
                    params.append(v0)
                    mae = genericFFT(params, )
                    
                    if (rmse < rmseMin):
                        rmseMin = rmse
                        params2 = params
                        
print(params)