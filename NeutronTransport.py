# Master script for determining solution to 1D slab neutron transport

import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
from scipy import constants

# set up grid 
a = 1
I = 10001
N = 8
x = np.linspace(0, a, I)
delta = x[2] - x[0]

# preallocate angular flux vectors and scalar flux and set boundary conditions
psi = np.zeros((N,I))
psi[:,0] = 0
psi[:,-1] = 0
phi = np.zeros(I)
phiPrev = np.zeros(len(phi))
phi_0 = 0 # initial guess for phi



# Set up piecewise cross section vector 
sig = 2.74E-24
sigV = np.zeros(I)
sigV[:] = sig

# set error tolerances and q vector
error = 10
err = 1E-9
S = 10E-5 # specified source term TBD
q = np.zeros(I)
q[:] = 0.5*sig*phi_0 + S

# P_N quadrature for N = 8 sum of w_n = 2
mu_n = np.array([-0.9602898564,-0.7966664774,-0.5255324099,-0.18343464240,0.1834346424,0.52553240990,0.7966664774,0.9602898564])
w_n = np.array([-0.1012285363,-0.2223810344,-0.3137066459,-0.3626837834,0.3626837834,0.3137066459,0.2223810344,0.1012285363])



while error > err:
    for n in range(len(mu_n)):

        if mu_n[n] > 0:

            for i in range(1, len(x)-1, 2):                
                psi[n,i] = (1 + 0.5*sigV[i]*delta/abs(mu_n[n]))**(-1)*(psi[n,i-1] + 0.5*delta*q[i]/abs(mu_n[n]))
                psi[n,i+1] = 2*psi[n,i] - psi[n,i-1]
            
        else:

            for i in range(I-2, 0, -2):
                psi[n,i] = (1 + 0.5*sigV[i]*delta/abs(mu_n[n]))**(-1)*(psi[n,i+1] + 0.5*delta*q[i]/abs(mu_n[n]))
                psi[n,i-1] = 2*psi[n,i] - psi[n,i+1]
    
    for i in range(0, I-2, 2):
        phi[i] = np.dot(w_n, psi[:,i])
        q[i] = 0.5*sigV[i] + S
    
    error = max(abs(phiPrev - phi))
    phiPrev = phi




