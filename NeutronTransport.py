#%% 

# Master script for determining solution to 1D slab neutron transport

from os import write
import numpy as np
import math as math
import matplotlib.pyplot as plt
from numpy.core.numeric import Inf
import scipy as scipy
from scipy.constants import constants
import scipy.special
# set up grid 
a = 5
I = 200
N = 8
x = np.linspace(0, a, I)
delta = x[1] - x[0]

# preallocate angular flux vectors and scalar flux and set boundary conditions
psiCenter = np.zeros((N,I))
psiEdge = np.zeros((N,I+1))
phi = np.zeros(I)
phiPrev = np.zeros(len(phi)) 
phi_0 = 0 # initial guess for phi



# Set up piecewise cross section vector 
    # sig_t = np.zeros(I)+ 2.74E-24*6.02E23
sig_t = np.zeros(I)+ 1 # total cross section
sig_s = np.zeros(I) + 0 # scattering cross section

# set error tolerances and q vector
error = 10 # initial error so while loop is true
err = 1E-10
S = 0 # specified source term 
q = np.zeros(I)+ 0.5*sig_s*phi_0 + S


# P_N quadrature for N = 8 sum of w_n = 2
# mu_n = np.array([-0.9602898564,-0.7966664774,-0.5255324099,-0.18343464240,0.1834346424,0.52553240990,0.7966664774,0.9602898564])
# w_n = np.array([-0.1012285363,-0.2223810344,-0.3137066459,-0.3626837834,0.3626837834,0.3137066459,0.2223810344,0.1012285363])
mu_n, w_n = scipy.special.roots_legendre(N)
# w_n = w_n/np.sum(w_n)
rightSweep = 1

while error > err:
    for n in range(len(mu_n)):

        if mu_n[n] > 0:
            if rightSweep:
                psiEdge[n, 0] = 1 #left boundary condtion for --> sweep
            else:
                psiEdge[n, 0] = 0 #left boundary condtion for <-- sweep

            for i in range(I):                
                psiCenter[n,i] = (1 + 0.5*sig_t[i]*delta/abs(mu_n[n]))**(-1)*(psiEdge[n,i] + 0.5*delta*q[i]/abs(mu_n[n]))
                psiEdge[n,i+1] = 2*psiCenter[n,i] - psiEdge[n,i]
                
            
        else:
            if rightSweep:
                psiEdge[n,-1] = 0 # right boundary condition for --> sweep
            else:
                psiEdge[n,-1] = 1 # right boundary condition for <-- sweep

            for i in range(I-1, -1, -1):
                psiCenter[n,i] = (1 + 0.5*sig_t[i]*delta/abs(mu_n[n]))**(-1)*(psiEdge[n,i+1] + 0.5*delta*q[i]/abs(mu_n[n]))                    
                psiEdge[n,i] = 2*psiCenter[n,i] - psiEdge[n,i+1]
    
    for i in range(I):
        phi[i] = np.dot(w_n, psiCenter[:,i])
        q[i] = sig_s[i]*phi[i] + S
        
    
    # error = max(abs(phiPrev - phi)) # RMSerror = np.norm(phiPrev - phi)
    error = np.linalg.norm(phiPrev - phi)
    phiPrev = phi.copy()

psiExact = np.zeros((N, I))
phiExact = np.zeros(I)

xRL = a - np.linspace(0,a,I)
for n in range(N):
    for i in range(I):
        if mu_n[n] < 0:
            psiExact[n,i] = math.exp(-xRL[i]/abs(mu_n[n]))
        else:
            psiExact[n,i] = math.exp(-x[i]/abs(mu_n[n]))

if rightSweep:
    phiExact = scipy.special.expn(2, x)
else:
    phiExact = scipy.special.expn(2, xRL)

plt.figure(1)
for i in range(N):
    if mu_n[i] > 0:
        # plt.plot(x, psiExact[i,:], label="Exact %f"%(mu_n[i]))
        plt.plot(x, psiCenter[i,:],"--")
    
plt.legend()


plt.figure(2)
plt.plot(x, phiExact, label="Exact")
plt.plot(x,phi, label="Num")
plt.legend()
plt.xlabel("x")
plt.ylabel("Phi")
plt.show


# %%


