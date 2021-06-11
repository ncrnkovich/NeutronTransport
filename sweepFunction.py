

# import libraries
from os import write
import numpy as np
import math as math
import matplotlib.pyplot as plt
from numpy.core.numeric import Inf
import scipy as scipy
from scipy.constants import constants
import scipy.special


def sweep(a, I, N, sig_t, sig_s, S, psiEdgeL, psiEdgeR):

    ## a = total thickness
    ## I = number of points
    ## N = number of discrete ordinates
    ## sig_t = total cross section vector
    ## sig_s = scattering cross section vector
    ## S = source
    ## psiEdgeL == left boundary condition 
    ## psiEdgeR == right boundary condition

    # set up grid
    x = np.linspace(0, a, I)
    delta = x[1] - x[0]

    # preallocate angular flux vectors and scalar flux and set boundary conditions
    psiCenter = np.zeros((N,I))
    psiEdge = np.zeros((N,I+1))
    psiEdge[:,0] = psiEdgeL
    psiEdge[:,-1] = psiEdgeR
    phi = np.zeros(I)
    phiPrev = np.zeros(len(phi)) 
    phi_0 = 0 # initial guess for phi

    # set error tolerances and q vector
    error = 10 # initial error so while loop is true
    err = 1E-7
    q = np.zeros(I)+ sig_s*phi_0 + S[0]


    # P_N quadrature for N = 8 w_n normalized to 1
    mu_n, w_n = scipy.special.roots_legendre(N)
    w_n = w_n/np.sum(w_n)

    it = 1
    while error > err:
        for n in range(len(mu_n)):

            if mu_n[n] > 0:

                for i in range(I):                
                    psiCenter[n,i] = (1 + 0.5*sig_t[i]*delta/abs(mu_n[n]))**(-1)*(psiEdge[n,i] + 0.5*delta*q[i]/abs(mu_n[n]))
                    psiEdge[n,i+1] = 2*psiCenter[n,i] - psiEdge[n,i]
                
            else:

                for i in range(I-1, -1, -1):
                    psiCenter[n,i] = (1 + 0.5*sig_t[i]*delta/abs(mu_n[n]))**(-1)*(psiEdge[n,i+1] + 0.5*delta*q[i]/abs(mu_n[n]))                    
                    psiEdge[n,i] = 2*psiCenter[n,i] - psiEdge[n,i+1]

                # reflective boundary at x = 0
                psiEdge[(N-1-n),0] = psiEdge[n,0]
                
        for i in range(I):
            phi[i] = np.dot(w_n, psiCenter[:,i])
            q[i] = sig_s[i]*phi[i] + S[i]
            
        
        # error = max(abs(phiPrev - phi)) # RMSerror = np.norm(phiPrev - phi)
        error = np.linalg.norm(phiPrev - phi)
        phiPrev = phi.copy()

        print("Iteration = ", it, "error = ", error)
        it += 1

    return psiCenter, phi
    