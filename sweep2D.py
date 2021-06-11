# %%

# import libraries
from ast import NodeTransformer
import numpy as np
import math as math
import matplotlib.pyplot as plt
from numpy.core.numeric import Inf
import scipy as scipy
from scipy.constants import constants
import scipy.special

def prod_quad(N):
    """Compute ordinates and weights for product quadrature
    Inputs:
        N:               Order of Legendre or Chebyshev quad
    Outputs:
        w:               weights
        eta,xi,mu:       direction cosines (x,y,z)
    """
    assert (N % 2 == 0)
    #get legendre quad
    MUL, WL = np.polynomial.legendre.leggauss(N)
    #print(MUL)
    #get chebyshev y's
    Y, WC = np.polynomial.chebyshev.chebgauss(N)
    #get all pairs
    place = 0
    eta = np.zeros(N*N*2)
    xi = np.zeros(N*N*2)
    mu = np.zeros(N*N*2)
    w = np.zeros(N*N*2)
    
    for i in range(N):
        for j in range(N):
            mul = MUL[i]
            y = Y[j]
            mu[place] = mul
            mu[place+1] = mul
            gamma = np.arccos(y)
            gamma2 = -gamma
            sinTheta = np.sqrt(1-mul*mul)
            eta[place] =   sinTheta*np.cos(gamma)
            eta[place+1] = sinTheta*np.cos(gamma2)
            xi[place] =   sinTheta*np.sin(gamma)
            xi[place+1] = sinTheta*np.sin(gamma2)
            w[place] = WL[i]*WC[j]
            w[place+1] = WL[i]*WC[j]
            place += 2
    return w[mu>0]/np.sum(w[mu>0]), eta[mu>0],xi[mu>0]


# set up grid
a = 1 # x wdith
b = 1 # y width
N = 2
I = 20 # number of points in x-dir
J = 20 # number of points in y-dir
x = np.linspace(0, a, I)
y = np.linspace(0, b, J)

dX = x[1] - x[0]
dY = y[1] - y[0]

psiCenter = np.zeros((I,J,math.ceil(N*(N+2)/2)))
psiEdge = np.zeros((2*I+1,2*J+1,math.ceil(N*(N+2)/2)))


phi = np.zeros((I,J))
phiPrev = np.zeros(phi.size)
phi_0 = 0 # phi initial guess

sig_t = np.zeros((I,J)) + 1 # total cross section
sig_s = np.zeros((I,J)) + 0 # scattering cross section
S = np.zeros((I,J))
q = np.zeros((I,J)) 
q[0,0] = sig_s[0,0]*phi_0 + S[0,0]

error = 10
err = 1E-8
w_n, mu_n, eta_n = prod_quad(N)

while error > err:
    for n in range(math.ceil(N*(N+2)/2)):

        ## for mu_n > 0 and eta_n > 0 
        psiEdge[0,:,:] = 0 # left side BC, x = 1/2, all y
        psiEdge[:,0,:] = 0 # bottom BC, all x, y = 1/2

        for j in range(J):
            for i in range(I):
                psiCenter[i,j,n] = (sig_t[i,j] + 2*abs(mu_n[n])/dX + 2*abs(eta_n[n])/dY)**(-1)*(2*abs(mu_n[n])/dX*psiEdge[i,j+1,n] + 2*abs(eta_n[n])/dY*psiEdge[i+1,j,n] + q[i,j])
                psiEdge[i+2,j+1,n] = 2*psiCenter[i,j,n] - psiEdge[i, j+1, n]

    for j in range(J):
        for i in range(I):
            for n in range(math.ceil(N*(N+2)/2)-1):
                phi[i,j] += 0.25*w_n[n]*psiCenter[i,j,n]
            q[i,j] = sig_s[i,j]*phi[i,j] + S[i,j]






