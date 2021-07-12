#%% 
# movingMaterialjacobi
#  import libraries
from os import write
import numpy as np
import math as math
import matplotlib.pyplot as plt
from numpy.core.numeric import Inf
import scipy as scipy
from scipy.constants import constants
import scipy.special
# import sweepFunction
from sweepFunction import sweep
# import crossSections
from crossSections import crossSections
from crossSections import reedsProblem
from sweepFunction import sweepMotion



def sweepMotion(psiCenter, psiEdge, psiCenterPrev, psiEdgePrev, u, v, a, sig_t, Q, boundary):

    ## a = total thickness
    ## I = number of points
    ## N = number of discrete ordinates
    ## sig_t = total cross section vector
    ## sig_s = scattering cross section vector
    ## S = source
    ## psiEdgeL == left boundary condition 
    ## psiEdgeR == right boundary condition

    # set up grid
    N, I = psiCenter.shape
    x = np.linspace(0, a, I)
    delta = x[1] - x[0]

    # temporary for constant flow
    u = np.average(u)
    q = v - u # uniform neutron vel relative to uniform material vel   

    # P_N Quadrature for order N w_n normalized to 1
    mu_n, w_n = scipy.special.roots_legendre(N)
    w_n = w_n/np.sum(w_n)

    for n in range(len(mu_n)):

        Xu = u/(np.abs(mu_n[n])*np.abs(q))

        if v*mu_n[n] + u > 0:

            psiEdge[n,0] = boundary[n]
            for i in range(I):

                uL = u # uL = u[i]
                uR = u# uR = u[i+1]

                psiCenter[n,i] = (Q[i] + psiEdge[n,i]*(2.0*np.abs(mu_n[n])/delta + uL/(q*delta) + uR/(q*delta)))/(sig_t[i] + 2.0*mu_n[n]/delta + 2*uR/(q*delta))
                psiEdge[n,i+1] = 2.0*psiCenter[n,i] - psiEdge[n,i]
                
        else:
            
            psiEdge[n,-1] = boundary[n]
            for i in range(I-1, -1, -1):
                uL = u # uL = u[i]
                uR = u# uR = u[i+1]
                psiCenter[n,i] = (Q[i] + (2.0*np.abs(mu_n[n])/delta - uR/(q*delta) - uL/(q*delta))*psiEdge[n,i+1])/(sig_t[i] + 2.0*np.abs(mu_n[n])/delta -2*uR/(q*delta)) 
                psiEdge[n,i] = 2.0*psiCenter[n,i] - psiEdge[n,i+1]

                ## reflective boundary at x = 0
                # psiEdge[(N-1-n),0] = psiEdge[n,0]
                    
    return psiCenter, psiEdge

def jacobiSolverNoMotion(psiCenter, psiCenterPrev, a, sig_t, sig_s, S, boundary):
    
    # set up grid
    N, I = psiCenter.shape
    x = np.linspace(0, a, I)
    delta = x[1] - x[0]

    # P_N Quadrature for order N w_n normalized to 1
    mu_n, w_n = scipy.special.roots_legendre(N)
    w_n = w_n/np.sum(w_n)

    for i in range(I):

        for n in range(N):
            mu = mu_n[n]
            phiSum = 0

            for k in range(N): 
                if k != n:
                    phiSum += w_n[k]*psiCenter[k,i] # use of psiCenter here effectively makes this Gauss-Seidel method

            if mu > 0:
                if i == 0:
                    psiCenter[n,i] = (S[i] + sig_s[i]*phiSum + np.abs(mu)/delta*boundary[n])/(np.abs(mu)/delta + sig_t[i] - sig_s[i]*w_n[n])
                else:
                    psiCenter[n,i] = (S[i] + sig_s[i]*phiSum + np.abs(mu)/delta*psiCenterPrev[n,i-1])/(np.abs(mu)/delta + sig_t[i] - sig_s[i]*w_n[n])
            else:

                if i == I - 1:
                    psiCenter[n,i] = (S[i] + sig_s[i]*phiSum + np.abs(mu)/delta*boundary[n])/(np.abs(mu)/delta + sig_t[i] - sig_s[i]*w_n[n])
                else:
                    psiCenter[n,i] = (S[i] + sig_s[i]*phiSum + np.abs(mu)/delta*psiCenterPrev[n,i+1])/(np.abs(mu)/delta + sig_t[i] - sig_s[i]*w_n[n])

                    
    return psiCenter

def jacobiSolverMotion(psiCenter, psiCenterPrev, u, v, a, sig_t, sig_s, S, boundary):
    
    # set up grid
    N, I = psiCenter.shape
    x = np.linspace(0, a, I)
    delta = x[1] - x[0]

    # temporary for constant flow
    u = np.average(u)
    q = v - u # uniform neutron vel relative to uniform material vel   

    # P_N Quadrature for order N w_n normalized to 1
    mu_n, w_n = scipy.special.roots_legendre(N)
    w_n = w_n/np.sum(w_n)

    for i in range(I):

        for n in range(N):
            mu = mu_n[n]
            gamma = mu + u/q
            phiSum = 0

            for k in range(N): 
                if k != n:
                    phiSum += w_n[k]*psiCenter[k,i] # use of psiCenter here effectively makes this Gauss-Seidel method

            if mu > 0:
                if i == 0:
                    psiCenter[n,i] = (S[i] + sig_s[i]*phiSum +gamma/delta*boundary[n])/(gamma/delta + sig_t[i] - sig_s[i]*w_n[n])
                else:
                    psiCenter[n,i] = (S[i] + sig_s[i]*phiSum + gamma/delta*psiCenterPrev[n,i-1])/(gamma/delta + sig_t[i] - sig_s[i]*w_n[n])
            else:

                if i == I - 1:
                    psiCenter[n,i] = (S[i] + sig_s[i]*phiSum - gamma/delta*boundary[n])/(-gamma/delta + sig_t[i] - sig_s[i]*w_n[n])
                else:
                    psiCenter[n,i] = (S[i] + sig_s[i]*phiSum - gamma/delta*psiCenterPrev[n,i+1])/(-gamma/delta + sig_t[i] - sig_s[i]*w_n[n])

                    
    return psiCenter


def jacobiSolverNonUniform(psiCenter, psiCenterPrev, u, v, a, sig_t, sig_s, S, boundary):
     
    # set up grid
    N, I = psiCenter.shape
    x = np.linspace(0, a, I)
    delta = x[1] - x[0]


    # P_N Quadrature for order N w_n normalized to 1
    mu_n, w_n = scipy.special.roots_legendre(N)
    w_n = w_n/np.sum(w_n)

    for i in range(I):

        for n in range(N):
            mu = mu_n[n]
            phiSum = 0
            q = np.abs(mu*v - u) # uniform neutron vel relative to uniform material vel   

            for k in range(N): 
                if k != n:
                    phiSum += w_n[k]*psiCenter[k,i] # use of psiCenter here effectively makes this Gauss-Seidel method
            if mu*v + u[i]/q[i] > 0:
                if i == 0:
                    psiCenter[n,i] = (S[i] + sig_s[i]*phiSum + boundary[n]*(u[i]/(delta*q[i]) + mu/delta))/(mu/delta + u[i+1]/(q[i+1]*delta) + sig_t[i] - sig_s[i]*w_n[n])
                else:
                    psiCenter[n,i] = (S[i] + sig_s[i]*phiSum + psiCenterPrev[n,i-1]*(u[i]/(delta*q[i]) + mu/delta))/(mu/delta + u[i+1]/(q[i+1]*delta) + sig_t[i] - sig_s[i]*w_n[n])
                    # psiCenter[n,i] = (S[i] + sig_s[i]*phiSum + psiCenterPrev[n,i-1]*(u[i]/(delta*q[i]) + mu/delta))/(mu/delta + u[i+1]/(q[i+1]*delta) + sig_t[i] - sig_s[i]*w_n[n])
            else:
                
                if i == I - 1:
                    psiCenter[n,i] = (S[i] + sig_s[i]*phiSum - boundary[n]/delta*(u[i+1]/q[i+1] + mu))/(-mu/delta - u[i]/(delta*q[i]) + sig_t[i] - sig_s[i]*w_n[n])
                else:
                    
                    psiCenter[n,i] = (S[i] + sig_s[i]*phiSum - psiCenterPrev[n,i+1]/delta*(u[i+1]/q[i+1] + mu))/(-mu/delta - u[i]/(delta*q[i]) + sig_t[i] - sig_s[i]*w_n[n])
                    a = (S[i] + sig_s[i]*phiSum - psiCenterPrev[n,i+1]/delta*(u[i+1]/q[i+1] - mu))
                    # print("Num = ", a)/

    return psiCenter

def phiSolver(psi, w):

    N, I = psi.shape
    phi = np.zeros((I))
    for n in range(N):
        phi += w[n]*psi[n,:]
    
    return phi
    
def fill(sig_t, sig_s, S):  

    # place to write any code to fill cross sections/external source vectors
    sig_t += 1
    sig_s += 0.1
    S += 1

    return sig_t, sig_s, S

def materialVel(I, dx):

    u = np.zeros(I+1)
    # u += 0

    for i in range(u.size):
        xpos = dx*(i- 0.5)
        if xpos > 4:
            u[i] = -20
        else:
            u[i] = 20

    return u


# random constants
# Mass of neutron: 1.675E-27 kg
# 1 eV neutron => 13.83 km/s = 13.83E5 cm/s
# 1 MeV neutron => 13830 km/s
# 1 eV = 1.602E-19 J

# set grid parameters
a = 8
I = 200
x = np.linspace(0, a, I)
dx = x[1] - x[0]
# specify discrete ordinates
N = 8
u = materialVel(I,dx)
v = 100

# cross sections
sig_t = np.zeros(I) # total cross section
sig_s = np.zeros(I) # scattering cross section
S = np.zeros(I) # external source
sig_t, sig_s, S = fill(sig_t, sig_s, S)
# alpha = 1
# sig_t, sig_s, S = reedsProblem(x, 1, sig_t, sig_s, S)

# preallocate angular flux vectors and scalar flux and set boundary conditions
psiCenter = np.zeros((N,I))
psiCenterPrev = np.zeros((N,I))
psiEdge = np.zeros((N,I+1))
psiEdgePrev = np.zeros((N, I+1))
phiPrev = np.zeros(I)
Q = np.zeros(I)+ sig_s[0]*phiPrev[0] + S[0]

# boundary conditions
mu, w = scipy.special.roots_legendre(N)
w = w/np.sum(w)
boundary = np.zeros(N)
boundary[mu > 0] = 0
boundary[mu < 0] = 0
errorPrev = 100
error = 10
errTol = 1E-7
it = 1
while error > errTol:

    # psiCenter = jacobiSolverNoMotion(psiCenter, psiCenterPrev, a, sig_t, sig_s, S, boundary)
    # psiCenter, psiEdge = sweepMotion(psiCenter, psiEdge, psiCenterPrev, psiEdgePrev, u, v, a, sig_t, Q, boundary)
    psiCenter = jacobiSolverNonUniform(psiCenter, psiCenterPrev, u, v, a, sig_t, sig_s, S, boundary)
    # psiCenter = jacobiSolverMotion(psiCenter, psiCenterPrev, u, v, a, sig_t, sig_s, S, boundary)
    phi = phiSolver(psiCenter, w)
    # Q = sig_s*phiPrev + S # iterate on source
    error = np.linalg.norm(phiPrev - phi)

    # copy values for next iteration
    phiPrev = phi.copy()
    psiCenterPrev = psiCenter.copy()
    # psiEdgePrev = psiEdge.copy()

    print("Iteration = ", it, "error = ", error)
    
    
    it += 1

    if error > 1000:
        break
    # elif error > 2*errorPrev:
    #     break
    else:
        errorPrev = error.copy()

    




plt.figure(1)
for i in range(N):
    # if i != 3:
    plt.plot(x, psiCenter[i,:],"--",label="mu = %.2f"%(mu[i]))
    plt.legend()
    
# plt.legend()
plt.figure(2)
plt.plot(x,phi, label="Num")
plt.legend()
plt.xlabel("x")
plt.ylabel("Phi")
plt.show

#%%
# %%
