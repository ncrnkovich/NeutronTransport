#%%

# nonuniform sweep
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
# from sweepFunction import sweepMotion



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

    # P_N Quadrature for order N w_n normalized to 1
    mu_n, w_n = scipy.special.roots_legendre(N)
    w_n = w_n/np.sum(w_n)

    for n in range(mu_n.size):
        mu = mu_n[n]
        # q = np.abs(mu*v - u) # uniform neutron vel relative to uniform material vel  

        if mu + u[0]/q[0] > 0:
            i = 0
            psiEdge[n,0] = boundary[n]
            psiEdge[n,-1] = boundary[(N-1)-n]
            while i < I:
                if mu + u[i+1]/q[i+1] > 0:
                    psiCenter[n,i] = (delta*Q[i] + psiEdge[n,i]*(2*mu + u[i+1]/q[i+1] + u[i]/q[i]))/(2*mu + delta*sig_t[i] + 2*u[i+1]/q[i+1])
                    psiEdge[n,i+1] = 2.0*psiCenter[n,i] - psiEdge[n,i]
                    i += 1
                    

                elif mu + u[i+1]/q[i+1] < 0: # change in direction has occurred
                    j = i
                    while mu + u[j+1]/q[j+1] < 0: # when false, j is center of B-type cell (no in-flux)
                        j += 1
                        if j == I- 1:
                            break
                    
                    if j == I - 1:
                        psiCenter[n,j] = (delta*Q[j] - psiEdge[n,j+1]*(2*mu + u[j+1]/q[j+1] + u[j]/q[j]))/(-2*mu + delta*sig_t[j] - 2*u[j]/q[j])
                        psiEdge[n,j] = 2.0*psiCenter[n,j] - psiEdge[n,j+1]
                    else:
                        psiCenter[n,j] = Q[j]/sig_t[j]
                        psiEdge[n,j] = psiCenter[n,j]
                        psiEdge[n,j+1] = psiCenter[n,j]

                    for k in range(j-1, i, -1):
                        psiCenter[n,k] = (delta*Q[k] - psiEdge[n,k+1]*(2*mu + u[k+1]/q[k+1] + u[k]/q[k]))/(-2*mu + delta*sig_t[k] - 2*u[k]/q[k])
                        psiEdge[n,k] = 2.0*psiCenter[n,k] - psiEdge[n,k+1]

                    psiCenter[n,i] = 0.5*(psiEdge[n,i] + psiEdge[n, i+1])
                    i = j + 1

                else:
                    print("error: mu*v + u[i] = 0")

        elif mu + u[-1]/q[-1] < 0:
            i = I - 1
            psiEdge[n,-1] = boundary[n]
            psiEdge[n, 0] = boundary[(N-1)-n]
            while i > -1:
                # print("n = ", n, mu*v + u[i])
                if mu + u[i]/q[i] < 0:
                    psiCenter[n,i] = (delta*Q[i] - psiEdge[n,i+1]*(2*mu + u[i+1]/q[i+1] + u[i]/q[i]))/(-2*mu + delta*sig_t[i] - 2*u[i]/q[i])
                    psiEdge[n,i] = 2.0*psiCenter[n,i] - psiEdge[n,i+1]
                    i -= 1
                    
                elif mu + u[i]/q[i] > 0:
                    j = i
                    while mu + u[j]/q[j] > 0:
                        j -= 1
                    
                    if j == 0:
                        psiCenter[n,j] = (delta*Q[j] + psiEdge[n,j]*(2*mu + u[j+1]/q[j+1] + u[j]/q[j]))/(2*mu + delta*sig_t[j] + 2*u[j+1]/q[j+1])
                        psiEdge[n,j+1] = 2.0*psiCenter[n,j] - psiEdge[n,j]
                    else:
                        psiCenter[n,j] = Q[j]/sig_t[j]
                        psiEdge[n,j+1] = psiCenter[n,j] # flux out of cells with no in-flux is isotropic
                        psiEdge[n,j] = psiCenter[n,j] 

                    for k in range(j+1, i, 1):
                        psiCenter[n,k] = (delta*Q[k] + psiEdge[n,k]*(2*mu + u[k+1]/q[k+1] + u[k]/q[k]))/(2*mu + delta*sig_t[k] + 2*u[k+1]/q[k+1])
                        psiEdge[n,k+1] = 2.0*psiCenter[n,k] - psiEdge[n,k]
                    psiCenter[n,i] = 0.5*(psiEdge[n,i] + psiEdge[n, i+1])
                    i = j - 1
        else:
            print("error: cant start sweep from left or right. mu = %.2f"%(mu), "mu + u/q = %.3f"%(mu + u[0]/q[0]), "mu + u[I]/q[I] = %.3f"%(mu + u[-1]/q[-1]))
    return psiCenter, psiEdge

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

def materialVel(I,dx):

    u = np.zeros(I+1)
    # u += 100
    # for i in range(u.size):
    #     xpos = dx*(i - 0.5)
    #     if xpos > 4 and xpos < 6:
    #         u[i] = -30
    #     else:
    #         u[i] = 10
    for i in range(u.size):
        xpos = dx*(i- 0.5)
        if xpos > 4:
            u[i] = -60
        # elif xpos < 2:
        #     u[i] = 50
        else:
            u[i] = 60

    return u


# random constants
# Mass of neutron: 1.675E-27 kg
# 1 eV neutron => 13.83 km/s = 13.83E5 cm/s
# 1 MeV neutron => 13830 km/s
# 1 eV = 1.602E-19 J

# set grid parameters
a = 8
I = 100
x = np.linspace(0, a, I)
dx = x[1] - x[0]
# specify discrete ordinates
N = 8
u = materialVel(I,dx)
q = np.zeros(I+1) + 100

# cross sections
sig_t = np.zeros(I) # total cross section
sig_s = np.zeros(I) # scattering cross section
S = np.zeros(I) # external source
sig_t, sig_s, S = fill(sig_t, sig_s, S)
# sig_t, sig_s, S = reedsProblem(x, 1, sig_t, sig_s, S)

# preallocate angular flux vectors and scalar flux and set boundary conditions
psiCenter = np.zeros((N,I))
psiCenterPrev = np.zeros((N,I))
psiEdge = np.zeros((N,I+1))
psiEdgePrev = np.zeros((N, I+1))
phiPrev = np.zeros(I)
Q = np.zeros(I)+ sig_s*phiPrev[0] + S
# boundary conditions
mu, w = scipy.special.roots_legendre(N)
w = w/np.sum(w)
boundary = np.zeros(N)
boundary[mu > 0] = 0
boundary[mu < 0] = 0

error = 10
errTol = 1E-8
it = 1
while error > errTol:
    psiCenter, psiEdge = sweepMotion(psiCenter, psiEdge, psiCenterPrev, psiEdgePrev, u, q, a, sig_t, Q, boundary)
    phi = phiSolver(psiCenter, w)
    Q = sig_s*phi + S # iterate on source
    error = np.linalg.norm(phiPrev - phi)

    # copy values for next iteration
    phiPrev = phi.copy()
    psiCenterPrev = psiCenter.copy()
    psiEdgePrev = psiEdge.copy()

    print("Iteration = ", it, "error = ", error)
    it += 1

    if error > 100000:
        break
    elif it > 1:
        break

    
plt.figure(1)
plt.plot(x,u[0:(I)])
plt.title("u")
plt.grid(True)

plt.figure(2)
for n in range(N):
    plt.plot(x, mu[n] + u[0:I]/q[0:I],label="mu = %.2f"%(mu[n]))
    plt.legend()
    plt.title("mu + u/q")
plt.grid(True)

plt.figure(3)
for n in range(N):
    plt.plot(x, psiCenter[n,:],"--", label="mu = %.2f"%(mu[n]))
    plt.legend()
    plt.title("Psi")
plt.grid(True)

# plt.legend()
plt.figure(4)
plt.plot(x,phi, label="Num")
plt.legend()
plt.xlabel("x")
plt.ylabel("Phi")
plt.show


# %%
