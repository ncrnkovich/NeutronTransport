#%% 
# nonuniform sweep function
#  import libraries
from os import write
import numpy as np
import math as math
import matplotlib.pyplot as plt
from numpy.core.numeric import Inf
import scipy as scipy
from scipy.constants import constants
import scipy.special
# import crossSections
from crossSections import reedsProblem
# from sweepFunctionNonUniform import sweepMotion


def motion1D(a, I, N, Q_f, q):

    x = np.linspace(0,a,I)
    dx = x[1] - x[0]
    u = materialVel(I,dx, a)    
    # preallocate angular flux vectors and scalar flux and set boundary conditions
    psiCenter = np.zeros((N,I))
    psiEdge = np.zeros((N,I+1))
    phiPrev = np.zeros(I)
    psiEdgePrev = np.zeros((N,I+1))
    psiCenterPrev = np.zeros((N,I))
    
    # fill cross section and source vectors
    sig_t, sig_s, sig_f, S = fill(I,dx)
    mu, w = scipy.special.roots_legendre(N)
    # w = w/np.sum(w)
    boundary = np.zeros(N)
    Q = np.zeros(I) + 0.5*sig_s*phiPrev[0] + Q_f
    error = 10
    errTol = 1E-8
    it = 1
    while error > errTol:
        psiCenter, psiEdge = sweepMotion(psiCenter, psiEdge, psiCenterPrev, psiEdgePrev, u, q, a, sig_t, Q, boundary)
        phi = phiSolver(psiCenter, w)
        Q = 0.5*sig_s*phi + Q_f # iterate on source
        error = np.linalg.norm(phiPrev - phi)

        # copy values for next iteration
        phiPrev = phi.copy()

        # print("Iteration = ", it, "error = ", error)
        it += 1

        if error > 100000:
            break
        # elif it > 1:
        #     break

    return phi, psiCenter

def sweepMotion(psiCenter, psiEdge, psiCenterPrev, psiEdgePrev, u, q, a, sig_t, Q, boundary):

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
    
def fill(I,dx):  

    # place to write any code to fill cross sections/external source vectors
    sig_t = np.zeros(I) # total cross section
    sig_s = np.zeros(I) # scattering cross section
    sig_f = np.zeros(I)
    S = np.zeros(I)

    # Pu - 239
    # sig_t += 0.32640
    # sig_s += 0.225216
    # sig_f += 0.081600
    # Ur-235
    sig_t += 0.32640
    sig_s += 0.248064
    sig_f += 0.065280

    # U - D20
    # sig_t += 0.54628
    # sig_s += 0.464338
    # sig_f += 0.054628
    S += 0
    Fes = 0.23209488
    Fet = 0.23256
    U235f = 0.06922744
    U235s = 0.328042
    U235t = 0.407407
    Nas = 0.086368032
    Nat = 0.086368032
    # multimaterial problem
    # for i in range(sig_t.size):
    #     xpos = dx*(i-0.5)
    #     if xpos < 0.317337461:
    #         sig_s[i] = Fes
    #         sig_t[i] = Fet
    #         sig_f[i] = 0
    #     elif xpos < 5.437057544:
    #         sig_s[i] = U235s
    #         sig_t[i] = U235t
    #         sig_f[i] = U235f
    #     elif xpos < 5.754395005:
    #         sig_s[i] = Fes
    #         sig_t[i] = Fet
    #         sig_f[i] = 0
    #     else:
    #         sig_s[i] = Nas
    #         sig_t[i] = Nat
    #         sig_f[i] = 0

    return sig_t, sig_s, sig_f, S

def materialVel(I,dx, a):

    u = np.zeros(I+1)
    u += 0.95
    # for i in range(u.size):
    #     xpos = dx*(i-0.5)
    #     if xpos/a > 0.5:
    #         u[i] = -0.3
    #     else:
    #         u[i] = 0.3
    # for i in range(u.size):
    #     xpos = dx*(i - 0.5)
    #     if xpos > 4 and xpos < 6:
    #         u[i] = -30
    #     else:
    #         u[i] = 10
    # for i in range(u.size):
    #     xpos = dx*(i- 0.5)
    #     if xpos/a > 0.75:
    #         u[i] = 0.3
    #     elif xpos/a < 0.25:
    #         u[i] = 0.3
    #     else:
    #         u[i] = -0.3

    return u


# random constants
# Mass of neutron: 1.675E-27 kg
# 1 eV neutron => 13.83 km/s = 13.83E5 cm/s
# 1 MeV neutron => 13830 km/s
# 1 eV = 1.602E-19 J

# a = 2*1.853722
# a = 2*2.256751
a = 2*2.872934
# a = 2*10.371065
# a = 7.757166007
I = 300
N = 10
q = np.zeros(I+1) + 1
nu = 2.7
x = np.linspace(0, a, I)
dx = x[1] - x[0]

sig_t, sig_s, sig_f, S = fill(I,dx)
phi0 = np.zeros(I) + 3
phi0 = phi0/np.linalg.norm(phi0) # do whatever to normalize phi0 to 1
k = 0.8
kprev = 0
Q_f = nu*0.5*sig_f*phi0
errTol = 1E-8
error = 10
it = 1

while error > errTol:

    phi, psi = motion1D(a, I, N, Q_f, q)
    k = np.linalg.norm(phi)
    phi = phi/k
    Q_f = 0.5*nu*sig_f*phi
    error = np.linalg.norm(k - kprev)
    kprev = k.copy()

    print("k iteration = ", it, "k = %0.7f"%(k))
    it += 1


plt.plot(x,phi)



#%%