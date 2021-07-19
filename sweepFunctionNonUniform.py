
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
        q = np.abs(mu*v - u) # uniform neutron vel relative to uniform material vel  

        if mu + u[0]/q[0] > 0:
            i = 0
            psiEdge[n,0] = boundary[n]
            psiEdge[n,-1] = boundary[(N-1)-n]
            while i < I:
                if mu + u[i+1]/q[i+1] > 0:
                    psiCenter[n,i] = (delta*Q[i] + psiEdge[n,i]*(2*mu + u[i+1]/q[i+1] + u[i]/q[i]))/(2*mu + delta*sig_t[i] + 2*u[i+1]/q[i+1])
                    psiEdge[n,i+1] = 2.0*psiCenter[n,i] - psiEdge[n,i]
                    # if n == 4:
                    #     print("i = ", i, "psiCent = %0.2f"%(psiCenter[n,i]), Q[i], sig_t[i], q[i])
                    i += 1
                    

                elif mu + u[i+1]/q[i+1] < 0: # change in direction has occurred
                    j = i
                    while mu + u[j+1]/q[j+1] < 0: # when false, j is center of B-type cell (no in-flux)
                        j += 1
                        if j == I- 1:
                            break

                    psiCenter[n,j] = Q[j]/sig_t[j]
                    # if n == 4:
                    #     print("j = ", j, "psiCent = %0.2f"%(psiCenter[n,j]))
                    psiEdge[n,j] = psiCenter[n,j]
                    psiEdge[n,j+1] = psiCenter[n,j]
                    for k in range(j-1, i, -1):
                        psiCenter[n,k] = (delta*Q[k] - psiEdge[n,k+1]*(2*mu + u[k+1]/q[k+1] + u[k]/q[k]))/(-2*mu + delta*sig_t[k] - 2*u[k]/q[k])
                        psiEdge[n,k] = 2.0*psiCenter[n,k] - psiEdge[n,k+1]
                        # if n == 4:
                        #     print("k = ", k, "psiCent = %0.2f"%(psiCenter[n,k]))

                    psiCenter[n,i] = 0.5*(psiEdge[n,i] + psiEdge[n, i+1])
                    # if n == 4:
                    #     print("i = ", i, "psiCent = %0.2f"%(psiCenter[n,i]))
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
                    # if n == 3:
                    #     print("i = ",i)
                    psiCenter[n,i] = (delta*Q[i] - psiEdge[n,i+1]*(2*mu + u[i+1]/q[i+1] + u[i]/q[i]))/(-2*mu + delta*sig_t[i] - 2*u[i]/q[i])
                    psiEdge[n,i] = 2.0*psiCenter[n,i] - psiEdge[n,i+1]
                    i -= 1
                    
                elif mu + u[i]/q[i] > 0:
                    j = i
                    while mu + u[j]/q[j] > 0:
                        j -= 1
                        # if n == 3:
                        #     print("j = ", j)

                    psiCenter[n,j] = Q[j]/sig_t[j]
                    psiEdge[n,j+1] = psiCenter[n,j] # flux out of cells with no in-flux is isotropic
                    psiEdge[n,j] = psiCenter[n,j] 
                    for k in range(j+1, i, 1):
                        psiCenter[n,k] = (delta*Q[k] + psiEdge[n,k]*(2*mu + u[k+1]/q[k+1] + u[k]/q[k]))/(2*mu + delta*sig_t[k] + 2*u[k+1]/q[k+1])
                        psiEdge[n,k+1] = 2.0*psiCenter[n,k] - psiEdge[n,k]
                        # if n == 4:
                        #     print("k = ", k)
                    psiCenter[n,i] = 0.5*(psiEdge[n,i] + psiEdge[n, i+1])
                    i = j - 1
        else:
            print("error: cant start sweep from left or right. mu = %.2f"%(mu), "mu + u/q = %.3f"%(mu + u[0]/q[0]), "mu + u[I]/q[I] = %.3f"%(mu + u[-1]/q[-1]))
    return psiCenter, psiEdge
