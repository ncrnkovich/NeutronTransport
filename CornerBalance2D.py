
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
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib import cm

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

def sweep2D(Nx, Ny, dx, dy, sig_s, sig_t, Q, mu, eta, boundaryX, boundaryY):

    # matrices for corner balance method
    Ax = np.array(((1, 1, 0, 0), (-1, -1, 0, 0), (0, 0, -1, -1), (0, 0, 1, 1)))
    Ay = np.array(((1, 0, 0, 1), (0, 1, 1, 0), (0, -1, -1, 0,), (-1, 0, 0, -1)))
    Lmat = np.zeros((4,4)) # left hand side matrix, combine all into L*u = b
    b = np.zeros((4))
    Lmat = mu/dx*Ax + eta/dy*Ay
    psi = np.zeros((Nx, Ny, 4))

    if (mu > 0) and (eta > 0):
        # add in other matrices as appropriate, i get the pattern but don't get what I'm adding
        Lmat[1,1] += 2*mu/dx 
        Lmat[2,2] += 2*mu/dx + 2*eta/dy
        Lmat[3,3] += 2*eta/dy

        for i in range(Nx):
            for j in range(Ny):
                
                if i == 0:
                    psiLeft1 = boundaryX
                    psiLeft2 = boundaryX
                else:
                    psiLeft1 = psi[i-1,j,1]
                    psiLeft2 = psi[i-1,j,2]
                if j == 0:
                    psiBottom1 = boundaryY
                    psiBottom2 = boundaryY
                else:
                    psiBottom1 = psi[i,j-1,3]
                    psiBottom2 = psi[i,j-1,2]

                b[0] = Q[i,j,0] + 2*mu/dx*psiLeft1 + 2*eta/dy*psiBottom1
                b[1] = Q[i,j,1] + 2*eta/dy*psiBottom2
                b[2] = Q[i,j,2] 
                b[3] = Q[i,j,3] + 2*mu/dx*psiLeft2

                cellLmat = Lmat + np.diag(sig_t[i,j,:])
                psi[i,j,:] = np.linalg.solve(cellLmat, b)        

    elif (mu > 0) and (eta <0):
        Lmat[0,0] += -2*eta/dy
        Lmat[1,1] += 2*mu/dx - 2*eta/dy
        Lmat[2,2] += 2*mu/dx

        for i in range(Nx):
            for j in range(Ny-1, -1, -1):
                
                if i == 0:
                    psiLeft1 = boundaryX
                    psiLeft2 = boundaryX
                else:
                    psiLeft1 = psi[i-1,j,1]
                    psiLeft2 = psi[i-1,j,2]
                if j == Ny-1:
                    psiTop1 = boundaryY
                    psiTop2 = boundaryY
                else:
                    psiTop1 = psi[i,j+1,0]
                    psiTop2 = psi[i,j+1,1]

                b[0] = Q[i,j,0] + 2*mu/dx*psiLeft1
                b[1] = Q[i,j,1] 
                b[2] = Q[i,j,2] - 2*eta/dy*psiTop1
                b[3] = Q[i,j,3] + 2*mu/dx*psiLeft2 - 2*eta/dy*psiTop2

                cellLmat = Lmat + np.diag(sig_t[i,j,:])
                psi[i,j,:] = np.linalg.solve(cellLmat, b)
        

    elif (mu < 0) and (eta > 0):

        Lmat[0,0] += -2*mu/dx 
        Lmat[2,2] += 2*eta/dy
        Lmat[3,3] += -2*mu/dx + 2*eta/dy
        for i in range(Nx-1, -1, -1):
            for j in range(Ny):
                
                if i == Nx-1:
                    psiRight1 = boundaryX
                    psiRight2 = boundaryX
                else:
                    psiRight1 = psi[i+1,j,0]
                    psiRight2 = psi[i+1,j,3]
                if j == 0:
                    psiBottom1 = boundaryY
                    psiBottom2 = boundaryY
                else:
                    psiBottom1 = psi[i,j-1,3]
                    psiBottom2 = psi[i,j-1,2]

                b[0] = Q[i,j,0] + 2*eta/dy*psiBottom1
                b[1] = Q[i,j,1] - 2*mu/dx*psiRight1 + 2*eta/dy*psiBottom2
                b[2] = Q[i,j,2] - 2*mu/dx*psiRight2
                b[3] = Q[i,j,3]

                cellLmat = Lmat + np.diag(sig_t[i,j,:])
                psi[i,j,:] = np.linalg.solve(cellLmat, b)

    elif (mu < 0) and (eta < 0):

        Lmat[0,0] += -2*mu/dx - 2*eta/dy
        Lmat[1,1] += -2*eta/dy
        Lmat[3,3] += -2*mu/dx

        for i in range(Nx-1, -1, -1):
            for j in range(Ny-1, -1, -1):
                
                if i == Nx-1:
                    psiRight1 = boundaryX
                    psiRight2 = boundaryX
                else:
                    psiRight1 = psi[i+1,j,0]
                    psiRight2 = psi[i+1,j,3]
                if j == Ny-1:
                    psiTop1 = boundaryY
                    psiTop2 = boundaryY
                else:
                    psiTop1 = psi[i,j+1,1]
                    psiTop2 = psi[i,j+1,0]

                b[0] = Q[i,j,0]
                b[1] = Q[i,j,1] - 2*mu/dx*psiRight1
                b[2] = Q[i,j,2] - 2*mu/dx*psiRight2 - 2*eta/dy*psiTop1
                b[3] = Q[i,j,3] - 2*eta/dy*psiTop2

                cellLmat = Lmat + np.diag(sig_t[i,j,:])
                psi[i,j,:] = np.linalg.solve(cellLmat, b) 


    return psi

def phiSolver(N, Nx, Ny, dx, dy, sig_s, sig_t, Q, boundaryX, boundaryY):

    w, mu, eta = prod_quad(N)
    phi = np.zeros((Nx,Ny,4)) # phi for each corner of every point in grid

    for n in range(w.size):
        
        psi = sweep2D(Nx, Ny, dx, dy, sig_s, sig_t, Q, mu[n], eta[n], boundaryX[n], boundaryY[n])
        phi += w[n]*psi

    return phi

def convergenceCheck(phi, phiPrev, errTol):

    L2norm = np.max(np.abs(phi - phiPrev))
    if L2norm < errTol:
        converged = True
    else:
        converged = False

    return converged, L2norm

def grid(Nx, Ny, dx, dy, phi):

    x = np.zeros((2*Nx, 2*Ny))
    y = np.zeros((2*Nx, 2*Ny))
    X = np.linspace(dx/2, (Nx*dx) - dx/2, Nx)
    Y = np.linspace(dy/2, (Ny*dy) - dy/2, Ny)
    phi_grid = np.zeros((2*Nx, 2*Ny))

    for j in range(Ny):
        for i in range(Nx):

            phi_grid[2*i, 2*j] = phi[i,j,0]
            phi_grid[2*i+1, 2*j] = phi[i,j,1]
            phi_grid[2*i+1, 2*j+1] = phi[i,j,2]
            phi_grid[2*i, 2*j+1] = phi[i,j,3]

            x[2*i,2*j] = X[i] - dx/4
            x[2*i,2*j+1] = X[i] - dx/4
            x[2*i+1,2*j] = X[i] + dx/4
            x[2*i+1,2*j+1] = X[i] + dx/4

            y[2*i, 2*j] = Y[j] - dy/4
            y[2*i+1, 2*j] = Y[j] - dy/4
            y[2*i, 2*j+1] = Y[j] + dy/4
            y[2*i+1, 2*j+1] = Y[j] + dy/4

    return x,y, phi_grid

def crossSections2D(sig_t, sig_s, S, dx, dy):

    # Fill cross section patterns here
    sig_t += 0.1
    sig_s += 0.01
    # S += 0
    # lower_fact = 0.001

    # Nx, Ny, Nz = sig_t.shape
    # test problem 1
    #put channel in
    # for i in range(Nx):
    #     for j in range(Ny):
    #         ypos = (j-.5)*dy
    #         xpos = (i-.5)*dx
    #         if (ypos <= 2) and (math.fabs(xpos-5) > 1):
    #             sig_t[i,j,:] *= lower_fact
    #             sig_s[i,j,:] *= lower_fact
    #         elif (ypos <= 4) and (math.fabs(xpos-5) > 1) and (math.fabs(xpos-5) < 2):
    #             sig_t[i,j,:] *= lower_fact
    #             sig_s[i,j,:] *= lower_fact
    #         elif (ypos >2) and (ypos <= 4) and (math.fabs(xpos-5) < 1) and (math.fabs(xpos-5) < 2):
    #             sig_t[i,j,:] *= lower_fact
    #             sig_s[i,j,:] *= lower_fact

    # test problem 2
    # for i in range(Nx):
    #     for j in range(Ny):
    #         ypos = (j - 0.5)*dy
    #         xpos = (i-0.5)*dx
    #         if xpos < 2:
    #             sig_t[i,j, :] = 1
    #             sig_s[i,j] = 0
    #             S[i,j,:] = 1
    #         elif xpos < 4 and ypos > 2:
    #             sig_t[i,j,:] = 0.001
    #             sig_s[i,j,:] = 0.001
    #             S[i,j,:] = 0
    #         elif xpos < 6 and ypos > 2:
    #             sig_t[i,j,:] = 1000
    #             sig_s[i,j,:] = 0
    #             S[i,j,:] = 0
    #         elif xpos < 6:
    #             sig_t[i,j,:] = 0.001
    #             sig_s[i,j,:] = 0.001
    #             S[i,j,:] = 0
    #         elif ypos > 2:
    #             sig_t[i,j,:] = 1000
    #             sig_s[i,j,:] = 0
    #             S[i,j,:] = 0
    #         else:
    #             sig_t[i,j,:] = 100
    #             sig_s[i,j,:] = 100
    #             S[i,j,:] = 0

    # test problem 1



    return sig_t, sig_s, S

def latticeCrossSections(sig_t, sig_s, S, dx, dy):

    Nx, Ny, Nz = sig_t.shape

    for i in range(Nx):
        for j in range(Ny):
            xpos = (i - 0.5)*dx
            ypos = (j - 0.5)*dy

            if xpos < 4 and xpos > 3 and ypos < 4 and ypos > 3:
                sig_s[i,j,:] = 1
                sig_t[i,j,:] = 1
                S[i,j,:] = 1
            elif xpos > 1 and xpos < 2:
                if ypos > 1 and ypos < 2:
                    sig_s[i,j,:] = 0
                    sig_t[i,j,:] = 10
                    S[i,j,:] = 0
                elif ypos > 3 and ypos < 4:
                    sig_s[i,j,:] = 0
                    sig_t[i,j,:] = 10
                    S[i,j,:] = 0
                elif ypos > 5 and ypos < 6:
                    sig_s[i,j,:] = 0
                    sig_t[i,j,:] = 10
                    S[i,j,:] = 0
                else:
                    sig_s[i,j,:] = 1
                    sig_t[i,j,:] = 1
                    S[i,j,:] = 0
                    
            elif xpos > 2 and xpos < 3:
                if ypos > 2 and ypos < 3:
                    sig_s[i,j,:] = 0
                    sig_t[i,j,:] = 10
                    S[i,j,:] = 0
                elif ypos > 4 and ypos < 5:
                    sig_s[i,j,:] = 0
                    sig_t[i,j,:] = 10
                    S[i,j,:] = 0
                else:
                    sig_s[i,j,:] = 1
                    sig_t[i,j,:] = 1
                    S[i,j,:] = 0
            elif xpos > 3 and xpos < 4: 
                if ypos > 1 and ypos < 2:
                    sig_s[i,j,:] = 0
                    sig_t[i,j,:] = 10
                    S[i,j,:] = 0
                else:
                    sig_s[i,j,:] = 1
                    sig_t[i,j,:] = 1
                    S[i,j,:] = 0
            elif xpos > 4 and xpos < 5:
                if ypos > 2 and ypos < 3:
                    sig_s[i,j,:] = 0
                    sig_t[i,j,:] = 10
                    S[i,j,:] = 0
                elif ypos > 4 and ypos < 5:
                    sig_s[i,j,:] = 0
                    sig_t[i,j,:] = 10
                    S[i,j,:] = 0
                else:
                    sig_s[i,j,:] = 1
                    sig_t[i,j,:] = 1
                    S[i,j,:] = 0
            elif xpos > 5 and xpos < 6:
                if ypos > 1 and ypos < 2:
                    sig_s[i,j,:] = 0
                    sig_t[i,j,:] = 10
                    S[i,j,:] = 0
                elif ypos > 3 and ypos < 4:
                    sig_s[i,j,:] = 0
                    sig_t[i,j,:] = 10
                    S[i,j,:] = 0
                elif ypos > 5 and ypos < 6:
                    sig_s[i,j,:] = 0
                    sig_t[i,j,:] = 10
                    S[i,j,:] = 0
                else:
                    sig_s[i,j,:] = 1
                    sig_t[i,j,:] = 1
                    S[i,j,:] = 0
            elif xpos > 2 and xpos < 3:
                if ypos > 2 and ypos < 3:
                    sig_s[i,j,:] = 0
                    sig_t[i,j,:] = 10
                    S[i,j,:] = 0
                elif ypos > 4 and ypos < 5:
                    sig_s[i,j,:] = 0
                    sig_t[i,j,:] = 10
                    S[i,j,:] = 0
                else:
                    sig_s[i,j,:] = 1
                    sig_t[i,j,:] = 1
                    S[i,j,:] = 0
            else: 
                sig_s[i,j,:] = 1
                sig_t[i,j,:] = 1
                S[i,j,:] = 0

    return sig_t, sig_s, S



print("Started Program")
# set up grid
a = 7 # x length
b = 7 # y length
N = 8 # number of ordinates
Nx = 70 # Num x points
Ny = 70 # num y points
dx = a/Nx # cell x size
dy = b/Ny # cell y size

# set up cross section/source vectors
sig_t = np.zeros((Nx,Ny,4))
sig_s = np.zeros((Nx,Ny,4))
S = np.zeros((Nx, Ny, 4)) # specified external source
# sig_t, sig_s, S = crossSections2D(sig_t, sig_s, S, dx, dy)
sig_t, sig_s, S = latticeCrossSections(sig_t, sig_s, S, dx, dy)
Q = (sig_t)*0 + S

x,y,sigt = grid(Nx, Ny,dx, dy, sig_t)

plt.pcolormesh(x,y,sigt)
plt.colorbar()
plt.show()

w, mu, eta = prod_quad(N)
# boundary conditions
boundaryX = np.zeros(w.size)
boundaryY = np.zeros(w.size)

# change boundary for each n
boundaryX[mu > 0] = 0
# boundaryX[mu < 0] = 0
# boundaryY[eta > 0] = 100
boundaryY[eta < 0] = 0


converged = False
errTol = 1E-8
phiPrev = np.zeros((Nx,Ny,4))
it = 1

while not converged:
    print("begin sweep")
    phi = phiSolver(N, Nx, Ny, dx, dy, sig_s, sig_t, Q, boundaryX, boundaryY)
    converged, error = convergenceCheck(phi, phiPrev, errTol)
    phiPrev = np.copy(phi)
    Q = sig_s*phiPrev + S

    print("iteration = ", it, "error = ", error)
    it += 1
x, y, phi_grid = grid(Nx, Ny, dx, dy, phi)

# plt.contourf(x,y,phi_grid, 20)




plt.pcolormesh(x, y, phi_grid, norm=colors.LogNorm(vmin=phi_grid.min(), vmax=phi_grid.max()))
plt.colorbar()
# %%


