#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 09:27:21 2018

@author: ryanmcclarren
"""
# %%
import numpy as np
import math
import matplotlib.pyplot as plt
def sweep2D_scb(Nx,Ny,hx,hy,etax,etay,sigmat,Q,boundaryx, boundaryy):
    #q will have dimensions Nx,Ny,4
    #sigmat will have the same
    #in cell unknown layout
    # ----------
    # | 3    2 |
    # | 0    1 |
    #-----------
    Lmat = np.zeros((4,4))
    tmpLmat = Lmat.copy()
    psi = np.zeros((Nx,Ny,4))
    Ax = np.array(((1,1,0,0),(-1,-1,0,0),(0,0,-1,-1),(0,0,1,1)))
    Ay =  np.array(((1,0,0,1),(0,1,1,0),(0,-1,-1,0),(-1,0,0,-1)))
    ihx = 1/hx
    ihy = 1/hy
    Lmat = etax*ihx*Ax + etay*ihy*Ay
    b = np.zeros(4)
    if (etax > 0) and  (etay > 0):
        Lmat[1,1] += 2*etax*ihx
        Lmat[2,2] += 2*etax*ihx + 2*etay*ihy
        Lmat[3,3] += 2*etay*ihy
        
        for i in range(Nx):
            for j in range(Ny):
                if i==0:
                    psileft1 = boundaryx
                    psileft2 = boundaryx
                else:
                    psileft1 = psi[i-1,j,1]
                    psileft2 = psi[i-1,j,2]
                if j==0:
                    psibottom3 = boundaryy
                    psibottom2 = boundaryy
                else:
                    psibottom3 = psi[i,j-1,3]
                    psibottom2 = psi[i,j-1,2]
                
                print("Q = ", Q[10,:,0])
                tmpLmat = Lmat + np.diag(sigmat[i,j,:])
                #print(tmpLmat,Lmat,np.diag(sigmat[i,j,:]))
                b[0] = Q[i,j,0] + psibottom3*2*etay*ihy + psileft1*2*etax*ihx
                b[1] = Q[i,j,1] + psibottom2*2*etay*ihy
                b[2] = Q[i,j,2]
                b[3] = Q[i,j,3] + psileft2*2*etax*ihx
                psi[i,j,:] = np.linalg.solve(tmpLmat,b)
    
    elif (etax < 0) and (etay < 0):
        Lmat[1,1] += -2*etay*ihy
        Lmat[0,0] += -2*etax*ihx - 2*etay*ihy
        Lmat[3,3] += -2*etax*ihx
        
        for i in range(Nx-1,-1,-1):
            for j in range(Ny-1,-1,-1):
                if i==Nx-1:
                    psiright3 = boundaryx
                    psiright0 = boundaryx
                else:
                    psiright3 = psi[i+1,j,3]
                    psiright0 = psi[i+1,j,0]
                if j==Ny-1:
                    psitop0 = boundaryy
                    psitop1 = boundaryy
                else:
                    psitop0 = psi[i,j+1,0]
                    psitop1 = psi[i,j+1,1]
                    
                    
                tmpLmat = Lmat + np.diag(sigmat[i,j,:])
                b[0] = Q[i,j,0]
                b[1] = Q[i,j,1] - psiright0*2*etax*ihx
                b[2] = Q[i,j,2] - psitop1*2*etay*ihy - psiright3*2*etax*ihx
                b[3] = Q[i,j,3] - psitop0*2*etay*ihy
                psi[i,j,:] = np.linalg.solve(tmpLmat,b)
    elif (etax > 0) and (etay < 0):
        Lmat[1,1] += 2*etax*ihx- 2*etay*ihy
        Lmat[0,0] += - 2*etay*ihy
        Lmat[2,2] += 2*etax*ihx
        
        for i in range(Nx):
            for j in range(Ny-1,-1,-1):
                if i==0:
                    psileft1 = boundaryx
                    psileft2 = boundaryx
                else:
                    psileft1 = psi[i-1,j,1]
                    psileft2 = psi[i-1,j,2]
                if j==Ny-1:
                    psitop0 = boundaryy
                    psitop1 = boundaryy
                else:
                    psitop0 = psi[i,j+1,0]
                    psitop1 = psi[i,j+1,1]
                

                tmpLmat = Lmat + np.diag(sigmat[i,j,:])
                b[0] = Q[i,j,0] +  psileft1*2*etax*ihx
                b[1] = Q[i,j,1]
                b[2] = Q[i,j,2] - psitop1*2*etay*ihy
                b[3] = Q[i,j,3] - psitop0*2*etay*ihy + psileft2*2*etax*ihx
                psi[i,j,:] = np.linalg.solve(tmpLmat,b)
    elif (etax < 0) and (etay > 0):
        
        Lmat[0,0] += -2*etax*ihx
        Lmat[2,2] +=  2*etay*ihy
        Lmat[3,3] += 2*etay*ihy -2*etax*ihx
        
        for i in range(Nx-1,-1,-1):
            for j in range(Ny):
                if i==Nx-1:
                    psiright3 = boundaryx
                    psiright0 = boundaryx
                else:
                    psiright3 = psi[i+1,j,3]
                    psiright0 = psi[i+1,j,0]
                if j==0:
                    psibottom3 = boundaryy
                    psibottom2 = boundaryy
                else:
                    psibottom3 = psi[i,j-1,3]
                    psibottom2 = psi[i,j-1,2]


                tmpLmat = Lmat + np.diag(sigmat[i,j,:])
                
                b[0] = Q[i,j,0] + psibottom3*2*etay*ihy
                b[1] = Q[i,j,1] + psibottom2*2*etay*ihy - psiright0*2*etax*ihx
                b[2] = Q[i,j,2] - psiright3*2*etax*ihx
                b[3] = Q[i,j,3]
                psi[i,j,:] = np.linalg.solve(tmpLmat,b)
    return psi


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

def flatten_phi(phi,Nx,Ny,hx,hy):
    x = np.zeros((2*Nx,2*Ny))
    X = np.linspace(hx/2,(Nx*hx)-hx/2,Nx)
    y = np.zeros((2*Nx,2*Ny))
    Y = np.linspace(hy/2,(Ny*hy)-hy/2,Ny)
    phi_out = np.zeros((2*Nx,2*Ny))
    for i in range(Nx):
        for j in range(Ny):
            phi_out[2*i,2*j] = phi[i,j,0]
            phi_out[2*i+1,2*j] = phi[i,j,1]
            phi_out[2*i+1,2*j+1] = phi[i,j,2]
            phi_out[2*i,2*j+1] = phi[i,j,3]
            
            x[2*i,2*j] = X[i] - hx/4
            x[2*i+1,2*j] = X[i] + hx/4
            x[2*i+1,2*j+1] = X[i] + hx/4
            x[2*i,2*j+1] = X[i] - hx/4
            
            y[2*i,2*j] = Y[j] - hy/4
            y[2*i+1,2*j] = Y[j] - hy/4
            y[2*i+1,2*j+1] = Y[j] + hy/4
            y[2*i,2*j+1] = Y[j] + hy/4
    return x,y,phi_out

def SI(Nx,Ny,hx,hy,phi_old,Nord,sigmat,sigmas,q,boundaryx, boundaryy):
    w,etax,etay = prod_quad(N=Nord)
    Qin = (q + sigmas*phi_old)
    print("Qin = ", Qin[10,:,0])
    #print(w,np.sum(w),np.sum(Qin))
    angles = w.size
    phi = np.zeros((Nx,Ny,4))
    for n in range(angles):
        phi += w[n]*sweep2D_scb(Nx,Ny,hx,hy,etax[n],etay[n],sigmat,Qin,boundaryx[n], boundaryy[n])
    return phi

def SI_solve(Nx,Ny,hx,hy,phi_old,Nord,sigmat,sigmas,q,
             boundaryx, boundaryy, L2tol=1e-8, Linftol = 1e-3, LOUD = 0, maxits = 20000):
    phi = phi_old.copy()
    converged = 0
    iteration = 0
    while not(converged):
        
        #plt.pcolor(phi_old[:,:,0])
        #plt.colorbar()
        #plt.title("Phi_old")
        #plt.show()
        phi = SI(Nx,Ny,hx,hy,phi_old,Nord,sigmat,sigmas,q,boundaryx, boundaryy)
        #plt.pcolor(phi[:,:,0])
        #plt.colorbar()
        #plt.title("Phi")
        #plt.show()
        L2diff = np.sqrt(np.sum(((phi-phi_old)/(np.abs(phi)+1e-14))**2/(Nx*Ny*4)))
        Linfdiff = np.max(np.abs(phi-phi_old)/(np.abs(phi)+1e-14))
        if LOUD:
            print("Iteration:",iteration+1,"L2 Diff:",L2diff,"Linf Diff:",Linfdiff)
        if (L2diff < L2tol) and (Linfdiff < Linftol):
            converged = True
        elif (iteration >= maxits):
            converged = True
        iteration += 1
        phi_old = phi.copy()
    return phi,iteration


#Set up problem and solve

Nx = 100
Ny = 60
Lx = 10
Ly = 6
hx = Lx/Nx
hy = Ly/Ny
sigma_t = np.zeros((Nx,Ny,4))+100
sigma_s = sigma_t*0.1
Q = (sigma_t)*0 # specified source
lower_fact = 0.001
#put channel in
for i in range(Nx):
    for j in range(Ny):
        ypos = (j-.5)*hy
        xpos = (i-.5)*hx
        if (ypos <= 2) and (math.fabs(xpos-5) > 1):
            sigma_t[i,j,:] *= lower_fact
            sigma_s[i,j,:] *= lower_fact
        elif (ypos <= 4) and (math.fabs(xpos-5) > 1) and (math.fabs(xpos-5) < 2):
            sigma_t[i,j,:] *= lower_fact
            sigma_s[i,j,:] *= lower_fact
        elif (ypos >2) and (ypos <= 4) and (math.fabs(xpos-5) < 1) and (math.fabs(xpos-5) < 2):
            sigma_t[i,j,:] *= lower_fact
            sigma_s[i,j,:] *= lower_fact

x,y,sigt = flatten_phi(sigma_t,Nx,Ny,hx,hy)
plt.pcolor(x,y,sigt)
plt.colorbar()
plt.show()
phi_old = Q*0 #+1-1e-8
Nord = 8
w,etax,etay = prod_quad(Nord)
boundaryx = np.zeros(w.size)
boundaryy = np.zeros(w.size) + 0#+ 1/np.sum(w)
boundaryx[etax>0] = 1/np.sum(w)
boundaryx[etax > 0] = 1
boundaryy[etay > 0] = 1
phi,iteration = SI_solve(Nx,Ny,hx,hy, phi_old, Nord, sigma_t, sigma_s, Q, boundaryx, boundaryy, LOUD=1)

# phi,iteration = SI_solve(Nx,Ny,hx,hy, phi_old, Nord, sigma_t, sigma_s, Q, boundaryx, boundaryy, LOUD=1)

x,y,phi_flat = flatten_phi(phi,Nx,Ny,hx,hy)

# phi2 = SI(Nx,Ny,hx,hy, phi, Nord, sigma_t, sigma_s, Q, boundaryx, boundaryy)
# x,y,phi_flat2 = flatten_phi(phi2,Nx,Ny,hx,hy)

plt.pcolor(x,y,phi_flat)
plt.colorbar()
plt.show()

# plt.plot(x[:,0],phi_flat[:,0])
# plt.plot(y[0,:],phi_flat[0,:])
# plt.show()

# %%


