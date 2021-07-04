# Master script for determining solution to 1D slab neutron transport
# %%
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

# random constants
# Mass of neutron: 1.675E-27 kg
# 1 eV neutron => 13.83 km/s = 13.83E5 cm/s
# 1 MeV neutron => 13830 km/s
# 1 eV = 1.602E-19 J

# set grid parameters
a = 8
I = 500
x = np.linspace(0, a, I)
# specify discrete ordinates and source
N = 8
S = np.zeros(I) + 0
u = 00 # material velocity
v = 100

# cross sections
sig_t = np.zeros(I) + 0 # total cross section
sig_s = np.zeros(I) + 0 # scattering cross section
sig_tA = 1 # first total cross section 
sig_tB = 10 # second total cross section 
sig_sA = 0 # scattering cross sections 
sig_sB = 4 # scattering cross sections
A = 20 # number of consecutive elements for sig_A
B = 5  # number of consecutive elements for sig_B
# sig_t, sig_s = crossSections(sig_t, sig_s, sig_tA, sig_tB, sig_sA, sig_sB, A, B)

alpha = 1
sig_t, sig_s, S = reedsProblem(x, alpha, sig_t, sig_s, S)

# generate psis with boundary conditions
mu, w = scipy.special.roots_legendre(N)
w = w/np.sum(w)

boundary = np.zeros(N)
boundary[mu > 0] = 0
boundary[mu < 0] = 0
# psi, phi = sweep(a, I, N, sig_t, sig_s, S, boundary)
psi, phi = sweepMotion(u, v, a, I, N, sig_t, sig_s, S, boundary)
plt.figure(1)
for i in range(N):
    plt.plot(x, psi[i,:],"--")
    
# plt.legend()
plt.figure(2)
plt.plot(x,phi, label="Num")
plt.legend()
plt.xlabel("x")
plt.ylabel("Phi")
plt.show


# %%
psi0 = 0
u = 100 # material velocity
v = 100

psiExact = np.zeros((N,x.size))
mu, w = scipy.special.roots_legendre(N)
for n in range(N):

    if v-u == 0:
        gamma = mu[n]
    else:
        gamma = mu[n] + u/(v-u)

    Q = 1
    for i in range(x.size):
        psiExact[n,i] = psi0*math.exp(-sig_t[0]/gamma*x[i]) + Q/gamma*(1 - math.exp(-sig_t[0]/gamma*x[i]))

    
    if n > -1:
        plt.figure(3)
        plt.plot(x,psiExact[n], label=gamma)
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("psi")
        plt.show
        plt.title("gamma = %f"%(gamma))

# %%

# %%
