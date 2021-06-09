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
import sweepFunction
from sweepFunction import sweep
import crossSections
from crossSections import crossSections

# set grid parameters
a = 1
I = 100
x = np.linspace(0, a, I)
# specify discrete ordinates and source
N = 4
S = 0

# cross sections
sig_t = np.zeros(I)  # total cross section
sig_s = np.zeros(I)  # scattering cross section
sig_tA = 1 # first total cross section 
sig_tB = 0 # second total cross section 
sig_sA = 0 # scattering cross sections 
sig_sB = 0 # scattering cross sections
A = 10 # number of consecutive elements for sig_A
B = 0 # number of consecutive elements for sig_B
sig_t, sig_s = crossSections(sig_t, sig_s, sig_tA, sig_tB, sig_sA, sig_sB, A, B)

# generate psis with boundary conditions
psiEdgeL = 1
psiEdgeR = 0

psi, phi = sweep(a, I, N, sig_t, sig_s, S, psiEdgeL, psiEdgeR)

plt.figure(1)
for i in range(N):
    plt.plot(x, psi[i,:],"--")
    
plt.legend()
plt.figure(2)
plt.plot(x,phi, label="Num")
plt.legend()
plt.xlabel("x")
plt.ylabel("Phi")
plt.show


# %%