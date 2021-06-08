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

# set grid parameters
a = 1
I = 100
x = np.linspace(0, a, I)
# specify discrete ordinates and source
N = 4
S = 0

# cross sections
sig_t = np.zeros(I)+ 1 # total cross section
sig_s = np.zeros(I) + 0 # scattering cross section

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