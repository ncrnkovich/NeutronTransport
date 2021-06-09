# file to set up cross section vectors for various conditions

# import libraries
from operator import iadd
from os import write
import numpy as np
import math as math
import matplotlib.pyplot as plt
from numpy.core.numeric import Inf
import scipy as scipy
from scipy.constants import constants
import scipy.special

def crossSections(sig_t, sig_s, sig_tA, sig_tB, sig_sA, sig_sB, A, B):
    
    j = 0
    for i in range(len(sig_t)):

        if j == A+B:
            j = 0

        if j < A:
            sig_t[i] = sig_tA
            sig_s[i] = sig_sA
            j += 1
        elif j < A+B:
            sig_t[i] = sig_tB
            sig_s[i] = sig_sB
            j += 1
        
        

    return sig_t, sig_s

