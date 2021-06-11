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


def reedsProblem(x, alpha, sig_t, sig_s, S):

    # for i in range(len(x)):
    #     if x[i] < 2:
    #         sig_s[i] = 0.9*alpha
    #         sig_t[i] = 0.9*alpha + 0.1*alpha 
    #         S[i] = 0
    #     elif x[i] < 3:
    #         sig_s[i] = 0.9*alpha
    #         sig_t[i] = 0.9*alpha + 0.1*alpha 
    #         S[i] = alpha
    #     elif x[i] < 5:
    #         sig_s[i] = 0.0*alpha
    #         sig_t[i] = 0.0*alpha 
    #         S[i] = 0
    #     elif x[i] < 6:
    #         sig_s[i] = 0.0*alpha
    #         sig_t[i] = 5.0*alpha 
    #         S[i] = 0
    #     elif x[i] <= 8:
    #         sig_s[i] = 0.0*alpha
    #         sig_t[i] = 50.0*alpha 
    #         S[i] = 50*alpha
    #     else:
    #         sig_s[i] = 0.0*alpha
    #         sig_t[i] = 0.0*alpha 
    #         S[i] = 0*alpha
    for i in range(len(x)):
        if x[i] < 2:
            sig_s[i] = 0.0*alpha
            sig_t[i] = 50*alpha 
            S[i] = 50*alpha
        elif x[i] < 3:
            sig_s[i] = 0*alpha
            sig_t[i] = 5*alpha 
            S[i] = 0
        elif x[i] < 5:
            sig_s[i] = 0.0*alpha
            sig_t[i] = 0.0*alpha 
            S[i] = 0
        elif x[i] < 6:
            sig_s[i] = 0.9*alpha
            sig_t[i] = 1.0*alpha 
            S[i] = 1
        elif x[i] <= 8:
            sig_s[i] = 0.9*alpha
            sig_t[i] = 1.0*alpha 
            S[i] = 0*alpha
        else:
            sig_s[i] = 0.0*alpha
            sig_t[i] = 0.0*alpha 
            S[i] = 0*alpha

    return sig_t, sig_s, S