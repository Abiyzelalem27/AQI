

import matplotlib.pyplot as plt
import numpy as np  
from qutip import *
import math
from scipy.special import hermite as herm
import scipy.sparse as sparse 
from Adva_Quant_Inform.operators import sx_list, sz_list 

# Construct the trivial Hamiltonian
def H0(N):
    H0_flag = 0    
    for n in range(N):
        H0_flag += - sx_list(N)[n]
    return(H0_flag)

# Construct the problem Hamiltonian s
def H1(N): 
    H1_flag = 0 
    # array of single qubit longitudinal fields. This is useful to break the 
    # ground state degeneracy between |01010> and |10101> configurations
    h = 0.5 * 2 * np.pi * np.ones(N) 
    Jz = 1.0 * 2 * np.pi * np.ones(N) # antiferromagnetic spin-spin interaction coefficients
    for n in range(N):
        H1_flag += - 0.5 * h[n] * sz_list(N)[n]

    for n in range(N-1):
        # interaction terms
        H1_flag +=  0.5 * Jz[n] * sz_list(N)[n] * sz_list(N)[n+1] 
    
    return(H1_flag) 
    
# define function for H(t)
def H_t(t, H0, H1, N):
    H_at_t = (1-t/taumax)*H0(N) + (t/taumax)*H1(N)
    return H_at_t 
    
# compute the ground state of H1
def psiH1(N):
    _, psiH1_flag = H1(N).eigenstates()
    psiH1_flag = psiH1_flag[0] # selecting the ground state
    return psiH1_flag 
    
