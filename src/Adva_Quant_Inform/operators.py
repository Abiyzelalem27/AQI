

import math
import numpy as np  
import numpy.linalg as LA
import scipy.linalg as sciLA
from qutip import basis 
import scipy.sparse as sparse
from ipywidgets import interactive, interact
from ipywidgets import FloatSlider 
from qutip import tensor, qeye, sigmax, sigmay, sigmaz 
from numpy import (array, pi, cos, sin, ones, size, sqrt, real, mod, append, arange, exp)

X = np.array([[0, 1],
              [1, 0]])

Y = np.array([[0, -1j],
              [1j, 0]])

Z = np.array([[1, 0],
              [0, -1]])

H = np.array([[1, 1],
              [1, -1]]) / np.sqrt(2)

S = np.array([[1, 0],
              [0, 1j]])

T = np.array([[1, 0],
              [0, np.exp(1j*np.pi/4)]])

P0 = np.array([[1, 0],
               [0, 0]])

P1 = np.array([[0, 0],
               [0, 1]])

I = np.identity(2) 

ket0 = 1/np.sqrt(2)*basis(2,0) + 1/np.sqrt(2)*basis(2,1) 
def psi0(N):
    psi0_flag = tensor([ket0 for n in range(N)])
    return(psi0_flag)
    
# helper function for generating basis vectors
def basisvec(n, k):
    v = np.zeros(2**n)
    v[k] = 1
    return v 
    
def rotation(ax,theta):
    return sciLA.expm(-1j * theta/2 * (ax[0]*X + ax[1]*Y + ax[2]*Z)) 
    

def buildSparseGateSingle(n, i, gate):
    """
    Construct a single-qubit gate acting on qubit i in an n-qubit system
    using sparse Kronecker products.

    This embeds a 2×2 quantum gate into the full 2^n-dimensional Hilbert space:

        I ⊗ I ⊗ G ⊗ I ⊗ ... ⊗ I

    where the gate G is applied at position i.

    Parameters
    ----------
    n : Total number of qubits in the system.
    i : Target qubit index (0 ≤ i < n).
    gate :  2×2 quantum gate to apply (e.g., X, H, Z).
    
    Notes
    -----
    - Uses Kronecker products to embed single-qubit operations.
    """

    sgate = sparse.csr_matrix(gate)
    return sparse.kron(
        sparse.kron(
            sparse.identity(2**i, format="csr"),
            sgate
        ),
        sparse.identity(2**(n - i - 1), format="csr")
    )


def buildSparseCNOT(n, ic, it):
    """
    Construct an n-qubit controlled-NOT (CNOT) gate using sparse matrices.
    The CNOT gate is built using projector decomposition:
        CNOT = |0><0|_c ⊗ I_t + |1><1|_c ⊗ X_t

    where:
        - n  : total number of qubits
        - ic : control qubit index (0 ≤ ic < n)
        - it : target qubit index (0 ≤ it < n, it ≠ ic)
    Notes
    -----
    - Uses tensor-product construction with sparse matrices.
    - Computational cost grows exponentially with n (O(2^n)).
    """

    P0_term = buildSparseGateSingle(n, ic, P0)
    P1_term = buildSparseGateSingle(n, ic, P1)
    X_term  = buildSparseGateSingle(n, it, X)
    return P0_term + P1_term @ X_term
    
def U_N_qubits(ops):
    """
    Constructs an N-qubit operator using tensor products.
    """
    U = ops[0]
    for op in ops[1:]:
        U = np.kron(U, op)
    return U

def U_one_gate(V, i, N):
    """
    Applies a single-qubit gate to qubit i in an N-qubit system.

    Parameters
   ...........
    V : Single-qubit gate.
    i : Target qubit index.
    N : Total number of qubits.
    """
    ops = [I] * N
    ops[i] = V
    return U_N_qubits(ops)
    
def controlled_gate(U, control, target, N):
    """
    Controlled-U gate on an N-qubit register.
    Parameters
    ...........
   U:Single-qubit gate
   N: total number of qubits 
    """
    if control == target:
        raise ValueError("Control and target must be different.")

    # Operator acting on the subspace where control qubit is |0⟩
    P0_ops = [
        P0 if i == control else I
        for i in range(N)
    ]

    # Operator acting on the subspace where control qubit is |1⟩
    P1_ops = [
        P1 if i == control else U if i == target else I
        for i in range(N)
    ]

    return U_N_qubits(P0_ops) + U_N_qubits(P1_ops)

def deutschJosza(f, n):
    psi = initRegister(n)
    # apply the Hadamards
    for i in arange(n):
        psi = buildSparseGateSingle(n,i,H) @ psi
    # apply U_f
    psi = buildUf(f, n) @ psi
    # apply the Hadamards again
    for i in arange(n):
        psi = buildSparseGateSingle(n,i,H) @ psi

    # If the probability of having the all zero state is 1, then f is constant.
    # Since the state of all zero is represented in the computational basis by 1 in the first entry 
    # and then all zeros, one can just check np.isclose(np.abs(psi[0])**2, 1), 
    # namely the probability equal to one for the first element of the vector psi[0].
    # Even simpler, f is constant iff psi[0]=\pm 1 the function is constant (we have a binary choice).
    if psi[0] == 0:
        print('The function is balanced.')
    else:
        print('The function is constant.')

    # checking
    ratio = np.sum([f(indToState(n,x)) for x in range(2**n)])/2**n
    print("The ratio of ones to zeros (computed directly) is:", ratio)
    return psi 

def initRegister(n):
    return basisvec(n,0)

def indToState(n, k):
    num = bin(k)[2:].zfill(n)
    return array([int(x) for x in str(num)])

def stateToInd(state):
    return int("".join(str(x) for x in state),2)

def buildUf(f, n):
    return sparse.diags([(-1)**f(indToState(n,x)) for x in range(2**n)]) 

def systemSizeFromState(psi):
    return int(np.log2(len(psi)))

# The following function picks a random vector out of the possible outcomes 
def doMeasurement(psi):
    n = systemSizeFromState(psi)
    pvec = np.abs(psi)**2
    thresholds = np.cumsum(pvec)
    r = np.random.rand()
    indOutcome = np.sum(thresholds < r)
    return indToState(n, indOutcome)


# The previous task can be also performed differently.
# In a more abstract language, one might do:

# ind = 0
# cumprob = 0
# r = rand(0,1)
# while cumprob < r
#    cumprob += P(ind)
#    ind++

# where the index of interest when one stops is ind-1. 
# P(ind) is the probability associated to the outcome identified by ind.

def varianceSigmaZ(x):
    return 1 - (2*x**2-1)**2  
# Define the function to maximize (we negate it to use minimize_scalar)
def NegVarianceSigmaZ(x):
    return -varianceSigmaZ(x)

# pre-allocate operators
si = qeye(2) # identity
sx = sigmax()
sy = sigmay()
sz = sigmaz()

def sx_list(N):
    sx_list_flag = []
    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)
        op_list[n] = sx
        sx_list_flag.append(tensor(op_list))
    return(sx_list_flag)

def sy_list(N):
    sy_list_flag = []
    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)
        op_list[n] = sy
        sy_list_flag.append(tensor(op_list))
    return(sy_list_flag)

def sz_list(N):
    sz_list_flag = []
    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)
        op_list[n] = sz
        sz_list_flag.append(tensor(op_list))
    return(sz_list_flag) 


#################################### shor algorithms ##############################################

def find_order_brute_force(x, N):
    """
    Find the multiplicative order of x modulo N using brute force.

    The order r is the smallest positive integer such that:

        x**r ≡ 1 mod N

    This function is useful in number theory and in Shor's algorithm,
    where finding the period/order helps factor a composite number N.

   
    Returns
    -------
    int or str
        The order r if x and N are coprime.
        Returns an error message if x and N are not coprime or if x = 1.
    """
    if math.gcd(x, N) != 1:
        return "Error: x and N are not coprime"

    if x == 1:
        return "Error: x=1 is trivial"

    r = 1
    y = x

    while y != 1:
        y = (x * y) % N
        r += 1

    return r

def try_find_factors(x, N):
    """
    find non-trivial factors of N using the order r of x modulo N.

    This is the classical post-processing step used in Shor's algorithm.

        x**r ≡ 1 mod N

    If r is even and x**(r/2) is not congruent to -1 mod N,
    then gcd(x**(r/2) - 1, N) and gcd(x**(r/2) + 1, N)
    may give factors of N.

    Parameters
    ----------
    x:The chosen base number.
    N :The composite number to factor.

    Returns
    -------
    list or str
        A list of possible factors if successful.
        Otherwise, an error message explaining why it failed.
    """
    r=find_order_brute_force(x,N)
    if isinstance(r, str):
        # passing on the error message thrown by order finding
        return r
    if r%2==0:
        if pow(x,r//2,N)!=N-1:
            # it worked! We have a factor!
            guesses = [math.gcd(x**(r//2)-1, N), math.gcd(x**(r//2)+1, N)]
            return guesses
        else:
            return "x^(r/2)=-1"
    else:
        return "r was odd"
def Rk(k):
    return array([[1, 0],
                  [0, exp(1j*2*pi/2**k)]])


def build_QFT(n):
    QFT = sparse.identity(2**n)
    for i in range(n):
        QFT = buildSparseGateSingle(n, i, H) @ QFT
        for j in range(i+1,n):
            QFT = buildSparseCRk(n, j, i, j-i+1) @ QFT
    for i in range(n//2):
        QFT = buildSparseSwap(n, i, n-i-1) @ QFT
    return QFT

def build_invQFT(n):
    QFT = sparse.identity(2**n)
    # swaps
    for i in range(n//2):
        QFT = buildSparseSwap(n, i, n-i-1) @ QFT
    # H and C-Rk
    for i in range(n-1,-1,-1):
        for j in range(n-1,i,-1):
            QFT = buildSparseCRk(n, j, i, j-i+1).conjugate() @ QFT
        QFT = buildSparseGateSingle(n, i, H) @ QFT
    return QFT


def apply_invQFT(n,Psi):
     # swaps
     for i in range(n//2):
         Psi = buildSparseSwap(n, i, n-i-1) @ Psi
     # H and C-Rk
     for i in range(n-1,-1,-1):
         for j in range(n-1,i,-1):
             Psi = buildSparseCRk(n, j, i, j-i+1).conjugate() @ Psi
         Psi = buildSparseGateSingle(n, i, H) @ Psi
     return Psi

# only apply the inverse QFT to the first n qubits. This is what will be needed for order finding!
def apply_invQFT_reg1(n,Psi):
    ntot = systemSizeFromState(Psi)
    # swaps
    for i in range(n//2):
        Psi = buildSparseSwap(ntot, i, n-i-1) @ Psi
    # H and C-Rk
    for i in range(n-1,-1,-1):
        for j in range(n-1,i,-1):
            Psi = buildSparseCRk(ntot, j, i, j-i+1).conjugate() @ Psi
        Psi = buildSparseGateSingle(ntot, i, H) @ Psi
    return Psi
    
def qubits_for_number(N):
    "Return minimum number of qubits needed to encode L"
    return int(np.ceil(np.log2(N)))

def build_x_tothe_z(t, x, N):
    L = qubits_for_number(N)
    dim1=2**t
    dim2=2**L
    dim=dim1*dim2
    row=arange(dim) # indexes all rows
    col=arange(dim) # will contain the position (col) of the non-zero entry for each row
    for j in range(dim1): # loop over states of the first register
        for y in range(dim2): # loop over states of the second register
            if y < N: # if y>=N, we want the identity, so we leave the
                # row index unchanged (it was initialized to be equal to the col index)
                row[j*dim2+y]=j*dim2 + np.mod(y*pow(x,j,N),N)
    return sparse.csr_matrix((np.ones(dim), (row, col))) 

# solution

def find_order(t, x, N):
    L = qubits_for_number(N)
    n=t+L
    # initialize register
    psi = initRegister(n)
    # flip last qubit of second register to prepare the register in state |1>
    psi = buildSparseGateSingle(n, n-1, X) @ psi
    # apply Hadamards to the first register
    for i in arange(t):
        psi = buildSparseGateSingle(n,i,H) @ psi
    # apply |z>|y> -> |z>|y*x^z mod N>
    psi = build_x_tothe_z(t,x,N) @ psi
    # apply inverse QFT to the first register
    psi = apply_invQFT_reg1(t,psi)
    return psi 