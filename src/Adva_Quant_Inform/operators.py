

import math
from fractions import Fraction

import numpy as np
import numpy.linalg as LA
import scipy.linalg as sciLA
import scipy.sparse as sparse

from qutip import basis, tensor, qeye, sigmax, sigmay, sigmaz
from ipywidgets import interactive, interact, FloatSlider

from numpy import (
    array, pi, cos, sin, ones, size, sqrt,
    real, mod, append, arange, exp
)


# Basic one-qubit gates and projectors

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
              [0, np.exp(1j * np.pi / 4)]])

P0 = np.array([[1, 0],
               [0, 0]])

P1 = np.array([[0, 0],
               [0, 1]])

I = np.identity(2)

C01 = np.array([[0, 1],
                [0, 0]])

C10 = np.array([[0, 0],
                [1, 0]])


ket0 = (1 / np.sqrt(2)) * basis(2, 0) + (1 / np.sqrt(2)) * basis(2, 1)

def psi0(N):
    """
    Return the N-qubit product state |+>|+>...|+> using qutip objects.
    """
    psi0_flag = tensor([ket0 for n in range(N)])
    return(psi0_flag)

si = qeye(2)
sx = sigmax()
sy = sigmay()
sz = sigmaz()

def sx_list(N):
    """
    Return a list of sigma_x operators acting on each site of an N-qubit system.
    """
    sx_list_flag = []

    for n in range(N):
        op_list = []

        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list_flag.append(tensor(op_list))

    return sx_list_flag

def sy_list(N):
    """
    Return a list of sigma_y operators acting on each site of an N-qubit system.
    """
    sy_list_flag = []

    for n in range(N):
        op_list = []

        for m in range(N):
            op_list.append(si)

        op_list[n] = sy
        sy_list_flag.append(tensor(op_list))

    return sy_list_flag


def sz_list(N):
    """
    Return a list of sigma_z operators acting on each site of an N-qubit system.
    """
    sz_list_flag = []

    for n in range(N):
        op_list = []

        for m in range(N):
            op_list.append(si)

        op_list[n] = sz
        sz_list_flag.append(tensor(op_list))

    return sz_list_flag

# ============================================================
# Basic state helpers
# ============================================================

def basisvec(n, k):
    """
    Return computational basis vector |k> for an n-qubit Hilbert space.
    """
    v = np.zeros(2**n, dtype=complex)
    v[k] = 1
    return v

    
def initRegister(n):
    """
    Initialize an n-qubit register in the computational state |0...0>.
    """
    return basisvec(n, 0)

def indToState(n, k):
    """
    Convert integer k to an n-bit binary state array.
    """
    num = bin(k)[2:].zfill(n)
    return np.array([int(x) for x in num])


def stateToInd(state):
    """
    Convert a binary state array to an integer index.
    """
    return int("".join(str(x) for x in state), 2)


def systemSizeFromState(psi):
    """
    Return the number of qubits from a state vector of length 2^n.
    """
    return int(np.log2(len(psi)))
    

def doMeasurement(psi):
    """
    Simulate a projective measurement in the computational basis.

    Parameters
    ----------
    psi : array-like
        State vector.

    Returns
    -------
    numpy.ndarray
        Measured bit string.
    """
    n = systemSizeFromState(psi)
    pvec = np.abs(psi)**2
    thresholds = np.cumsum(pvec)
    r = np.random.rand()
    indOutcome = np.sum(thresholds < r)

    return indToState(n, indOutcome)

# Dense helpers functions


def rotation(ax, theta):
    """
    Return single-qubit rotation around axis ax by angle theta.
    """
    return sciLA.expm(-1j * theta / 2 * (ax[0] * X + ax[1] * Y + ax[2] * Z))
    

def U_N_qubits(ops):
    """
    Construct an N-qubit dense operator using tensor products.
    """
    U = ops[0]

    for op in ops[1:]:
        U = np.kron(U, op)

    return U


def U_one_gate(V, i, N):
    """
    Apply a single-qubit dense gate V to qubit i in an N-qubit system.
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

# ============================================================
# Sparse gate helpers 
# ============================================================

def buildSparseGateSingle(n, i, gate):
    """
    Construct a single-qubit gate acting on qubit i in an n-qubit system
    using sparse Kronecker products.

    This embeds a 2×2 quantum gate into the full 2^n-dimensional Hilbert space:

        I ⊗ I ⊗ G ⊗ I ⊗ ... ⊗ I

    where the gate G is applied at position i.
    """
    if not (0 <= i < n):
        raise ValueError("Target qubit index must satisfy 0 <= i < n.")

    sgate = sparse.csr_matrix(gate)

    return sparse.kron(
        sparse.kron(
            sparse.identity(2**i, format="csr", dtype=complex),
            sgate,
            format="csr"
        ),
        sparse.identity(2**(n - i - 1), format="csr", dtype=complex),
        format="csr"
    )
def buildSparseCNOT(n, ic, it):
    """
    Construct an n-qubit controlled-NOT gate using sparse matrices.
    """
    if ic == it:
        raise ValueError("Control and target must be different.")

    if not (0 <= ic < n and 0 <= it < n):
        raise ValueError("Qubit indices must satisfy 0 <= ic, it < n.")

    P0_term = buildSparseGateSingle(n, ic, P0)
    P1_term = buildSparseGateSingle(n, ic, P1)
    X_term = buildSparseGateSingle(n, it, X)

    return P0_term + P1_term @ X_term


def Rk(k):
    """
    Phase rotation gate used in QFT circuits.
    """
    return np.array(
        [[1, 0],
         [0, np.exp(1j * 2 * np.pi / 2**k)]],
        dtype=complex
    )


def buildSparseCRk(n, ic, it, k):
    """
    Construct an n-qubit controlled-Rk gate using sparse matrices.
    """
    if ic == it:
        raise ValueError("Control and target qubits must be different.")

    if not (0 <= ic < n and 0 <= it < n):
        raise ValueError("Qubit indices must satisfy 0 <= ic, it < n.")

    P0ic = buildSparseGateSingle(n, ic, P0)
    P1ic = buildSparseGateSingle(n, ic, P1)
    Rksp = buildSparseGateSingle(n, it, Rk(k))

    return P0ic + P1ic @ Rksp


def buildSparseSwap(n, i, j):
    """
    Construct an n-qubit SWAP gate between qubits i and j using sparse matrices.
    """
    if i == j:
        return sparse.identity(2**n, format="csr", dtype=complex)

    if not (0 <= i < n and 0 <= j < n):
        raise ValueError("Qubit indices must satisfy 0 <= i, j < n.")

    P0i = buildSparseGateSingle(n, i, P0)
    P1i = buildSparseGateSingle(n, i, P1)
    C01i = buildSparseGateSingle(n, i, C01)
    C10i = buildSparseGateSingle(n, i, C10)

    P0j = buildSparseGateSingle(n, j, P0)
    P1j = buildSparseGateSingle(n, j, P1)
    C01j = buildSparseGateSingle(n, j, C01)
    C10j = buildSparseGateSingle(n, j, C10)

    return P0i @ P0j + P1i @ P1j + C01i @ C10j + C10i @ C01j

# ============================================================
# Deutsch-Jozsa helpers
# ============================================================

def buildUf(f, n):
    """
    Build the phase oracle U_f for the Deutsch-Jozsa algorithm.
    """
    return sparse.diags(
        [(-1)**f(indToState(n, x)) for x in range(2**n)],
        format="csr"
    )


def deutschJosza(f, n):
    """
    Run the Deutsch-Jozsa algorithm for a Boolean function f.
    """
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

# Simple variance helpers

def varianceSigmaZ(x):
    """
    Variance of sigma_z for a simple one-parameter state.
    """
    return 1 - (2 * x**2 - 1)**2


def NegVarianceSigmaZ(x):
    """
    Negative variance, useful for minimization routines.
    """
    return -varianceSigmaZ(x)


# ============================================================
# QFT and inverse QFT
# ============================================================

def build_QFT(n):
    """
    Build the n-qubit Quantum Fourier Transform as a sparse matrix.
    """
    QFT = sparse.identity(2**n, format="csr", dtype=complex)

    for i in range(n):
        QFT = buildSparseGateSingle(n, i, H) @ QFT

        for j in range(i + 1, n):
            QFT = buildSparseCRk(n, j, i, j - i + 1) @ QFT

    for i in range(n // 2):
        QFT = buildSparseSwap(n, i, n - i - 1) @ QFT

    return QFT


def build_invQFT(n):
    """
    Build the n-qubit inverse Quantum Fourier Transform as a sparse matrix.
    """
    invQFT = sparse.identity(2**n, format="csr", dtype=complex)

    for i in range(n // 2):
        invQFT = buildSparseSwap(n, i, n - i - 1) @ invQFT

    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            invQFT = buildSparseCRk(n, j, i, j - i + 1).conjugate() @ invQFT
         # H is its own inverse
        invQFT = buildSparseGateSingle(n, i, H) @ invQFT

    return invQFT

def apply_invQFT(n, Psi):
    """
    Apply the n-qubit inverse QFT directly to state vector Psi.
    """
    # swaps
    for i in range(n // 2):
        Psi = buildSparseSwap(n, i, n - i - 1) @ Psi

    # H and inverse C-Rk
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            Psi = buildSparseCRk(n, j, i, j - i + 1).getH() @ Psi

        Psi = buildSparseGateSingle(n, i, H) @ Psi

    return Psi

def apply_invQFT_reg1(n, Psi):
    """
    Apply inverse QFT only to the first n qubits of a larger register.
    """
    ntot = systemSizeFromState(Psi)

    # final swaps from QFT, but only inside first register
    for i in range(n // 2):
        Psi = buildSparseSwap(ntot, i, n - i - 1) @ Psi

    # Apply inverse controlled rotations and Hadamards((C-Rk and H))
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            Psi = buildSparseCRk(ntot, j, i, j - i + 1).getH() @ Psi

        # H is self-inverse
        Psi = buildSparseGateSingle(ntot, i, H) @ Psi

    return Psi
    
# ============================================================
# Shor algorithm: classical helpers
# ============================================================

def qubits_for_number(N):
    """
    Return the minimum number of qubits needed to encode integers 0,...,N-1(L).
    """
    if N <= 1:
        raise ValueError("N must be greater than 1.")

    return int(np.ceil(np.log2(N)))
   

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
    Find non-trivial factors of N using the order r of x modulo N.

    This is the classical post-processing step used in Shor's algorithm.

        x**r ≡ 1 mod N

    If r is even and x**(r/2) is not congruent to -1 mod N,
    then gcd(x**(r/2) - 1, N) and gcd(x**(r/2) + 1, N)
    may give factors of N.

    Parameters
    ----------
    x : int
        The chosen base number.
    N : int
        The composite number to factor.

    Returns
    -------
    list or str
        A list of possible factors if successful.
        Otherwise, an error message explaining why it failed.
    """
    # If x already shares a factor with N, we are done.
    g = math.gcd(x, N)

    if 1 < g < N:
        return [g, N // g]

    r = find_order_brute_force(x, N)

    if isinstance(r, str):
        return r

    if r % 2 != 0:
        return "r was odd"

    a = pow(x, r // 2, N)

    if a == N - 1:
        return "x^(r/2)=-1"

    guesses = [
        math.gcd(a - 1, N),
        math.gcd(a + 1, N)
    ]

    factors = [g for g in guesses if 1 < g < N]

    if factors:
        return factors

    return "No non-trivial factors found"


# ============================================================
# Continued fraction helpers
# ============================================================

def eval_contfrac(frac):
    """
    Evaluate a continued fraction as a floating-point number.
    """
    if len(frac) == 0:
        raise ValueError("Continued fraction list cannot be empty.")
    value = 0
    for a in reversed(frac[1:]):
        value = 1 / (a + value)

    return frac[0] + value


def eval_contfrac_rational(frac):
    """
    Evaluate a continued fraction exactly as [numerator, denominator].
    """
    if len(frac) == 0:
        raise ValueError("Continued fraction list cannot be empty.")
    if len(frac) == 1:
        return [frac[0], 1]
    numer = 1
    denom = frac[-1]
    for a in reversed(frac[1:-1]):
        numer, denom = denom, a * denom + numer
    numer = frac[0] * denom + numer

    return [numer, denom]


def cont_frac(phi, max_denom):
    """
    Compute the continued fraction approximation of phi.

    The expansion is stopped before the denominator becomes larger
    than max_denom.

    Parameters
    ----------
    phi : float
        Number to approximate.
    max_denom : int
        Maximum allowed denominator.

    Returns
    -------
    list
        Continued fraction coefficients.
    """
    if max_denom < 1:
        raise ValueError("max_denom must be positive.")
    frac = []
    a = int(phi // 1)
    r = phi - a
    frac.append(a)
    while r > 1 / max_denom:
        a = int((1 / r) // 1)
        candidate = frac + [a]
        numerator, denominator = eval_contfrac_rational(candidate)
        if denominator > max_denom:
            break
        frac = candidate
        r = (1 / r) - a

    return frac 


# Shor algorithm: quantum order finding
   

def build_x_tothe_z(t, x, N):
    """
    Build the modular exponentiation operator:

        |z>|y> -> |z>|y*x^z mod N>

    The first register has t qubits. The second register has L qubits,
    where L = qubits_for_number(N).
    """
    L = qubits_for_number(N)
    dim1 = 2**t
    dim2 = 2**L
    dim = dim1 * dim2

    row = arange(dim) # indexes all rows
    col = arange(dim) # will contain the position (col) of the non-zero entry for each row

    for j in range(dim1): # loop over states of the first register
        for y in range(dim2): # loop over states of the second register
            if y < N:          # if y>=N, we want the identity, so we leave the
                row[j * dim2 + y] = j * dim2 + np.mod(y * pow(x, j, N), N) # row index unchanged (it was initialized to be equal to the col index)

    return sparse.csr_matrix((np.ones(dim), (row, col)))
    
def find_order(t, x, N):
    """
    Quantum order-finding routine used in Shor's algorithm.

    This prepares two quantum registers, applies modular exponentiation,
    and then applies the inverse QFT to the first register.

    Returns
    -------
    numpy.ndarray
        Final quantum state after inverse QFT.
    """
    L = qubits_for_number(N)
    n = t + L

    psi = initRegister(n)

    # Prepare the second register in state |1>.
    psi = buildSparseGateSingle(n, n - 1, X) @ psi

    # Apply Hadamards to the first register.
    for i in arange(t):
        psi = buildSparseGateSingle(n, i, H) @ psi

    # Apply |z>|y> -> |z>|y*x^z mod N>.
    psi = build_x_tothe_z(t, x, N) @ psi

    # Apply inverse QFT to the first register.
    psi = apply_invQFT_reg1(t, psi)

    return psi


def try_find_order(N, t, x, max_runs):
    """
    Try to find the order of x modulo N using quantum order finding.

    The quantum circuit is run up to max_runs times. Each measurement
    gives a value from the first register. A continued-fraction
    approximation is then used to guess the order.
    """
    print("Trying order finding with x =", x)

    psi = find_order(t, x, N)
    probs = np.abs(psi)**2

    print("Most likely state index:", np.argmax(probs))
    print("Maximum probability:", max(probs))

    runs = 0
    order = 1

    while runs < max_runs:
        runs += 1

        # Measure all qubits from the quantum state
        result = doMeasurement(psi)

        # Keep only the first register
        reg1 = result[:t]
        reg1_num = stateToInd(reg1)

        print("The first register was measured in state", reg1, "i.e. k =", reg1_num)

        if reg1_num == 0:
            print("Measured k = 0, trying again")
            continue

        # Exact continued fraction approximation of k / 2^t
        frac = Fraction(reg1_num, 2**t).limit_denominator(N)
        s = frac.numerator
        order_guess = frac.denominator

        print("Continued fraction gave s =", s, ", r =", order_guess)

        # Combine guesses using least common multiple
        order = int(np.lcm(order, order_guess))

        # Check whether this is really the order
        if pow(x, order, N) == 1:
            print("Order was found:", order)
            return order, True

    print("Order finding failed. Try again or use larger t.")
    return None, False
    
def try_factoring(N, t, x, max_runs):
    """
    Try to factor N by finding the order of x modulo N.

    This function first checks whether x already shares a non-trivial
    factor with N. If not, it uses quantum order finding to estimate
    the order r of x modulo N. If r is even and x^(r/2) is not -1 mod N,
    then gcd(x^(r/2) ± 1, N) may give a non-trivial factor of N.
    """
    # Check coprimality first
    direct_factor = math.gcd(x, N)

    if direct_factor > 1:
        print("Factor found directly")
        return direct_factor

    order, success = try_find_order(N, t, x, max_runs)

    if not success:
        return None

    if order % 2 != 0:
        print("r was odd, try different x")
        return None

    a = pow(x, order // 2, N)

    if a == N - 1:
        print("x^(r/2) = -1 (mod N), try different x")
        return None

    factor = math.gcd(a - 1, N)
    if factor > 1 and N % factor == 0:
        return factor

    factor = math.gcd(a + 1, N)
    if factor > 1 and N % factor == 0:
        return factor

    print("No non-trivial factor found, try again")
    return None

def factoring(N, t, x_start, max_runs):
    """
    Try to factor N using Shor-style order finding.

    This function tests possible base values x starting from x_start.
    For each x, it calls try_factoring(N, t, x, max_runs). If a
    non-trivial factor is found, the function returns it.

    Parameters
    ----------
    N : int
        Composite number to factor.
    t : int
        Number of qubits in the first/order-finding register.
    x_start : int
        First value of x to try.
    max_runs : int
        Maximum number of measurement attempts for each x.

    Returns
    -------
    int or None
        A non-trivial factor of N if found. Otherwise None.
    """
    for x in range(x_start, N): # we change x incementally for simplicity.
         # One could also select random x each time.
        factor = try_factoring(N, t, x, max_runs) # Function below!

        if factor is not None:
            print("Success! A factor was found!")
            return factor

    print("Giving up. No factor found :-(")
    return None 
