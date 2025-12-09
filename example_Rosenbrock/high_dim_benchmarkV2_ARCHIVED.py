
"""
R.CLOAREC
10-11-2025

High dimensional benchmark
ARCHIVED DO NOT TOUCH

Hyperparameters ato be tuned are :
scaling factor (dep on dimension adn order)
lambda in ridge regression
tol in ALID
rank for LR

in LR see how to build efficiently the Z and avoid full inversion
in ridge use Cholesky and 
"""


# imports
import numpy as np
import tensoropV2 as to

from scipy.linalg import solve_triangular

import matplotlib.pyplot as plt


#==========================================================
# utils

def LHS(
    n_samples: int,
    n_dim: int,
    bounds: np.ndarray | list[tuple[float, float]] | None = None,
    seed: int = None
) -> np.ndarray:
    """
    Generate a Latin Hypercube Sample (LHS) using only NumPy, with optional parameter ranges.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_dim : int
        Number of dimensions.
    bounds : array-like of shape (n_dim, 2), optional
        Lower and upper bounds for each dimension.
        If None, defaults to [0, 1] for all dimensions.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    samples : np.ndarray
        A (n_samples, n_dim) array containing samples within the specified bounds.
    """
    rng = np.random.default_rng(seed)

    # Divide [0,1] into n_samples intervals for each dimension
    cut = np.linspace(0, 1, n_samples + 1)

    # Generate random points within each interval
    u = rng.random((n_samples, n_dim))
    a = cut[:n_samples]
    b = cut[1:n_samples + 1]
    points = a[:, None] + (b - a)[:, None] * u

    # Randomly permute each column (dimension)
    for j in range(n_dim):
        rng.shuffle(points[:, j])

    # If bounds are provided, scale samples to the given ranges
    if bounds is not None:
        bounds = np.asarray(bounds)
        if bounds.shape != (n_dim, 2):
            raise ValueError(f"`bounds` must have shape ({n_dim}, 2)")
        lower, upper = bounds[:, 0], bounds[:, 1]
        points = lower + points * (upper - lower)

    return points

#==========================================================

# set dimension of the problem
dim = 300

# number of samples 
m = 1500

# mesh to plot result
domain_1D = (-5., 10.)

# define intervals for parameters
param_bounds = dim*[domain_1D]

# exact function
def f(x):
    """
    Rosenbrock function

    input:
    x (tuple) dim gives the dimension of the problem
    
    output:
    evaluation of the Rosenbrick function at x
    """

    out = 0.

    for i in range(len(x)-1):

        out += 100.*(x[i+1]-x[i]**2)**2 + (x[i]-1.)**2
    
    return out 

# build LHS
lhs = LHS(n_samples=m, n_dim=dim, bounds=param_bounds)

# build RHS
rhs = np.array([f(xi) for xi in lhs])

#===================== TENSOR DEFINITION =============

# scaling factor
sf = 1e-5

# define core functions
psi = lambda x: np.array([1., sf*x, sf*x**2, sf*x**3, sf*x**4])

# build m-tensor
phi = to.Tensor([np.array([psi(lhs[k, i]) for k in range(m)]) for i in range(dim)])

# build projector
P = phi@phi

# decompose P
U, S = np.linalg.svd(P)[:2]
S = np.sqrt(S)

print(S[:50])

#===================== RIDGE TENSOR REGRESSION =============

RIDGE = True

if RIDGE:
    
    # regul parameter
    l = 0.

    # compute vanilla Z from P and rhs
    Z = np.linalg.lstsq(P + l*np.eye(m), rhs)[0]

    check_rhs = phi@phi@Z

    # should be shit
    print('Error l2 Ridge',np.linalg.norm(rhs-check_rhs)/np.linalg.norm(rhs))

    # vanilla reconstruction
    def fR(x):

        phi_x = to.Tensor([psi(x[i]) for i in range(dim)])

        return ((phi_x@phi)@Z)[0]


#===================== LR TENSOR REGRESSION =============

LR = True

if LR:

    rnk = 300
    U_, S_ = U[:, :rnk], S[:rnk]
    P_ = (S_*U_) @ (S_*U_).T

    # test linearization by regressing
    # compute Z from P_ and rhs
    Z_ = (1/S_)**2*U_ @ U_.T @ rhs

    check_rhs = P@Z_

    print('Error l2 LR',np.linalg.norm(rhs-check_rhs)/np.linalg.norm(rhs))

    def fLR(x):

        phi_x = to.Tensor([psi(x[i]) for i in range(dim)])

        return ((phi_x@phi)@Z_)[0]


#===================== ALI TENSOR REGRESSION =============

ALID = True

if ALID:

    # tolerance in ALID
    #tau = 1e-5 # ok here
    tau = 0.15 # nice reduction
    tau = 0.12 # better
    tau = 0.1 # really nice
    tau = 0.075 # amazeball
    tau = 0.05

    # mask for kept data later for regression
    msk_rhs = [True]

    # initialize with first
    data_init = lhs[0]

    # build phi_ALI incrementally
    phi_ALI = to.Tensor([psi(data_init[i]) for i in range(dim)])

    # initialize cholesky factor [make it a 2D array]
    L = np.sqrt((phi_ALI@phi_ALI))

    t_list = []

    # stream like data loop
    for e in lhs[1:]:

        # evaluate phi(sample_i)
        phi_i = to.Tensor([psi(e[i]) for i in range(dim)])

        # vector
        k_i = (phi_ALI@phi_i)[:, 0]
        # scalar
        k_ii = (phi_i@phi_i)[0, 0]

        # evaluate coef minimizing dist in feature space
        s = solve_triangular(L, k_i, lower=True)
        a = solve_triangular(L.T, s)

        # evaluate dist in feature space
        dist = k_ii - k_i@a
        #print(f'dist {dist}')

        # depending on criterion keep or not [is phi_i ALD]
        if dist > tau:

            # keep track of  which samples to keep in rhs
            msk_rhs.append(True)

            # append phi_i to phi
            phi_ALI.append(phi_i.cores, axis=-1)

            # lower entry of L
            c_i = np.max([0., np.sqrt(k_ii-s.T@s)])

            # update cholesky factor
            L = np.vstack((np.c_[L, np.zeros(L.shape[0])], np.r_[s.T, c_i]))

        else: msk_rhs.append(False)

    # once best subset is identified based on given data, TREG
    m_ = phi_ALI.shape[0]
    print(m_)

    # rhs for weights computation
    Y = rhs[msk_rhs]

    # compute Z from P's Cholesky factor and rhs
    Z_ALI = solve_triangular(L.T, solve_triangular(L, Y, lower=True))

    # online computation
    def fALI(x):

        phi_x = to.Tensor([psi(x[i]) for i in range(dim)])

        return ((phi_x@phi_ALI)@Z_ALI)[0]

    check_rhs = phi@phi_ALI@Z_ALI

    # should be shit
    print('Error l2 ALI',np.linalg.norm(rhs-check_rhs)/np.linalg.norm(rhs))


print("================ VALIDATION ====================")

#===================== TEST RECONSTRUCTION =============

# number of tests
n_tests = 10000

# build LHS
lhs_test = LHS(n_samples=n_tests, n_dim=dim, bounds=param_bounds)

# build RHS
rhs_test = np.array([f(xi) for xi in lhs_test])

# build reconstruction vectors
if RIDGE :

    valid_R = np.array([fR(xi) for xi in lhs_test])
    #pdb.set_trace()
    print('Error l2 validation Ridge',np.linalg.norm(rhs_test-valid_R)/np.linalg.norm(rhs_test))
    plt.scatter(rhs_test, valid_R, label='R', marker='o')

if LR :

    valid_LR = np.array([fLR(xi) for xi in lhs_test])
    print('Error l2 validation LR',np.linalg.norm(rhs_test-valid_LR)/np.linalg.norm(rhs_test))
    plt.scatter(rhs_test, valid_LR, label='LR', marker='x')


if ALID :

    valid_ALID = np.array([fALI(xi) for xi in lhs_test])
    print('Error l2 validation ALI',np.linalg.norm(rhs_test-valid_ALID)/np.linalg.norm(rhs_test))
    plt.scatter(rhs_test, valid_ALID, label='ALID', marker='+')

plt.legend()
plt.show()
