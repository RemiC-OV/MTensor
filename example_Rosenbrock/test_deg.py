
"""
R.CLOAREC
17-11-2025

High dimensional benchmark

plot the evolution of error wrt dimension
still based on Rosenbrock benchmark

plot x-axis (dimension) in log scale
check if logscale necessary in y-axis (error)

for each dimension draw LHS multiple times
for training and plot wrt n_sample by axis

plot resolution time wrt dimension as well
"""


# imports
import numpy as np
import tensoropV2 as to

from scipy.linalg import solve_triangular

import matplotlib.pyplot as plt

import time



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

# l2 error evaluation
def error_l2(v, v_true):

    return np.linalg.norm(v_true-v)/np.linalg.norm(v_true)

# max error evaluation
def error_max(v, v_true):

    return np.absolute(v_true - v).max()

# timer decorator
def timer_analysis(func):

    def wrapper():
                
        t1 = time.perf_counter()
        func()
        print(f'time: {time.perf_counter() - t1}s')

    return wrapper

t_init = 0.

def ti():

    global t_init

    t_init = time.perf_counter()

def tf():

    return time.perf_counter() - t_init

#==========================================================

# global parameters
domain_1D = (-5., 10.)

# solver choice (separate hyperparameter tuning)
RIDGE = True
LR = False
ALID = False

# QOI to track (output to be averaged over n_trial)
lst_err_l2 = []
lst_err_max = []

# list of tested dimensions
lst_dim = [2, 20, 50, 100, 200, 300]
lst_dim = [2, 200] # test 

# list of 1D sampling size
lst_m_axis = [5, 10, 20, 30, 40]
lst_m_axis = [10, 30] # test

# number of training and check to run
n_trial = 2

# hyperparameters per dim?
# lambda for Ridge
l = 0.
# rank for LR
rnk_s = [3, 20, 50, 100, 200, 300]
# scale factor for all methods, this should be defined computationally
# wrt amount of data and dimension, need to check
sf_s = [1e-3, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7]
sf_s = [1e-3, 1e-7] # test

# ALID tolerance is directly dependent to sf and problem dependent
tau_s = [1e-5, 5e-12, 1e-11, 1e-10, 1e-9, 1e-9]
tau_s = [1e-5, 1e-9] # test

# for deg 4 ok, otherwise need to update the scaling factor
# 5 gives 1e-6 error; 6 gives 1e-10 error
# 7 and above is fucked up, updated ALID, neeed to[ set lower sf]
deg_appx = 4

# number of QOI to store
# SET MANUALLY PER APPROX 
n_add_qoi = 2 if RIDGE else 3 
n_qoi = 5 + n_add_qoi

"""
We include in QOI
    online unit solving time (inference time)
    time to solve for Z
    error L2
    error max

"""

QOI = np.zeros((n_trial, len(lst_dim), len(lst_m_axis), n_qoi))
# will refer to elts as 
# QOI[iT, iD, iM, iQ] = qoi # where iQ only is manually set

PLOT_SCATTER = True
SHOW_SV = False
PRINT_DIST = False
OUTPUT = False

#==========================================================

## main loop
# dimension of the problem
for iD in range(len(lst_dim)):

    dim = lst_dim[iD]

    for iM in range(len(lst_m_axis)):

        m_axis = lst_m_axis[iM]

        # number of samples per axis
        m = m_axis*dim

        # define intervals for parameters
        param_bounds = dim*[domain_1D]

        # define the ref function
        def f(x):

            out = 0.

            for i in range(len(x)-1):

                out += 100.*(x[i+1]-x[i]**2)**2 + (x[i]-1.)**2
            
            return out 

        for iT in range(n_trial):

            
            print("================ TRIAL ====================")
            print(f'dim: {lst_dim[iD]}; m_axis: {lst_m_axis[iM]}; trial: {iT+1}; ')

            # build LHS
            lhs = LHS(n_samples=m, n_dim=dim, bounds=param_bounds)

            # build RHS
            ti()
            rhs = np.array([f(xi) for xi in lhs])
            print(f'time to build RHS: {tf()}s')

            #===================== TENSOR DEFINITION =============

            # update sf wrt dim
            sf = sf_s[iD]
            print(f'scale factor: {sf}')

            # define core functions
            #psi = lambda x: np.array([1., sf*x, sf*x**2, sf*x**3, sf*x**4])
            psi = lambda x: np.array([1.]+[sf*(x**i) for i in range(1, deg_appx+1)])

            # build m-tensor
            ti()
            phi = to.Tensor([np.array([psi(lhs[k, i]) for k in range(m)]) for i in range(dim)])
            print(f'time to build m-tensor: {tf()}s')

            #===================== RIDGE TENSOR REGRESSION =============

            if RIDGE:

                # build projector
                ti()
                P = phi@phi
                print(f'time to build P: {tf()}s')
                QOI[iT, iD, iM, 5] = tf()

                # compute vanilla Z from P and rhs
                ti()
                Z = np.linalg.lstsq(P + l*np.eye(m), rhs)[0]
                print(f'time to build Z: {tf()}s')
                QOI[iT, iD, iM, 6] = tf()

                # check RHS
                ti()
                check_rhs = P@Z
                print(f'time to build check_rhs: {tf()}s')


            #===================== LR TENSOR REGRESSION =============

            if LR:
                # need to do a PCA, SVD is not sufficient!!!

                # set rank
                rnk = rnk_s[iD]

                # test PCA
                ti()
                U, S = phi.pca()
                print(f'time to compmute PCA of P: {tf()}s')
                QOI[iT, iD, iM, 5] = tf()
                
                if SHOW_SV:

                    plt.plot(S, label='PCA')
                    plt.title(f'dim {lst_dim[iD]}')
                    plt.legend()
                    plt.show()

                U_, S_ = U[:, :rnk], S[:rnk]
                print(f'rank: {rnk}')

                # compute Z from P_ and rhs
                ti()
                Z = (1/S_)**2*U_ @ U_.T @ rhs
                print(f'time to build Z: {tf()}s')
                QOI[iT, iD, iM, 6] = tf()

                # check RHS
                ti()
                check_rhs = phi@phi@Z
                print(f'time to build check_rhs: {tf()}s')


            #================ GREEDY ALI TENSOR REGRESSION ==========

            if ALID:

                tau = tau_s[iD]

                # tolerance in ALID
                print(f'ALID tol: {tau}')

                # init time ALID
                ti()

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
                    if PRINT_DIST : print(f'dist {dist}')

                    # depending on criterion keep or not [is phi_i ALD]
                    if dist > tau:

                        # keep track of  which samples to keep in rhs
                        msk_rhs.append(True)

                        # append phi_i to phi
                        phi_ALI.append(phi_i.cores, axis=-1)

                        # lower entry of L
                        #c_i = np.max([0., np.sqrt(k_ii-s.T@s)])
                        # we test sthing to avoid negative randicand
                        c_i = np.sqrt(np.max([0., np.sqrt(k_ii-s.T@s)]))# why ?!!!
                        c_i = np.sqrt(np.max([0., k_ii-s.T@s])) # test this one instead ?

                        # update cholesky factor
                        L = np.vstack((np.c_[L, np.zeros(L.shape[0])], np.r_[s.T, c_i]))

                    else: msk_rhs.append(False)

                print(f'time for greedy ALID: {tf()}s')
                QOI[iT, iD, iM, 5] = tf()

                # once best subset is identified based on given data, TREG
                m_ = phi_ALI.shape[0]
                print(f'ALID final size {m_}')
                QOI[iT, iD, iM, 6] = m_

                # rhs for weights computation
                Y = rhs[msk_rhs]

                # compute Z from P's Cholesky factor and rhs
                ti()
                Z = solve_triangular(L.T, solve_triangular(L, Y, lower=True))
                print(f'time to build Z: {tf()}s')
                QOI[iT, iD, iM, 7] = tf()

                # check RHS
                ti()
                check_rhs = phi@phi_ALI@Z
                print(f'time to build check_rhs: {tf()}s')
                
        
            #===================== COMMON OFFLINE DEFS ==============

            # print errors
            e2, emax = error_l2(rhs, check_rhs), error_max(rhs, check_rhs)
            print('Error on training sample (l2)', e2)
            print('Error on training sample (max)', emax)
            QOI[iT, iD, iM, 0] = e2
            QOI[iT, iD, iM, 1] = emax

            #if e2 > 0.1:

            #    exit()

            # define reconstruction function
            def f_(x):

                global ALID

                phi_x = to.Tensor([psi(x[i]) for i in range(dim)])

                if ALID:

                    return ((phi_x@phi_ALI)@Z)[0]

                return ((phi_x@phi)@Z)[0]


            print("================ VALIDATION ====================")
            print(f'dim: {lst_dim[iD]}; m_axis: {lst_m_axis[iM]}; trial: {iT+1}; ')

            # number of tests
            n_tests = m*3

            print(f'Building testing sample of {n_tests} points')

            # build LHS
            lhs_test = LHS(n_samples=n_tests, n_dim=dim, bounds=param_bounds)

            # build RHS
            rhs_test = np.array([f(xi) for xi in lhs_test])

            # build reconstruction vectors
            t1 = time.perf_counter()

            valid = np.array([f_(xi) for xi in lhs_test])
            
            print(f'time to solve: {time.perf_counter() - t1}s')
            print(f'avg unit inference time: {(time.perf_counter() - t1)/n_tests}s')
            QOI[iT, iD, iM, 4] = (time.perf_counter() - t1)/n_tests
            
            e2_v, emax_v = error_l2(rhs_test, valid), error_max(rhs_test, valid)

            print('Error on validation sample (l2)', e2_v)
            print('Error on validation sample (max)', emax_v)
            
            #if e2_v > 0.2:

            #    exit()

            QOI[iT, iD, iM, 2] = e2_v
            QOI[iT, iD, iM, 3] = emax_v

            if PLOT_SCATTER:

                plt.scatter(rhs_test, valid)
                plt.title(f'dim: {lst_dim[iD]}; m_axis: {lst_m_axis[iM]}; trial: {iT+1}; ')
                plt.show()

#=============== out of loop ================
"""
here we plot a surface evaluating error for
(dim, sample_size)

therefore require 

"""
if not OUTPUT: exit()

# output QOI
if RIDGE: np.save('./saved_QOI/ridge_QOI.npy', QOI)
if LR : np.save('./saved_QOI/lr_QOI.npy', QOI)
if ALID : np.save('./saved_QOI/alid_QOI.npy', QOI)