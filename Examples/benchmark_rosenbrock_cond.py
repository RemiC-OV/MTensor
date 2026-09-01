"""
Rosenbrock benchmark to compare:
- M-Tensor regression:
    - lstsq
    - ST
    - ALD
- kernel reg
- kFSA

Evaluate scaling of 
- CPU time to build
- CPU time to evaluate (inference)
- error on test

this will require 3 plots with each method in.

Then evaluation of stability with condition in a table for 
varying order of regression at few dim. Effect of scaling factor as well ?

sample must be m x d
"""

# =========================================================
import numpy as np
import matplotlib.pyplot as plt
from time import process_time as _t
import mtensor as mt
from scipy.stats import qmc
import scipy.linalg as sp
from math import comb

# =========================================================
# CONTROL

DATA_GEN = False
RESAMPLE = False
CHECK_BUILD = False
DIST_ALD = False
SHOW_INFO = False # heavier
SHOW_INFO_2 = True # light


# =========================================================
# FIXED PARAMETERS

domain_1D = (-5., 10.)

alpha = 150 # might be a sweet sport to find here as well

n_trial = 3

d = 20

q_list = [10, 8, 5, 4, 3, 2]

# threshold d=20
tau_kFSA =  [5e26, 1e21, 5e12, 5e9, 1e-11, 1e-11]
tau_ST =    [1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5]
tau_sub =   [1e-9, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3]

# threshold d=10            ok>
#tau_kFSA =  [1e23, 1e18, 1e10, 1e4, 1e3, 1e2]
#tau_sub =   [1e-10, 1e-10, 1e-9, 1e-9, 1e-8, 1e-6]
#tau_ST =    [1e-16, 1e-14, 1e-12, 1e-11, 1e-9, 1e-8]

filename = "./save/result_q_d20_sig.npz"

# =========================================================
# FUNCTIONS

def rosenbrock(x):

    out = 0.
    
    for i in range(len(x)-1):

        out += 100.*(x[i+1]-x[i]**2)**2 + (x[i]-1.)**2
    
    return out 

# l2 error evaluation
def error_l2(v, v_true):

    return np.linalg.norm(v_true-v)/np.linalg.norm(v_true)

# build kernel matrix
def build_K(sample, kernel):

    m = len(sample)
    K = np.zeros((m,m))

    for i in range(m):

        x = sample[i]

        for j in range(i, m):

            y = sample[j]
            K[i,j] = kernel(x, y)

    return K + K.T - np.diag(K.diagonal())


# =========================================================
# MODELS

# kernel regression
def build_kreg(sample, rhs, kernel):

    K = build_K(sample, kernel)
    print('Operator Built')

    z = sp.solve(K, rhs, assume_a='sym')
    print('Solved Z!')

    if CHECK_BUILD: print(f'kreg build: {error_l2(K@z, rhs)}')

    def f(x):

        v = np.array([kernel(x, y) for y in sample])

        return v@z

    return f, np.linalg.cond(K), sp.eigvalsh(K,subset_by_index=[0,1])[0]


# M-Tensor least squares
def build_lstsq(sample, rhs, map1D):

    phi = mt.MTensor([np.array([map1D(sample[k,i]) for k in range(len(sample))]) for i in range(sample.shape[1])])
    print('Tensor built!')

    P = phi@phi
    print('operator built!')

    Z = sp.solve(P, rhs, assume_a='sym')
    print('Solved Z!')

    if CHECK_BUILD: print(f'MT_lstsq build: {error_l2(P@Z, rhs)}')

    def f(x):

        phi_x = mt.MTensor([map1D(x_i) for x_i in x])

        return ((phi_x@phi)@Z)[0]

    return f, np.linalg.cond(P), sp.eigvalsh(P ,subset_by_index=[0,1])[0]


# kFSA
def build_kFSA(sample, rhs, kernel, threshold = 0):

    d = sample.shape[1]
    tau = threshold

    # Compute Gram matrix
    K = build_K(sample, kernel)
    print('operator built')

    m = len(sample)

    left = np.arange(m)

    # first point
    maximum_distances = np.empty(m)

    for i in range(m):

        maximum_distances[i] = np.sum(
            K.diagonal() - K[:, i] ** 2 / K[i, i]
        )

    ind = np.argmin(maximum_distances)

    indices = [ind]

    left = left[left != ind]

    # Compute Z
    Z = np.linalg.solve(K[np.ix_(indices, indices)],K[np.ix_(indices, left)])

    while left.size > 0:

        # Approximation errors
        dist = (K.diagonal()[left] - np.sum(K[np.ix_(indices, left)] * Z, axis=0))

        if DIST_ALD: print(dist, '\n', d, m, tau, len(indices), len(left))

        # Maximum approximation error
        sub_index = np.argmax(dist)
        error = max(dist[sub_index], 0.0)
        ind = left[sub_index]

        # Remove samples with dist to current subspace <= threshold
        keep = np.where(dist > tau)[0]
        keep = keep[keep != sub_index]

        left = left[keep]

        if error > tau:

            # Add sample index
            indices.append(ind)

            # Update Z
            L = (K[ind, left] - K[ind, indices[:-1]] @ Z[:, keep]) / error

            Z = np.vstack([Z[:, keep] - np.outer(Z[:, sub_index], L), L])

    # here remove build K!!!
    #z = sp.solve(build_K(sample[indices], kernel), rhs[indices], assume_a='sym')
    z = sp.solve(K[np.ix_(indices, indices)], rhs[indices], assume_a='sym')
    print('Solved Z!')

    if CHECK_BUILD: print(f'kFSA build: {error_l2(K[:,indices]@z, rhs)}')
    print(f'm_tilde {len(indices)}')

    def f(x):

        v = np.array([kernel(x, y) for y in sample[indices]])

        return v@z

    return f, np.linalg.cond(K[np.ix_(indices, indices)]), sp.eigvalsh(K[np.ix_(indices, indices)],subset_by_index=[0,1])[0], len(indices)


# M-Tensor spectral truncation
def build_ST(sample, rhs, map1D, threshold = 0):

    #t1 = _t()
    phi = mt.MTensor([np.array([map1D(sample[k,i]) for k in range(len(sample))]) for i in range(sample.shape[1])])
    print("Tensor built!")
    #print(f'build tensor {_t()-t1}')

    #t1 = _t()
    P = phi@phi
    print("Operator built!")
    #print(f'build P {_t()-t1}')

    #t1 = _t()
    #U, S = np.linalg.svd(P)[:2]
    S, U = sp.eigh(P) # manageable in 100+ dim
    print('SVD built!')
    S, U = S[::-1], U[:, ::-1]
    # WARNING if scipy.linalg.eigh is used S are ordered inversely
    #print(f'eigdecomp P {_t()-t1}')

    # thresholding
    r = np.argwhere(np.array([1 - np.sum(S[:i])/np.sum(S) for i in range(1,len(S)+1)]) < threshold)[0,0]
    U, S = U[:, :r], S[:r]

    Z = U @ np.diag(1/S) @ U.T @ rhs
    print('Solved Z!')

    t1 = _t()
    if CHECK_BUILD: print(f'MT ST build: {error_l2(P@Z, rhs)}')
    if SHOW_INFO_2: print(f"cond: {S[0]/S[-1]}\nrank: {r}")
    t_check = _t() - t1

    def f(x):
    
        phi_x = mt.MTensor([map1D(x_i) for x_i in x])

        return ((phi_x@phi)@Z)[0]

    return f, S[0]/S[-1], S[-1], r


# M-Tensor streamed subsampling
def build_sub(sample, rhs, map1D, threshold = 0):

    # initialize mask
    msk_rhs = [True]

    # build phi_sub incrementally
    phi_sub = mt.MTensor([map1D(sample[0,i]) for i in range(sample.shape[1])])

    # initialize cholesky factor [make it a 2D array]
    L = np.sqrt((phi_sub@phi_sub))

    # stream like data loop
    for i in range(1,len(sample)):

        e = sample[i]

        # evaluate phi(sample_i)
        phi_i = mt.MTensor([map1D(e[i]) for i in range(sample.shape[1])])

        # vector
        k_i = (phi_sub@phi_i)[:, 0]
        # scalar
        k_ii = (phi_i@phi_i)[0, 0]

        # evaluate coef minimizing dist in feature space
        s = sp.solve_triangular(L, k_i, lower=True)
        a = sp.solve_triangular(L.T, s)

        # evaluate dist in feature space
        dist = np.max([0., k_ii - k_i@a])
        if DIST_ALD: print(d, m, np.count_nonzero(msk_rhs), i, threshold, dist)

        # depending on criterion keep or not [is phi_i ALD]
        if dist > threshold:

            # keep track of  which samples to keep in rhs
            msk_rhs.append(True)

            # append phi_i to phi
            phi_sub.append(phi_i.cores, axis=-1)

            # lower entry of L
            c_i = np.max([0., np.sqrt(k_ii-s.T@s)])

            # update cholesky factor
            L = np.vstack((np.c_[L, np.zeros(L.shape[0])], np.r_[s.T, c_i]))

        else: msk_rhs.append(False)

    Y = rhs[msk_rhs]

    Z = sp.solve_triangular(L.T, sp.solve_triangular(L, Y, lower=True))
    print('Z solved!')

    if CHECK_BUILD: print(f'MT sub build: {error_l2((mt.MTensor([np.array([map1D(sample[k,i]) for k in range(len(sample))]) for i in range(sample.shape[1])])@phi_sub)@Z, rhs)}')
    print(f'm_tilde: {len(Y)}')

    def f(x):
    
        phi_x = mt.MTensor([map1D(x_i) for x_i in x])

        return ((phi_x@phi_sub)@Z)[0]

    return f, np.linalg.cond(phi_sub@phi_sub), sp.eigvalsh(phi_sub@phi_sub,subset_by_index=[0,1])[0], len(Y)


# =========================================================
# TEST

def test(f, sample, true):

    t1 = _t()

    # compute approx on sample
    v = [f(e) for e in sample]

    # average inference time
    T_infer = (_t()-t1)/len(sample)

    return error_l2(np.array(v), true), T_infer
 
# =========================================================

# =========================================================
# DATA GEN

# assume of from other benchmark

# =========================================================
# STORAGE

"""
we need to store the 
- CPU building time
- CPU inference time
- error

median over trials
5 is nb of methods to test
"""
err_array = np.zeros((5, len(q_list), n_trial))
kappa_array = np.zeros((5, len(q_list), n_trial))
sigma_array = np.zeros((5, len(q_list), n_trial))
m_array = np.zeros((3, len(q_list), n_trial))

if __name__=="__main__":

    # =========================================================
    # BENCHMARK ON Dimension
    for iQ in range(len(q_list)):

        q = q_list[iQ]
        m = alpha*d

        order_map = q
        sf = 1./(order_map*np.abs(max(domain_1D))*10**(order_map-1))

        # default kernel function
        def kernel(x, y):

            return (1+x@y)**order_map

        # default 1D mapping
        def psi(x):

            return np.array([1.]+[sf*x**i for i in range(1, order_map+1)])


        print(f'\nd={d}, m={m}, q={q}')

        # build n_trial models to average results
        for iT in range(n_trial):

            print(f'>>> Trial {iT+1}/{n_trial}')

            # =============================================
            # Building sample

            print("Sampling for building...")

            if not RESAMPLE:

                print('Loading data...')
                dat = np.load(f'dat_rosenbrock_{d}.npy')
                print(f'shape of loaded data {dat.shape}')

                # each trial is a subset of the shuffled sample
                # here shuffle into train/test limitted by dim
                shfl = np.arange(dat.shape[0])
                np.random.shuffle(shfl)

                sample = dat[shfl][:m, :d]
                rhs = dat[shfl][:m, d]

            else:
        
                sampler = qmc.LatinHypercube(d=d)
            
                # m*d array
                sample = sampler.random(n=m)
            
                sample = qmc.scale(sample, d*[domain_1D[0]], d*[domain_1D[1]])

                rhs = np.array([rosenbrock(x) for x in sample])
            
            # =============================================
            # Building
            
            f_kFSA, cond_kfsa,sig_kfsa, m_kfsa = build_kFSA(sample, rhs, kernel, threshold=tau_kFSA[iQ])
            kappa_array[4, iQ, iT] = cond_kfsa
            sigma_array[4, iQ, iT] = sig_kfsa
            m_array[2, iQ, iT] = m_kfsa
            #input("Press Enter to continue...")

            t0 = _t()
            f_sub, cond_sub, sig_sub, m_sub = build_sub(sample, rhs, psi, threshold=tau_sub[iQ])
            kappa_array[3, iQ, iT] = cond_sub
            sigma_array[3, iQ, iT] = sig_sub
            m_array[1, iQ, iT] = m_sub
            #input("Press Enter to continue...")

            t0 = _t()
            f_ST, cond_ST, sig_ST, rnk_ST = build_ST(sample, rhs, psi, threshold=tau_ST[iQ])
            kappa_array[2, iQ, iT] = cond_ST
            sigma_array[2, iQ, iT] = sig_ST
            m_array[0, iQ, iT] = rnk_ST
            #input("Press Enter to continue...")

            t0 = _t()
            f_kreg, cond_kreg, sig_kreg = build_kreg(sample, rhs, kernel)
            kappa_array[0, iQ, iT] = cond_kreg
            sigma_array[0, iQ, iT] = sig_kreg
            
            t0 = _t()
            f_lstsq, cond_lstsq, sig_lstsq = build_lstsq(sample, rhs, psi)
            kappa_array[1, iQ, iT] = cond_lstsq
            sigma_array[1, iQ, iT] = sig_lstsq
            
            np.savez(filename,
                    err=err_array,
                    cond=kappa_array,
                    m=m_array)


            # =============================================
            # Testing sample

            print("Sampling for testing...")
            
            if not RESAMPLE:

                # each test is a subset of the shuffled sample
                # minus the building sample
                sample_test = dat[shfl][m:4*m, :d]
                true = dat[shfl][m:4*m, d]

            else:
        
                sampler = qmc.LatinHypercube(d=d)
            
                # m*d array
                sample_test = sampler.random(n=3*m)
            
                sample_test = qmc.scale(sample_test, d*[domain_1D[0]], d*[domain_1D[1]])

                true = np.array([rosenbrock(x) for x in sample_test])


            # =============================================
            # Testing

            print("Testing...")

            err_kreg, infer_kreg = test(f_kreg, sample_test, true)
            print(f"Error test kreg: {err_kreg}")
            err_kFSA, infer_kFSA = test(f_kFSA, sample_test, true)
            print(f"Error test kFSA: {err_kFSA}")
            err_lstsq, infer_lstsq = test(f_lstsq, sample_test, true)
            print(f"Error test lstsq: {err_lstsq}")
            err_ST, infer_ST = test(f_ST, sample_test, true)
            print(f"Error test ST: {err_ST}")
            err_sub, infer_sub = test(f_sub, sample_test, true)
            print(f"Error test sub: {err_sub}")

            err_array[:, iQ, iT] = np.array([err_kreg, err_lstsq, err_ST, err_sub, err_kFSA])
            
            np.savez(filename,
                    err=err_array,
                    cond=kappa_array,
                    m=m_array)

        #continue

        # =============================================
        # plot intermediate

        facecolors = plt.colormaps['jet'](np.linspace(0, 1, 5))
        labels = ["Kernel reg.", "MT - lstsq", "MT - spec. trunc.", "MT - subsample", "kFSA"]

        # average along axis 2 (n_trial)
        kappa_avg = np.average(kappa_array, axis=2)
        sig_avg = np.average(sigma_array, axis=2)
        err_array_avg = np.average(err_array, axis=2)
        # subplots
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Comparative benchmark Rosenbrock function')

        for i in range(5):

            # cond
            ax1.plot(q_list, sig_avg[i, :], "o", label=labels[i], ls="-", color=facecolors[i])
            
            # error l2
            ax2.plot(q_list, err_array_avg[i, :], "o", ls="-", color=facecolors[i])

        ax1.set(title="Lowest eigenvalue vs. mapping degree",
                xlabel="Degree [-]",
                ylabel="Lowest eigenvalue [-]",
                yscale="log")

        ax2.set(title="Test error vs. mapping degree",
                xlabel="Degree [-]",
                ylabel="Error [-]",
                yscale='log')

        fig.legend()
        plt.tight_layout()
        plt.show()

np.savez(filename,
        err=err_array,
        cond=kappa_array,
        m=m_array)