"""
Rosenbrock benchmark to compare:
- M-Tensor regression:
    - lstsq
    - ST
    - ALD

    evaluate cinvergence with growing alpha
"""

# =========================================================
import numpy as np
import matplotlib.pyplot as plt
from time import process_time as _t
import mtensor as mt
from scipy.stats import qmc
import scipy.linalg as sp

# =========================================================
# CONTROL

DATA_GEN = False
BUILD = False
DIST_ALD = False
TEST = False
PLOT_TEST = False
PLOT = True

# =========================================================
# FIXED PARAMETERS

domain_1D = (-5., 10.)

alpha_list = [50, 100, 150, 200, 250] 

n_models = 5
n_tests = n_models

d = 100

# threshold
tau_ST =    1e-3
tau_sub =   2e-2

modelname = './save_m/model_'
resname = './save_m/result_errors'
datname = './save_m/data_'
imgname = './save_m/img_'

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

order_map = 4
sf = 1./(order_map*np.abs(max(domain_1D))*10**(order_map-1))

# default 1D mapping
def psi(x):

    return np.array([1.]+[sf*x**i for i in range(1, order_map+1)])

# default mapping
def cores(sample, map1D = psi):

    return [np.array([map1D(sample[k,i]) for k in range(len(sample))])
                for i in range(sample.shape[1])]

def cores_x(x, map1D = psi):

    return [map1D(x[i]) for i in range(len(x))]

# =========================================================
# MODELS

# M-Tensor streamed subsampling
def build_sub(sample, rhs, map1D = psi, threshold = 0, show_dist=False):

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
        dist = k_ii - k_i@a
        if show_dist: print(d, m, np.count_nonzero(msk_rhs), i, threshold, dist, end="\r", flush=True)

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

    return Z, np.array(msk_rhs)


# =========================================================
# DATA GEN

if DATA_GEN:

    for alpha in alpha_list:

        for name in ["model", "test"]:

            for i in range(n_models):

                m = d*alpha

                sampler = qmc.LatinHypercube(d=d)

                # m*d array
                sample = sampler.random(n=m)

                sample = qmc.scale(sample, d*[domain_1D[0]], d*[domain_1D[1]])

                # evaluate rosenbrock
                v = np.array([rosenbrock(x) for x in sample])

                # save each
                np.save(f'{datname}_{alpha}_{name}_{i}', np.c_[sample, v])

                print(f"Saved sample for {name} {i+1}/{n_models} for alpha = {alpha}!",
                      end="\r",
                      flush=True)

    exit()
 

# =========================================================

if BUILD:

    print("BUILDING MODELS FOR MTENSOR REGRESSION")
    print(f"List of \u03B1: {alpha_list}")

    # BENCHMARK ON M
    for iA in range(len(alpha_list)):

        alpha = alpha_list[iA]
        m = alpha*d

        # build n_trial models to average results
        for iM in range(n_models):
        #for iM in range(n_models-1, -1, -1):

            print(f">>> \u03B1 = {alpha}, m = {m}, model n°{iM+1}/{n_models}")

            # loading sample
            print('Loading sample...')
            dat = np.load(f'{datname}_{alpha}_model_{iM}.npy')
            print(f'shape of loaded data {dat.shape}')

            sample = dat[:, :d]
            rhs = dat[:, -1]
            
            phi = mt.MTensor(cores(sample))
            print("Tensor built!")

            P = phi@phi
            print("Gram matrix built!")

            S, U = sp.eigh(P) # manageable in 100+ dim
            print('SVD built!')
            # WARNING if scipy.linalg.eigh is used S are ordered inversely
            #print(f'eigdecomp P {_t()-t1}')
            S, U = S[::-1], U[:, ::-1]

            # === lstsq ===================================
            Z = ( U*(1/S) )@ U.T @ rhs # was lstsq, capitalize more on eigh
            print("Least squares built!")
            print(f"Error build lstsq: {error_l2(P@Z, rhs)}")

            np.save(f"{modelname}{alpha}_lstsq_{iM}", Z)
            print("Saved!")

            # === ST ======================================
            # thresholding
            r = np.argwhere(np.array([1 - np.sum(S[:i])/np.sum(S) for i in range(1,len(S)+1)]) < tau_ST)[0,0]
            U, S = U[:, :r+1], S[:r+1]
            print(f"Rank: {r}")
        
            Z = ( U*(1/S) )@ U.T @ rhs
            print("Spect. Trunc. built!")
            print(f"Error build ST: {error_l2(P@Z, rhs)}")

            np.save(f"{modelname}{alpha}_ST_{iM}", Z)
            print("Saved!")

            del U, S, P

            # === Sub =====================================
            Z, msk_rhs = build_sub(sample, rhs, threshold=tau_sub, show_dist=DIST_ALD)
            print('Subsampling built!')
            print(f"m tilde: {np.count_nonzero(msk_rhs)}")
            print(f"Error build sub: {error_l2((P[:, msk_rhs]) @ Z, rhs)}")

            np.savez(f"{modelname}{alpha}_sub_{iM}", Z=Z, msk=msk_rhs)
            print("Saved!")

    exit()

elif TEST:
    
    print("TESTING MODELS FOR MTENSOR REGRESSION")
    print(f"List of \u03B1: {alpha_list}")

    # store errors (once then update!)
    #errors = np.zeros((len(alpha_list), n_models, n_tests, 3))

    errors = np.load(resname+'.npy')
    n_tests=2

    # BENCHMARK ON M
    for iA in range(len(alpha_list)):

        alpha = alpha_list[iA]
        m = alpha*d

        # build n_trial models to average results
        for iM in range(n_models):

            print("\nLoading models...")
            sample = np.load(f'{datname}_{alpha}_model_{iM}.npy')[:, :-1]
            Z_lstsq = np.load(f"{modelname}{alpha}_lstsq_{iM}.npy")
            Z_ST = np.load(f"{modelname}{alpha}_ST_{iM}.npy")
            dat_sub = np.load(f"{modelname}{alpha}_sub_{iM}.npz")
            Z_sub, msk_rhs = dat_sub["Z"], dat_sub["msk"]
            print("Models loaded!")

            phi = mt.MTensor(cores(sample))
            print("Training tensor built!")

            for iT in range(n_tests):
                
                print(f">>> \u03B1 = {alpha}, m = {m}, model n°{iM+1}/{n_models}")
                print(f">>> test n°{iT+1}/{n_tests}")
 
                print("Loading sample for testing...")
                dat = np.load(f'{datname}_{alpha}_test_{iT}.npy')

                sample = dat[:, :d]
                v = dat[:, -1]
                print("Testing data loaded!")

                phi_test = mt.MTensor(cores(sample))
                print("Testing tensor built!")

                P = phi_test @ phi # this is the big part!
                print("Verif operator built!")

                # =============================================
                # Testing

                print("Testing...")
                
                v_lstsq = P @ Z_lstsq
                err_lstsq = error_l2(v_lstsq, v)
                print(f"Test error lstsq: {err_lstsq}")

                v_ST = P @ Z_ST
                err_ST = error_l2(v_ST, v)
                print(f"Test error Spect. Trunc.: {err_ST}")

                #v_sub = (phi_test @ phi[msk_rhs]) @ Z_sub
                v_sub = P[:, msk_rhs] @ Z_sub
                err_sub = error_l2(v_sub, v)
                print(f"Test error sub: {err_sub}")

                # store and save
                errors[iA, iM, iT] = [err_lstsq, err_ST, err_sub]
                np.save(resname, errors)
                print("Results saved!")

        # =============================================
        # plot intermediate
        if PLOT_TEST:

            facecolors = plt.colormaps['jet'](np.linspace(0, 1, 3))
            labels = ["Lstsq", "Spec. trunc.", "Subsample"]
            marks = ["o", "+", "x"]

            # low alpha all scattered points
            for i in range(3):

                for iM in range(n_models):

                    for iT in range(n_tests):

                        if i==0: plt.scatter(alpha_list, errors[:, iM, iT, i], c='None', edgecolors=facecolors[i], marker=marks[i], alpha=.3)
                        else: plt.scatter(alpha_list, errors[:, iM, iT, i], c=facecolors[i], marker=marks[i], alpha=.3)

            med_errors = np.median(errors.reshape((len(alpha_list), n_models*n_tests, 3)), axis=1)
            
            # median w/ plot
            for i in range(3):

                plt.plot(alpha_list, med_errors[:, i], marks[i], label=labels[i], ls="--", color=facecolors[i])

            plt.legend()
            plt.title("Approximation error vs. size of sample")
            plt.xlabel("Sample per axis \u03B1 [-]")
            plt.ylabel("Error [-]")
            plt.yscale("log")
            plt.tight_layout()

            plt.savefig(f"{imgname}{alpha}.pdf")

            plt.show()

elif PLOT:

    print("PLOTTING RESULTS")

    errors = np.load(resname+'.npy')
    print("Results loaded!")

    n_tests = 2

    # =============================================
    # plot intermediate

    facecolors = plt.colormaps['jet'](np.linspace(0, 1, 3))
    labels = ["Lstsq", "Spec. trunc.", "Subsample"]
    marks = ["o", "+", "x"]

    # low alpha all scattered points
    for i in range(3):

        for iM in range(n_models):

            for iT in range(n_tests):

                if i==0: plt.scatter(alpha_list, errors[:, iM, iT, i], color='None', edgecolors=facecolors[i], marker=marks[i], alpha=.3)
                else: plt.scatter(alpha_list, errors[:, iM, iT, i], color=facecolors[i], marker=marks[i], alpha=.3)

    med_errors = np.median(errors.reshape((len(alpha_list), n_models*n_tests, 3)), axis=1)
    
    # median w/ plot
    for i in range(3):

        plt.plot(alpha_list, med_errors[:, i], marks[i], label=labels[i], ls="--", color=facecolors[i])

    plt.legend()
    plt.title("Approximation error vs. size of sample")
    plt.xlabel("Sample per axis \u03B1 [-]")
    plt.ylabel("Error [-]")
    plt.yscale("log")
    plt.tight_layout()

    plt.savefig(f"{imgname}_final.pdf")

    plt.show()