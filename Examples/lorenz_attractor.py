"""
================
Lorenz attractor
================

Author: R.CLOAREC

Example of plotting 3D Lorenz 63
Deterministic Nonperiodic Flow

journals.ametsoc.org/view/journals/atsc/20/2/1520-0469_1963_020_0130_dnf_2_0_co_2.xml

Dynamical system learning based on MTensor format

See refs:
[kRLS]  Engel et al.  (10.1109/TSP.2004.830985)
[kFSA]  Gelss et al.  (10.1016/j.knosys.2021.106935)
[LANDO] Baddoo et al. (10.1098/rspa.2021.0830)
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_triangular

import mtensor as mt


DIST = False
SAVEFIGS = False

#============ UTILS ================================

# exact Lorenz system
def lorenz(xyz, *, s=10, r=28, b=2.667):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z

    return np.array([x_dot, y_dot, z_dot])

init_pos = (0., 1., 1.05)

#=============== BUILD EXACT LORENZ ========================

# timestep and number of steps
# sampling strategy is baked in the construction of the Lorenz
# attractor through num_steps

dt = 0.01
dt = 0.001
num_steps = 500

# initialize position output
xyzs = np.empty((num_steps + 1, 3))  # Need one more for the initial values

# set initial position
xyzs[0] = init_pos

# initialize derivatives output
xyzsdt = np.empty((num_steps, 3))

# Step through time, calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    
    xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt
    xyzsdt[i] = lorenz(xyzs[i])


ax = plt.figure()
ax.add_subplot(projection='3d')

# Plot exact Lorenz system
def plot_lorenz(X, Y, Z, label=None):

   plt.plot(X, Y, Z, lw=0.5, linestyle='--', label=label if label else 'Lorenz')


#plt.plot(*xyzs.T, label='real', lw=0.5, linestyle='-', color='k')

#===================== FORMAT DATA ================

X = xyzs[:-1,0]
Y = xyzs[:-1,1]
Z = xyzs[:-1,2]

#===================== BUILD TENSOR REGRESSION =============

M = np.c_[X, Y, Z]

m = num_steps

psi = lambda x: np.array([1., x])

phi = mt.MTensor([np.array([psi(M[k, i]) for k in range(m)]) for i in range(3)])

#tmp_mtx = 1/m*np.ones((m,m))
P = phi@phi
#P = P - 2*tmp_mtx@P + tmp_mtx@P@tmp_mtx

# decompose P
U, S = np.linalg.svd(P)[:2]
S = np.sqrt(S)

print(S)

# regul parameter
l = 1e-8
l = 0

# compute Z from P and rhs
Z = np.linalg.lstsq(P + l*np.eye(m), xyzsdt)[0]

check_rhs = phi@phi@Z

# should be shit
print('Error l2 new',np.linalg.norm(xyzsdt-check_rhs)/np.linalg.norm(xyzsdt))

def f(x, y, z):

   phi_x = mt.MTensor([psi(x), psi(y), psi(z)])

   return ((phi_x@phi)@Z)[0]

#===================== BUILD LR TENSOR REGRESSION =============

rnk = 8
rnk = 15
U_, S_ = U[:, :rnk], S[:rnk]
#P_ = (S_*U_) @ (S_*U_).T

# test linearization by regressing
# compute Z from P_ and rhs
#Z_ = (1/S_)**2*U_ @ U_.T @ xyzsdt # nope
#Z_ = U @ np.c_[U_, np.zeros((m, m-rnk))].T @ np.linalg.lstsq(P_, xyzsdt)[0]# nope
#Z_ = U @ np.c_[U_, np.zeros((m, m-rnk))].T @ np.linalg.lstsq(P, xyzsdt)[0]
Z_ = U_ @ U_.T @ np.linalg.lstsq(P, xyzsdt)[0]

check_rhs = phi@phi@Z_

# should be shit
print('Error l2 LR',np.linalg.norm(xyzsdt-check_rhs)/np.linalg.norm(xyzsdt))

def f_(x, y, z):

   phi_x = mt.MTensor([psi(x), psi(y), psi(z)])

   return ((phi_x@phi)@Z_)[0]

# ===================== SUBSAMPLING ALID ============

"""
apply ALID approach to select the best fitting sample

this is the one minimizing all distances in feature space

then determine distances of all samples wrt first sample
select new amples up until precision criterion is reached
use cholecky decomposition of phi@phi for this step

iterative scheme for selection, need to evaluate influence
of the criterion on the model (speed vs precision) has this study been done ?
we are avaluating error in the feature space, how does it translate to real space
note that we still require computation of the least squares solutin to
go back to data-space

this las step can be regularized as well... how do errors add up?

# TODO: test the on-the-fly method first
ie compute the model when data is streamed

"""

print(f'{' Here ALID ':=^80}')

import time

tau = 1.
data = M
dim = data.shape[1]
m = data.shape[0]

# mask for kept data latr for regression
msk_rhs = [True]

# initialize with first
data_init = data[0]

phi_alid = mt.MTensor([psi(data_init[i]) for i in range(dim)])

# initialize cholesky factor [make it a 2D array]
L = np.sqrt((phi_alid@phi_alid))

t_list = []
t_0 = time.time()

# stream like data loop
for sample in data[1:]:

   # evaluate phi(sample_i)
   phi_i = mt.MTensor([psi(sample[i]) for i in range(dim)])

   # vector
   k_i = (phi_alid@phi_i)[:, 0]
   # scalar
   k_ii = (phi_i@phi_i)[0, 0]

   # evaluate coef minimizing dist in feature space
   s = solve_triangular(L, k_i, lower=True)
   a = solve_triangular(L.T, s)

   # evaluate dist in feature space
   dist = k_ii - k_i@a
   if DIST : print(f'dist {dist}')

   # depending on criterion keep or not [is phi_i ALD]
   if dist > tau:

      # keep track of  which samples to keep in rhs
      msk_rhs.append(True)

      # append phi_i to phi
      phi_alid.append(phi_i.cores, axis=-1)

      # lower entry of L
      c_i = np.max([0., np.sqrt(k_ii-s.T@s)])
      #print(s, c_i)

      # update cholesky factor
      L = np.vstack((np.c_[L, np.zeros(L.shape[0])], np.r_[s.T, c_i]))

   else: msk_rhs.append(False)

   t_u = time.time() - t_0
   t_list.append(t_u)

#print(t_list)

# once best subset is identified based on given data, TREG
m_ = phi_alid.shape[0]

# regul parameter
l = 0.#1e-8

# rhs for weights computation
Y = xyzsdt[msk_rhs]

# timing of solving for weights
t1 = time.perf_counter()

# define projector for ALID (necessary single cost exclusive to this approach)
P_alid = phi_alid@phi_alid

# compute Z from P and rhs
#Z_alid_ = np.linalg.lstsq(P_alid + l*np.eye(m_), Y)[0]
Z_alid_ = np.linalg.lstsq(P_alid, Y)[0]
print(f'time for solving using full projector: {time.perf_counter() - t1}s')

# note on condition number
print('condition number of ALID-style projector',np.linalg.cond(P_alid))

# timing for solving using cholesky factorization
t1 = time.perf_counter()

# change this solving method and use L instead THEN how to regularize ?
Z_alid = np.linalg.solve(L.T, np.linalg.solve(L, Y))
print(f'time for solving using cholesky factor: {time.perf_counter() - t1}s')

"""
NOTE: here numpy.linalg.solve uses _gesv from LAPACK
which uses LU decomposition, better would beto use real backsubstitution
evaluation of high precision time to perform reveals that cholesky approach is 
twice faster than least squares
"""

print('difference between Z_alid lstsq - cholesky')
print(Z_alid_ - Z_alid)

print(f'Shapes for ALID system: phi_alid: {phi_alid.shape}, Z_alid: {Z_alid.shape}')

# online computation
def f_alid(x, y, z):

   phi_x = mt.MTensor([psi(x), psi(y), psi(z)])

   return ((phi_x@phi_alid)@Z_alid)[0]


check_rhs = phi@phi_alid@Z_alid

# should be shit
print('Error l2 ALID',np.linalg.norm(xyzsdt-check_rhs)/np.linalg.norm(xyzsdt))

# try to express the coefficients
def prod(lst):

   out = 1.

   for e in lst:

      out *= e

   return out

full_phi_alid = phi_alid.full().reshape(phi_alid.shape[0], int(prod(phi_alid.shape[1:])))

coefs = full_phi_alid.T @ Z_alid
print('coefs')
print(coefs)

print('Z_alid')
print(Z_alid)

print('full phi_alid')
print(full_phi_alid)

#===================== BUILD TENSOR REGRESSION =============
# PLOT TEST

num_steps_test = int(50*num_steps)
#num_steps_test = 20000

# ------------ REAL --------------

# initialize position output
xyzs = np.empty((num_steps_test + 1, 3))  # Need one more for the initial values

# set initial position
xyzs[0] = init_pos

# initialize derivatives output
xyzsdt = np.empty((num_steps_test, 3))


# timing for construction of trajectory
t1 = time.perf_counter()

# Step through time, calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps_test):
    
    xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt
    xyzsdt[i] = lorenz(xyzs[i])

# timer for reconstruction
print(f'time reconstructing REAL trajectory: {time.perf_counter() - t1}s')
print(f'ie unitary cost for f_: {(time.perf_counter() - t1)/num_steps_test}s')


plt.plot(*xyzs.T, label='real', lw=0.5, linestyle='-', color='k')


# ------------ TENSOR FULL RANK --------------

# re-generate data from initial point using tensor integrator
xyzs_ = np.empty((num_steps_test + 1, 3))  # Need one more for the initial values

# set initial position
xyzs_[0] = init_pos

# initialize derivatives output
xyzsdt_ = np.empty((num_steps_test, 3))


# timing for construction of trajectory
t1 = time.perf_counter()

# Step through time, calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps_test):
    
    xyzs_[i + 1] = xyzs_[i] + f(*xyzs_[i]) * dt

# timer for reconstruction
print(f'time reconstructing TENSOR FULL RANK trajectory: {time.perf_counter() - t1}s')
print(f'ie unitary cost for f_: {(time.perf_counter() - t1)/num_steps_test}s')


plot_lorenz(*xyzs_.T, label='Least squares')


# ------------ TENSOR LOW RANK --------------

# re-generate data from initial point using tensor integrator
xyzs_r = np.empty((num_steps_test + 1, 3))  # Need one more for the initial values

# set initial position
xyzs_r[0] = init_pos

# initialize derivatives output
xyzsdt_r = np.empty((num_steps_test, 3))

# timing for construction of trajectory
t1 = time.perf_counter()

# Step through time, calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps_test):
    
    xyzs_r[i + 1] = xyzs_r[i] + f_(*xyzs_r[i]) * dt

# timer for reconstruction
print(f'time reconstructing TENSOR LOW RANK trajectory: {time.perf_counter() - t1}s')
print(f'ie unitary cost for f_: {(time.perf_counter() - t1)/num_steps_test}s')

plot_lorenz(*xyzs_r.T, label='Spec. trunc.')


# ------------ TENSOR ALID SUBSAMPLING --------------

# re-generate data from initial point using tensor integrator
xyzs_l = np.empty((num_steps_test + 1, 3))  # Need one more for the initial values

# set initial position
xyzs_l[0] = init_pos

# initialize derivatives output
xyzsdt_l = np.empty((num_steps_test, 3))

# timing for construction of trajectory
t1 = time.perf_counter()

# Step through time, calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps_test):
    
    xyzs_l[i + 1] = xyzs_l[i] + f_alid(*xyzs_l[i]) * dt

# timer for reconstruction
print(f'time reconstructing TENSOR ALID trajectory: {time.perf_counter() - t1}s')
print(f'ie unitary cost for f_: {(time.perf_counter() - t1)/num_steps_test}s')

plot_lorenz(*xyzs_l.T, label='ALI reg.')

plt.legend()
if SAVEFIGS : plt.savefig('img_lorenz/lorenz_traj.pdf', bbox_inches='tight')
plt.show()

# 
delta_full_rank = np.linalg.norm(xyzs_ - xyzs, axis=1)/np.linalg.norm(xyzs, axis=1)
delta_low_rank = np.linalg.norm(xyzs_r - xyzs, axis=1)/np.linalg.norm(xyzs, axis=1)
delta_alid = np.linalg.norm(xyzs_l - xyzs, axis=1)/np.linalg.norm(xyzs, axis=1)

plt.plot(range(num_steps_test+1), delta_full_rank, label='full rank')
plt.plot(range(num_steps_test+1), delta_low_rank, label='low rank', linestyle='--')
plt.plot(range(num_steps_test+1), delta_alid, label='ALI regul')

plt.legend()
plt.yscale('log')
plt.xlabel('Timestep')
plt.ylabel('L2 error')
plt.grid(True, which='both')
if SAVEFIGS :  plt.savefig('img_lorenz/lorenz_err_ref.pdf', bbox_inches='tight')
plt.show()

#plt.plot(range(num_steps_test+1), xyzs.T[0])
#plt.show()

#exit()

# ============================================================================

print(f'{'-- TESTS --':^60}')

TEST_PLOT = False
TEST_PLOT_2 = False

n_tests = 50

# run with 50 times then replot with limitted window
num_steps_test = int(n_tests*num_steps)
#num_steps_test = 15000

# store for min-max-avg fill between plot
curves = np.zeros((n_tests, num_steps_test+1))

for it_test in range(n_tests):

   if TEST_PLOT:
      ax = plt.figure()
      ax.add_subplot(projection='3d')

   # test with other initial positions
   init_pos = np.random.random(3)

   # initialize position output
   xyzs = np.empty((num_steps_test + 1, 3))  # Need one more for the initial values

   # set initial position
   xyzs[0] = init_pos

   # initialize derivatives output
   xyzsdt = np.empty((num_steps_test, 3))

   # Step through time, calculating the partial derivatives at the current point
   # and using them to estimate the next point
   for i in range(num_steps_test):
      
      xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt
      xyzsdt[i] = lorenz(xyzs[i])

   if TEST_PLOT: plt.plot(*xyzs.T, label='real', lw=0.5, linestyle='-', color='k')

   # re-generate data from initial point using tensor integrator
   xyzs_ = np.empty((num_steps_test + 1, 3))  # Need one more for the initial values

   # set initial position
   xyzs_[0] = init_pos

   # initialize derivatives output
   xyzsdt_ = np.empty((num_steps_test, 3))

   # Step through time, calculating the partial derivatives at the current point
   # and using them to estimate the next point
   for i in range(num_steps_test):
      
      #xyzs_[i + 1] = xyzs_[i] + f(*xyzs_[i]) * dt
      xyzs_[i + 1] = xyzs_[i] + f_alid(*xyzs_[i]) * dt

   #if TEST_PLOT: plot_lorenz(*xyzs_.T, label='full rank TREG - integrated')
   if TEST_PLOT: plot_lorenz(*xyzs_.T, label='ALID TREG - integrated')

   # re-generate data from initial point using tensor integrator
   xyzs_r = np.empty((num_steps_test + 1, 3))  # Need one more for the initial values

   # set initial position
   xyzs_r[0] = init_pos

   # initialize derivatives output
   xyzsdt_r = np.empty((num_steps_test, 3))

   # Step through time, calculating the partial derivatives at the current point
   # and using them to estimate the next point
   for i in range(num_steps_test):
      
      xyzs_r[i + 1] = xyzs_r[i] + f_(*xyzs_r[i]) * dt

   if TEST_PLOT: plot_lorenz(*xyzs_r.T, label='LR TREG - integrated')

   if TEST_PLOT: 
      plt.legend()
      plt.show()


   # NAME MISLEADING, IT IS ALID
   # 
   delta_full_rank = np.linalg.norm(xyzs_ - xyzs, axis=1)/np.linalg.norm(xyzs, axis=1)
   #delta_low_rank = np.linalg.norm(xyzs_r - xyzs, axis=1)/np.linalg.norm(xyzs, axis=1)

   curves[it_test] = delta_full_rank

   if TEST_PLOT_2: 

      plt.plot(range(num_steps_test+1), delta_full_rank, label=f'full rank {it_test}')
      #plt.plot(range(num_steps_test+1), delta_low_rank, label=f'low rank {it_test}')


#plt.legend()
plt.fill_between(range(num_steps_test+1), curves.max(axis=0), curves.min(axis=0), alpha = .5, linewidth=0, color='green')

plt.plot(range(num_steps_test+1), curves.sum(axis=0)/n_tests, color='green')

plt.yscale('log')
plt.xlabel('Timestep')
plt.ylabel('L2 error')
plt.grid(True, which='both')
if SAVEFIGS : plt.savefig('img_lorenz/lorenz_test_ok.pdf', bbox_inches='tight')
plt.show()