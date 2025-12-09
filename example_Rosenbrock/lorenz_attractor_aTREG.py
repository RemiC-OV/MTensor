"""
================
Lorenz attractor
================

This is an example of plotting Edward Lorenz's 1963 `"Deterministic Nonperiodic
Flow"`_ in a 3-dimensional space using mplot3d.

.. _"Deterministic Nonperiodic Flow":
   https://journals.ametsoc.org/view/journals/atsc/20/2/1520-0469_1963_020_0130_dnf_2_0_co_2.xml

.. note::
   Because this is a simple non-linear ODE, it would be more easily done using
   SciPy's ODE solver, but this approach depends only upon NumPy.
"""

import matplotlib.pyplot as plt
import numpy as np

import tensorop as to

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

# basis functions
def basis(x, order=2):
    
    return np.array([x**i for i in range(order+1)])




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

#===================== BUILD TENSOR REGRESSION (full) ================

X = xyzs[:-1,0]
Y = xyzs[:-1,1]
Z = xyzs[:-1,2]

#===================== BUILD TENSOR REGRESSION =============

M = np.c_[X, Y, Z]

m = num_steps

psi = lambda x: np.array([1., x])
#psi = lambda x: np.array([1., x, 1e-3*x**2, 1e-6*x**3, 1e-9*x**4])

phi = to.Tensor([np.array([psi(M[k, i]) for k in range(m)]) for i in range(3)])

tmp_mtx = 1/m*np.ones((m,m))
P = phi@phi
P = P - 2*tmp_mtx@P + tmp_mtx@P@tmp_mtx

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

   phi_x = to.Tensor([psi(x), psi(y), psi(z)])

   return ((phi_x@phi)@Z)[0]

#===================== BUILD TENSOR REGRESSION =============

rnk = 8
rnk = 15
U_, S_ = U[:, :rnk], S[:rnk]
P_ = (S_*U_) @ (S_*U_).T

# test linearization by regressing
# compute Z from P_ and rhs
Z_ = (1/S_)**2*U_ @ U_.T @ xyzsdt # nope
Z_ = U @ np.c_[U_, np.zeros((m, m-rnk))].T @ np.linalg.lstsq(P_, xyzsdt)[0]# nope
Z_ = U @ np.c_[U_, np.zeros((m, m-rnk))].T @ np.linalg.lstsq(P, xyzsdt)[0]
Z_ = U_ @ U_.T @ np.linalg.lstsq(P, xyzsdt)[0]

check_rhs = phi@phi@Z_

# should be shit
print('Error l2 LR',np.linalg.norm(xyzsdt-check_rhs)/np.linalg.norm(xyzsdt))

def f_(x, y, z):

   phi_x = to.Tensor([psi(x), psi(y), psi(z)])

   return ((phi_x@phi)@Z_)[0]

# ===================== SUBSAMPLING LANDO-LIKE ============

"""
apply LANDO approach to select the best fitting sample

this is the one minimizing all distances in feature space

then determine distances of all samples wrt first sample
select new amples up until precisio criterio is reached
use cholecky decomposition of phi@phi for this step

iterative scheme fo rselection, need to evaluate influence
of the criterion on the model (speed vs precision) has this study been done ?
we are avaluating error in the feature space, how does it translate to real space
note that we still require computation of the least squares solutin to
go back to data-space

this las step can be regularized as well... how do errors add up?

# TODO: test the on-the-fly method first
compute the model when data is streamed

"""

print(f'{' Here LANDO ':=^80}')

import time

tau = 1.
data = M
dim = data.shape[1]
m = data.shape[0]

# mask for kept data latr for regression
msk_rhs = [True]

# initialize with first
data_init = data[0]

phi_lando = to.Tensor([psi(data_init[i]) for i in range(dim)])

# initialize cholesky factor [make it a 2D array]
L = np.sqrt((phi_lando@phi_lando))

t_list = []
t_0 = time.time()

# stream like data loop
for sample in data[1:]:

   # evaluate phi(sample_i)
   phi_i = to.Tensor([psi(sample[i]) for i in range(dim)])

   # vector
   k_i = (phi_lando@phi_i)[:, 0]
   # scalar
   k_ii = (phi_i@phi_i)[0, 0]

   # evaluate coef minimizing dist in feature space
   # np.linalg.solve will use cholesky automatically
   s = np.linalg.solve(L, k_i)
   a = np.linalg.solve(L.T, s)

   # evaluate dist in feature space
   dist = k_ii - k_i@a
   #print(f'dist {dist}')

   # depending on criterion keep or not [is phi_i ALD]
   if dist > tau:

      # keep track of  which samples to keep in rhs
      msk_rhs.append(True)

      # append phi_i to phi
      phi_lando.append(phi_i.cores, axis=-1)

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
m_ = phi_lando.shape[0]

# regul parameter
l = 0.#1e-8

# rhs for weights computation
Y = xyzsdt[msk_rhs]

# timing of solving for weights
t1 = time.perf_counter()

# define projector for LANDO (necessary single cost exclusive to this approach)
P_lando = phi_lando@phi_lando

# compute Z from P and rhs
#Z_lando_ = np.linalg.lstsq(P_lando + l*np.eye(m_), Y)[0]
Z_lando_ = np.linalg.lstsq(P_lando, Y)[0]
print(f'time for solving using full projector: {time.perf_counter() - t1}s')

# note on condition number
print('condition number of LANDO-style projector',np.linalg.cond(P_lando))

# timing for solving using cholesky factorization
t1 = time.perf_counter()

# change this solving method and use L instead THEN how to regularize ?
Z_lando = np.linalg.solve(L.T, np.linalg.solve(L, Y))
print(f'time for solving using cholesky factor: {time.perf_counter() - t1}s')

"""
NOTE: here numpy.linalg.solve uses _gesv from LAPACK
which uses LU decomposition, better would beto use real backsubstitution
evaluation of high precision time to perform reveals that cholesky approach is 
twice faster than least squares
"""

print('difference between Z_lando lstsq - cholesky')
print(Z_lando_ - Z_lando)

print(f'Shapes for LANDO system: phi_lando: {phi_lando.shape}, Z_lando: {Z_lando.shape}')

# online computation
def f_lando(x, y, z):

   phi_x = to.Tensor([psi(x), psi(y), psi(z)])

   return ((phi_x@phi_lando)@Z_lando)[0]


check_rhs = phi@phi_lando@Z_lando

# should be shit
print('Error l2 LANDO',np.linalg.norm(xyzsdt-check_rhs)/np.linalg.norm(xyzsdt))

# try to express the coefficients
def prod(lst):

   out = 1.

   for e in lst:

      out *= e

   return out

full_phi_lando = phi_lando.full().reshape(phi_lando.shape[0], int(prod(phi_lando.shape[1:])))

coefs = full_phi_lando.T @ Z_lando
print('coefs')
print(coefs)

print('Z_lando')
print(Z_lando)

print('full phi_lando')
print(full_phi_lando)

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


# ------------ TENSOR LANDO SUBSAMPLING --------------

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
    
    xyzs_l[i + 1] = xyzs_l[i] + f_lando(*xyzs_l[i]) * dt

# timer for reconstruction
print(f'time reconstructing TENSOR LANDO trajectory: {time.perf_counter() - t1}s')
print(f'ie unitary cost for f_: {(time.perf_counter() - t1)/num_steps_test}s')

plot_lorenz(*xyzs_l.T, label='ALI reg.')

plt.legend()
plt.savefig('img_lorenz/lorenz_traj.pdf', bbox_inches='tight')
plt.show()

# 
delta_full_rank = np.linalg.norm(xyzs_ - xyzs, axis=1)/np.linalg.norm(xyzs, axis=1)
delta_low_rank = np.linalg.norm(xyzs_r - xyzs, axis=1)/np.linalg.norm(xyzs, axis=1)
delta_lando = np.linalg.norm(xyzs_l - xyzs, axis=1)/np.linalg.norm(xyzs, axis=1)

plt.plot(range(num_steps_test+1), delta_full_rank, label='full rank')
plt.plot(range(num_steps_test+1), delta_low_rank, label='low rank', linestyle='--')
plt.plot(range(num_steps_test+1), delta_lando, label='ALI regul')

plt.legend()
plt.yscale('log')
plt.xlabel('Timestep')
plt.ylabel('L2 error')
plt.grid(True, which='both')
plt.savefig('img_lorenz/lorenz_err_ref.pdf', bbox_inches='tight')
plt.show()

#plt.plot(range(num_steps_test+1), xyzs.T[0])
#plt.show()

#exit()

# ============================================================================

print(f'{'-- TESTS --':^60}')

TEST_PLOT = False
TEST_PLOT_2 = True

for it_test in range(50):

   if TEST_PLOT:
      ax = plt.figure()
      ax.add_subplot(projection='3d')

   # test with other initial positions
   init_pos = np.random.random(3)

   # run with 50 times then replot with limitted window
   num_steps_test = int(50*num_steps)
   #num_steps_test = 15000

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
      xyzs_[i + 1] = xyzs_[i] + f_lando(*xyzs_[i]) * dt

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


   # 
   delta_full_rank = np.linalg.norm(xyzs_ - xyzs, axis=1)/np.linalg.norm(xyzs, axis=1)
   #delta_low_rank = np.linalg.norm(xyzs_r - xyzs, axis=1)/np.linalg.norm(xyzs, axis=1)

   if TEST_PLOT_2: 

      plt.plot(range(num_steps_test+1), delta_full_rank, label=f'full rank {it_test}')
      #plt.plot(range(num_steps_test+1), delta_low_rank, label=f'low rank {it_test}')

#plt.legend()
plt.yscale('log')
plt.xlabel('Timestep')
plt.ylabel('L2 error')
plt.grid(True, which='both')
plt.savefig('img_lorenz/lorenz_test.pdf', bbox_inches='tight')
plt.show()