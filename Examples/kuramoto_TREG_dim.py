"""
================
Kuramoto model
================

Author: R.CLOAREC
Date: 12/12/2025

Info: this file implements the example use of m-tensor
for dynamical system learning on the case of Kuramoto
oscillators.

"""

import matplotlib.pyplot as plt
import numpy as np

import tensorop as to

#============ UTILS ================================

# exact kuramoto system
def kuramoto(x, omega, *, K=2):
   
   d = len(x)

   x_dot = []

   for i in range(d):
       
       x_dot.append( omega[i] + (K/d)*np.sum([np.sin(x[j]-x[i]) for j in range(d)]) )

   return np.array(x_dot)

dims = [3, 10]
sfs  = [1, 1]
rnks = [50, 1000]
taus = [1e-3, 200]

PLOT_SVD = False
DIST = True


for i in range(len(dims)):

   dim = dims[i]
   sf = sfs[i]
   tau = taus[i]
   rnk = rnks[i]

   omega = 10*np.random.random(dim) - 5.
   print(omega)

   init_pos = np.random.random(dim)
   init_pos = init_pos / np.linalg.norm(init_pos)


   #=============== BUILD EXACT Kuramoto ========================

   # timestep and number of steps
   # sampling strategy is baked in the construction of the Lorenz
   # attractor through num_steps

   dt = 0.1
   num_steps = 1000

   # initialize position output
   X = np.empty((num_steps + 1, dim))  # Need one more for the initial values

   # set initial position
   X[0] = init_pos

   # initialize derivatives output
   Xdt = np.empty((num_steps, dim))

   # Step through time, calculating the partial derivatives at the current point
   # and using them to estimate the next point
   for i in range(num_steps):
      
      Xdt[i] = kuramoto(X[i], omega)
      X[i + 1] = X[i] + Xdt[i] * dt


   #===================== BUILD TENSOR REGRESSION =============

   m = num_steps

   psi_1 = lambda x: np.array([1., sf*np.cos(x)])
   psi_2 = lambda x: np.array([1., sf*np.sin(x)])

   phi = to.Tensor([np.array([psi_1(X[k, i]) for k in range(m)]) for i in range(dim)]+[np.array([psi_2(X[k, i]) for k in range(m)]) for i in range(dim)])

   P = phi@phi

   # regul parameter
   l = 0

   # compute Z from P and rhs
   Z = np.linalg.lstsq(P + l*np.eye(m), Xdt)[0]

   check_rhs = P@Z

   # should be shit
   print('Error l2 new',np.linalg.norm(Xdt-check_rhs)/np.linalg.norm(Xdt))

   def f(x):

      phi_x = to.Tensor([psi_1(x_i) for x_i in x]+[psi_2(x_i) for x_i in x])

      return ((phi_x@phi)@Z)[0]

   #===================== BUILD LR TENSOR REGRESSION =============

   U, S = np.linalg.svd(P)[:2]
   S = np.sqrt(S)

   if PLOT_SVD:

      plt.plot(S)
      plt.show()

   U_, S_ = U[:, :rnk], S[:rnk]

   # test linearization by regressing
   # compute Z from P_ and rhs
   Z_ = U_ @ U_.T @ np.linalg.lstsq(P, Xdt)[0]

   check_rhs = phi@phi@Z_

   # should be shit
   print('Error l2 LR',np.linalg.norm(Xdt-check_rhs)/np.linalg.norm(Xdt))

   def f_(x):

      phi_x = to.Tensor([psi_1(x_i) for x_i in x]+[psi_2(x_i) for x_i in x])

      return ((phi_x@phi)@Z_)[0]

   # ===================== SUBSAMPLING LANDO-LIKE ============

   print(f'{' Here ALID ':=^80}')

   import time

   data = X[:-1]
   dim = data.shape[1]
   m = data.shape[0]

   # mask for kept data latr for regression
   msk_rhs = [True]

   # initialize with first
   data_init = data[0]

   phi_lando = to.Tensor([psi_1(data_init[i]) for i in range(dim)]+[psi_2(data_init[i]) for i in range(dim)])

   # initialize cholesky factor [make it a 2D array]
   L = np.sqrt((phi_lando@phi_lando))

   count = 0

   # stream like data loop
   for sample in data[1:]:

      count += 1

      # evaluate phi(sample_i)
      phi_i = to.Tensor([psi_1(sample[i]) for i in range(dim)]+[psi_2(sample[i]) for i in range(dim)])

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
      if DIST: print(f'dist {dist}')

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

   print('here', count, len(msk_rhs), data.shape)

   # once best subset is identified based on given data, TREG
   m_ = phi_lando.shape[0]

   # rhs for weights computation
   Y = Xdt[msk_rhs]

   # change this solving method and use L instead THEN how to regularize ?
   Z_lando = np.linalg.solve(L.T, np.linalg.solve(L, Y))

   print(f'Shapes for LANDO system: phi_lando: {phi_lando.shape}, Z_lando: {Z_lando.shape}')

   # online computation
   def f_lando(x):

      phi_x = to.Tensor([psi_1(x_i) for x_i in x]+[psi_2(x_i) for x_i in x])

      return ((phi_x@phi_lando)@Z_lando)[0]


   check_rhs = phi@phi_lando@Z_lando

   # should be shit
   print('Error l2 LANDO',np.linalg.norm(Xdt-check_rhs)/np.linalg.norm(Xdt))


   #===================== BUILD TENSOR REGRESSION =============
   # PLOT TEST

   num_steps_test = int(50*num_steps)

   # ------------ REAL --------------

   # initialize position output
   X = np.empty((num_steps_test + 1, dim))  # Need one more for the initial values

   # set initial position
   X[0] = init_pos

   # initialize derivatives output
   Xdt = np.empty((num_steps_test, dim))


   # timing for construction of trajectory
   t1 = time.perf_counter()

   # Step through time, calculating the partial derivatives at the current point
   # and using them to estimate the next point
   for i in range(num_steps_test):
      
      Xdt[i] = kuramoto(X[i], omega)
      X[i + 1] = X[i] + Xdt[i] * dt

   # timer for reconstruction
   print(f'time reconstructing REAL trajectory: {time.perf_counter() - t1}s')
   print(f'ie unitary cost for f_: {(time.perf_counter() - t1)/num_steps_test}s')


   # ------------ TENSOR FULL RANK --------------

   # re-generate data from initial point using tensor integrator
   X_ = np.empty((num_steps_test + 1, dim))  # Need one more for the initial values

   # set initial position
   X_[0] = init_pos

   # timing for construction of trajectory
   t1 = time.perf_counter()

   # Step through time, calculating the partial derivatives at the current point
   # and using them to estimate the next point
   for i in range(num_steps_test):
      
      X_[i + 1] = X_[i] + f(X_[i]) * dt

   # timer for reconstruction
   print(f'time reconstructing TENSOR FULL RANK trajectory: {time.perf_counter() - t1}s')
   print(f'ie unitary cost for f_: {(time.perf_counter() - t1)/num_steps_test}s')


   # ------------ TENSOR LOW RANK --------------

   # re-generate data from initial point using tensor integrator
   X_LR = np.empty((num_steps_test + 1, dim))  # Need one more for the initial values

   # set initial position
   X_LR[0] = init_pos

   # timing for construction of trajectory
   t1 = time.perf_counter()

   # Step through time, calculating the partial derivatives at the current point
   # and using them to estimate the next point
   for i in range(num_steps_test):
      
      X_LR[i + 1] = X_LR[i] + f_(X_LR[i]) * dt

   # timer for reconstruction
   print(f'time reconstructing TENSOR LOW RANK trajectory: {time.perf_counter() - t1}s')
   print(f'ie unitary cost for f_: {(time.perf_counter() - t1)/num_steps_test}s')


   # ------------ TENSOR LANDO SUBSAMPLING --------------

   # re-generate data from initial point using tensor integrator
   X_ALI = np.empty((num_steps_test + 1, dim))  # Need one more for the initial values

   # set initial position
   X_ALI[0] = init_pos

   # timing for construction of trajectory
   t1 = time.perf_counter()

   # Step through time, calculating the partial derivatives at the current point
   # and using them to estimate the next point
   for i in range(num_steps_test):
      
      X_ALI[i + 1] = X_ALI[i] + f_lando(X_ALI[i]) * dt

   # timer for reconstruction
   print(f'time reconstructing TENSOR LANDO trajectory: {time.perf_counter() - t1}s')
   print(f'ie unitary cost for f_: {(time.perf_counter() - t1)/num_steps_test}s')

   # 
   delta_full_rank = np.linalg.norm(X_ - X, axis=1)/np.linalg.norm(X, axis=1)
   delta_low_rank = np.linalg.norm(X_LR - X, axis=1)/np.linalg.norm(X, axis=1)
   delta_lando = np.linalg.norm(X_ALI - X, axis=1)/np.linalg.norm(X, axis=1)

   plt.plot(range(num_steps_test+1), delta_full_rank, label='full rank')
   plt.plot(range(num_steps_test+1), delta_low_rank, label='low rank', linestyle='--')
   plt.plot(range(num_steps_test+1), delta_lando, label='ALI regul')
   plt.legend()
   plt.yscale('log')
   plt.xlabel('Timestep')
   plt.ylabel('L2 error')
   plt.grid(True, which='both')
   plt.show()
