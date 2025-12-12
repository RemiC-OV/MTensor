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
from scipy.linalg import solve_triangular
import time

import mtensor as mt

#============ UTILS ================================


# exact kuramoto system
def kuramoto(x, omega, *, K=2):
   
   d = len(x)

   x_dot = np.zeros(d)

   for i in range(d):
       
       x_dot[i] = omega[i] + (K/d)*np.sum([np.sin(x[j]-x[i]) for j in range(d)]) 

   return np.array(x_dot)

dims = [3, 10, 100]
sfs  = [1, 1e-1, 1e-2]
taus = [1e-3, 1e-4, 5e-5]

num_steps = 1000
num_steps_test = int(50*num_steps)

n_trials = 5

PLOT_SVD = False
DIST = False
SAVEFIGS = False


for i in range(len(dims)):

   # store data of error plots to build average
   curves_lstsq = np.zeros((n_trials, num_steps_test+1))
   curves_ALI = np.zeros((n_trials, num_steps_test+1))

   for k in range(n_trials):

      # set loop dep variables
      dim = dims[i]
      sf = sfs[i]
      tau = taus[i]

      # initialize random natural frequencies
      omega = 10*np.random.random(dim) - 5.
      print(omega)

      # set initial conditions
      init_pos = 2*np.pi*np.random.random(dim)
      print(init_pos)


      #=============== BUILD EXACT Kuramoto ========================

      # timestep and number of steps
      # sampling strategy is baked in the construction of the Lorenz
      # attractor through num_steps
      dt = 0.1

      # initialize position output
      X = np.empty((num_steps + 1, dim))  # Need one more for the initial values

      # set initial position
      X[0] = init_pos

      # initialize derivatives output
      Xdt = np.empty((num_steps, dim))

      # Step through time, calculating the partial derivatives at the current point
      # and using them to estimate the next point
      for t in range(num_steps):
         
         Xdt[t] = kuramoto(X[t], omega)
         X[t + 1] = X[t] + Xdt[t] * dt


      #===================== BUILD TENSOR REGRESSION =============

      m = num_steps

      psi_1 = lambda x: np.array([1., sf*np.cos(x)])
      psi_2 = lambda x: np.array([1., sf*np.sin(x)])

      phi = mt.MTensor([np.array([psi_1(X[k, i]) for k in range(m)]) for i in range(dim)]+[np.array([psi_2(X[k, i]) for k in range(m)]) for i in range(dim)])

      P = phi@phi

      # regul parameter
      l = 0

      # compute Z from P and rhs
      Z = np.linalg.lstsq(P + l*np.eye(m), Xdt)[0]

      check_rhs = P@Z

      # should be shit
      print('Error l2 new',np.linalg.norm(Xdt-check_rhs)/np.linalg.norm(Xdt))

      def f(x):

         phi_x = mt.MTensor([psi_1(x_i) for x_i in x]+[psi_2(x_i) for x_i in x])

         return ((phi_x@phi)@Z)[0]

      # ===================== SUBSAMPLING ALID ============

      print(f'{' Here ALID ':=^80}')


      data = X[:-1]
      dim = data.shape[1]
      m = data.shape[0]

      # mask for kept data latr for regression
      msk_rhs = [True]

      # initialize with first
      data_init = data[0]

      phi_alid = mt.MTensor([psi_1(data_init[i]) for i in range(dim)]+[psi_2(data_init[i]) for i in range(dim)])

      # initialize cholesky factor [make it a 2D array]
      L = np.sqrt((phi_alid@phi_alid))

      count = 0

      # stream like data loop
      for sample in data[1:]:

         count += 1

         # evaluate phi(sample_i)
         phi_i = mt.MTensor([psi_1(sample[i]) for i in range(dim)]+[psi_2(sample[i]) for i in range(dim)])

         # vector
         k_i = (phi_alid@phi_i)[:, 0]
         # scalar
         k_ii = (phi_i@phi_i)[0, 0]

         # evaluate coef minimizing dist in feature space
         s = solve_triangular(L, k_i, lower=True)
         a = solve_triangular(L.T, s)

         # evaluate dist in feature space
         dist = k_ii - k_i@a
         if DIST: print(f'dist {dist}')

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

      print('here', count, len(msk_rhs), data.shape)

      # once best subset is identified based on given data, TREG
      m_ = phi_alid.shape[0]

      # rhs for weights computation
      Y = Xdt[msk_rhs]

      # change this solving method and use L instead THEN how to regularize ?
      Z_alid = solve_triangular(L.T, solve_triangular(L, Y, lower=True))

      print(f'Shapes for alid system: phi_alid: {phi_alid.shape}, Z_alid: {Z_alid.shape}')

      # online computation
      def f_alid(x):

         phi_x = mt.MTensor([psi_1(x_i) for x_i in x]+[psi_2(x_i) for x_i in x])

         return ((phi_x@phi_alid)@Z_alid)[0]


      check_rhs = phi@phi_alid@Z_alid

      # should be shit
      print('Error l2 alid',np.linalg.norm(Xdt-check_rhs)/np.linalg.norm(Xdt))


      #===================== BUILD TENSOR REGRESSION =============
      # PLOT TEST

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
      for t in range(num_steps_test):
         
         Xdt[t] = kuramoto(X[t], omega)
         X[t + 1] = X[t] + Xdt[t] * dt

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
      for t in range(num_steps_test):
         
         X_[t + 1] = X_[t] + f(X_[t]) * dt

      # timer for reconstruction
      print(f'time reconstructing TENSOR FULL RANK trajectory: {time.perf_counter() - t1}s')
      print(f'ie unitary cost for f_: {(time.perf_counter() - t1)/num_steps_test}s')

      # ------------ TENSOR alid SUBSAMPLING --------------

      # re-generate data from initial point using tensor integrator
      X_ALI = np.empty((num_steps_test + 1, dim))  # Need one more for the initial values

      # set initial position
      X_ALI[0] = init_pos

      # timing for construction of trajectory
      t1 = time.perf_counter()

      # Step through time, calculating the partial derivatives at the current point
      # and using them to estimate the next point
      for t in range(num_steps_test):
         
         X_ALI[t + 1] = X_ALI[t] + f_alid(X_ALI[t]) * dt

      # timer for reconstruction
      print(f'time reconstructing TENSOR alid trajectory: {time.perf_counter() - t1}s')
      print(f'ie unitary cost for f_: {(time.perf_counter() - t1)/num_steps_test}s')

      # 
      delta_full_rank = np.linalg.norm(X_ - X, axis=1)/np.linalg.norm(X, axis=1)
      delta_alid = np.linalg.norm(X_ALI - X, axis=1)/np.linalg.norm(X, axis=1)

      plt.plot(range(num_steps_test+1), delta_full_rank, label='full rank')
      plt.plot(range(num_steps_test+1), delta_alid, label='ALI regul')
      plt.legend()
      plt.yscale('log')
      plt.xlabel('Timestep')
      plt.ylabel('L2 error')
      plt.grid(True, which='both')
      plt.show()

      # store to average
      print(k)
      curves_lstsq[k] = delta_full_rank
      curves_ALI[k] = delta_alid

   # here build averaged curves
   plt.fill_between(range(num_steps_test+1), curves_lstsq.max(axis=0), curves_lstsq.min(axis=0), alpha = .5, linewidth=0)
   plt.fill_between(range(num_steps_test+1), curves_ALI.max(axis=0), curves_ALI.min(axis=0), alpha = .5, linewidth=0)

   plt.plot(range(num_steps_test+1), curves_lstsq.sum(axis=0)/n_trials, label='Least squares')
   plt.plot(range(num_steps_test+1), curves_ALI.sum(axis=0)/n_trials, label='ALI regul.')
   
   plt.legend()
   plt.yscale('log')
   plt.xlabel('Timestep')
   plt.ylabel('L2 error')
   plt.ylim((1e-18, 1e2))
   plt.grid(True, which='both')

   if SAVEFIGS : plt.savefig(f'img_kuramoto/kuramoto_dim_{dim}.pdf', bbox_inches='tight')

   plt.show()

# TODO: test the model on unseen initial conditions