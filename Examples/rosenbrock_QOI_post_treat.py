"""
R.CLOAREC

post_treat for rosenbrock benchmark QOI

18/11/2025

post treatment for the QOI from error analysis
of the Rosenbrock benchmark in high dimensions

data in QOI is organized as

QOI = np.zeros((n_trial, len(lst_dim), len(lst_m_axis), n_qoi))
# will refer to elts as 
# QOI[iT, iD, iM, iQ] = qoi # where iQ only is manually set
    iT: trial no
    iD: dim
    iM: sample per axis
    iQ: qoi

iQ = 0 : l2 error on training
iQ = 1 : max error on training
iQ = 2 : l2 error on test
iQ = 3 : max error on test
iQ = 4 : unit inference time
iQ = 5 : time to build P | time for ALID    | time for SVD of P LR
iQ = 6 : time to build Z | m_ alid          | Time to build Z
iQ = 7 :                 | time to build Z


TODO: sum time to build P and solve for Z in the case of 
LR and Least squares
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

RIDGE = False
ALID = False
LR = True

SAVEFIGS = True

# check archived QOI as well

if RIDGE : QOI = np.load('./ridge_QOI.npy')
if ALID : QOI = np.load('./alid_QOI.npy')
if LR : QOI = np.load('./lr_QOI.npy')

print('Shape of QOI')
print(QOI.shape)

# list of tested dimensions
lst_dim = [20, 50, 100, 200, 300]
# list of 1D sampling size
lst_m_axis = [5, 10, 20, 30, 40]
# number of training and check to run
n_trial = 3

# ================= POST TREAT ===================

# average along the first axis
QOI = 1/n_trial * QOI.sum(axis=0)


if LR:

    QOI[:, :, 7] = QOI[:, :, 5] + QOI[:, :, 6]

# first two dims of QOI are a grid
X, Y = np.meshgrid(lst_m_axis, lst_dim)

# for each QOI, plot surface
for i in range(QOI.shape[2]):

    print(i)

    Z = QOI[:,:, i]

    # plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # case for logScale
    #ax.plot_surface(X, np.log10(Y), Z)
    #ax.plot_surface(X, Y, Z, cmap=cm.magma)

    ax.plot_surface(X, Y, Z, cmap=cm.YlGnBu)

    ax.set_xlabel('Sample per axis')
    ax.set_ylabel('Dimension')

    # same position for all
    elev, azim, roll = 30, 50, 0
    ax.view_init(elev, azim, roll)

    if SAVEFIGS: plt.savefig(f'img/QOI_{i}.pdf', bbox_inches='tight')
    plt.show()

    # plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # case for logScale
    #ax.plot_surface(X, np.log10(Y), Z)
    ax.plot_surface(X, Y, Z, cmap=cm.YlGnBu)

    ax.set_xlabel('Samples per axis')
    ax.set_ylabel('Dimension')

    # same position for all (case 2)
    elev, azim, roll = 30, -140, 0
    ax.view_init(elev, azim, roll)

    if SAVEFIGS: plt.savefig(f'img/QOI_{i}_2.pdf', bbox_inches='tight')

    plt.show()
