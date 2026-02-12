# toy problem

import numpy as np
import tensoropV2 as mt

def f(x, y):

    return 5 - x + 3*x*y - x**2 - 15*x**2*y**2 - 3*x*y**2 - x**2*y

#c_real = np.array([5, 0, 0, -1, 3, -3, -1, -1, -15])

P = [(-1,-1), (0,1), (1,0)]

y = np.array([f(*xi) for xi in P])

psi = lambda x: np.array([1, x, x**2])

psi_1 = np.array([psi(xi[0]) for xi in P])
psi_2 = np.array([psi(xi[1]) for xi in P])

phi = mt.Tensor([psi_1, psi_2])
print(phi)

phi_m = phi.full().reshape((3,9))
print(f'Tensor unfolding (matrix-based)\n{phi_m}')

# solve matrix way
c = np.linalg.lstsq(phi_m, y)[0]
print(f'Matrix-based least squares result\n{c}')

# solve tensor
z = np.linalg.lstsq(phi@phi, y)[0]
print(f'M-Tensor intermediate z\n{z}')
C = np.einsum('ijk,k->ij', phi.full().T, z)
print(f'M-Tensor-based least squares result\n{C}')