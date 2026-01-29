"""
Author: R.CLOAREC
VERSION: 0.2

update: 18/01/2026

MTensor operator

"""

#===============================================================

# IMPORTS

import numpy as np 
from itertools import product
# numpy has no specialized solver for triangular matrix
from scipy.linalg import solve_triangular
import scipy.sparse as sp

# gain for triangular matrix inversion
#from scipy.linalg.lapack import dtrtri

#===============================================================

# UTILS

def mode_tensor_product( *argv ):
	"""
	Build the rank-1 tensor for each row of input data
	"""

	args = [ np.array(arg) for arg in argv ]

	# should check all shapes
	#if not np.all([arg.ndim==2 for arg in args]):

	#	raise ValueError("All arg must be 2D ndarrays")

	# case all inputs are vectors
	if all([arg.ndim == 1 for arg in args]):

		tmp = np.multiply.outer(args[0], args[1])

		if len(args)==2:
			
			return tmp

		for i in range(2, len(args)):
			
			tmp = np.multiply.outer(tmp, args[i])
		
		return tmp

	# case all ndim==2
	m = args[0].shape[0]

	# check common shape
	if not np.all([arg.shape[0]==m for arg in args]):
		
		raise ValueError("All input ndarrays must have common first dimension")

	out = []

	for i in range(m):

		tmp = np.multiply.outer( args[0][i], args[1][i] )

		for j in range( 2, len(argv) ):

			tmp = np.multiply.outer( tmp, args[j][i] )
		
		out.append(tmp)

	return np.array(out)


def _prod(a):
	"""
	computes the product of the elements of an iterable
	
	"""

	out = a[0]
	
	for i in range(1, len(a)):
	
		out *= out[i]

	if hasattr(out, '__iter__'):

		out = _prod(out)

	return out


#===============================================================


# define a class to deal with tensor calc
class MTensor:
	"""
	Definition of the m-tensor format

	Arguments: (to initialize)
	----------
	cores (list[np.ndarray] | np.ndarray - required) : list or array of cores of the tensor to build
		default: []

	Methods:
	--------
	copy		: builds a copy of self
	full		: builds the full tensor as np.ndarray
	pinv		: computes self pseudoinverse
	__repr__	: str representation of self
	__getitem__	: item getter
	__iter__	: iterable behaviour of the m-tensor
	__neg__		: negative self
	__mul__		: scalar multiplication and Hadamard product (*)
	__rmul__	: same as before from the left
	__truediv__	: scalar division and Hadamard division (/)
	__rtruediv__: same as before from the left
	contract_r	: m-tensor row-dim contraction
	contract_c	: m-tensor column-dim contraction
	dot			: m-product and compatible tensor-dot products
	__matmul__	: dot (@)
	norm		: norm of self
	svd			: Singular Value Decomposition
	pca			: Principal Component Analysis
	alid		: Almost Linearly Independent Decomposition
	solve		: solves for X in 'self @ X = y'
	append		: appends cores or rows to self
	"""


	def __init__( self, cores : list[np.ndarray] = []):

		# deal with case where all input cores are vectors
		if np.all([c.ndim == 1 for c in cores]):

			cores_ = [np.array([c]) for c in cores]

		else:

			cores_ = cores

		if not np.all([c.shape[0] == cores_[0].shape[0] for c in cores_]):

			raise Exception(f"All cores should share same first shape (col dim)")

		# list of 2D np.ndarray : the cores should share same shape[0]
		self.cores = cores_

		# ndim (ndarray equiv)
		self.ndim = len(self.cores)

		# order of the tensor 
		self.order = self.ndim + 1

		# columns dimensions (m-mode fibers shape)
		# how to build with more than 1D ?
		self.cdim = (self.cores[0].shape[0],)

		# rows dimensions (p-modes fibers shapes)
		self.rdim = tuple([ core.shape[1] for core in self.cores ])

		# define the shape of full self
		self.shape = self.cdim + self.rdim


	def copy(self):
		"""
		Return a copy of self
		"""

		return MTensor([c.copy() for c in self.cores])


	def full(self):
		"""
		Returns the full version of the tensor

		Full tensor construction is based on the mode tensor product
		"""

		return mode_tensor_product( *self.cores )


	def pinv(self, separated=False):
		"""
		computes the transposed cores of the pseudoinverse of self

		separated (bool - optional ; default: False) :
			is separated pinv provided is the separated axes pinv
		"""
		# separated axes pinv
		if separated:

			cores = [np.linalg.pinv(c).T for c in self.cores]

			return MTensor(cores)
		
		# generalized pinv
		else:

			# this result will not be a m-tensor
			# is that product even implemented?? (__rmatmul__ by np.ndarray)
			# --> see __rmatmul__ : answer is NOPE!
			return self @ np.linalg.pinv(self@self)


	def __repr__(self) -> str:
		
		repr = f'MTensor with shape {self.shape}\nWith {self.ndim} cores:\n'
		
		for core in self.cores:

			repr += core.__repr__() + '\n'

		return repr


	def __getitem__(self, ind):

		# int index is a row of self
		if isinstance(ind, int):

			if ind >= self.cdim[0] :

				raise IndexError(f'Index {ind} out of range with size {self.cdim}')
			
			# if full is needed use self[ind].full()
			return MTensor([c[ind] for c in self.cores])
			
		
		# if index is given as an array
		elif isinstance(ind, np.ndarray):

			# apply mask to cdim (select rows)
			if ind.dtype == bool and ind.shape == self.cdim:

				return MTensor([c[ind] for c in self.cores])

			# apply mask to rdim, result is flattened in a (m, N_true) 2D array
			if ind.dtype == bool and ind.shape == self.rdim:
			
				# initialize with shape (m, nnz(mask))
				out = np.zeros((self.cdim[0], np.count_nonzero(ind)))

				# first retrieve coordinates from mask
				indx = np.where(ind)

				# here populate m-fibers (columns)
				for i in range(len(indx[0])):

					# populate full m-fiber reorienting case use in __getitem__:
	 				# using self.__getitem__(): case hasattr(ind,'__iter__'), w/ len(cores)
					out[:, i] = self[*[indx[j][i] for j in range(len(indx))]]

				return out
			
			# if array is not boolean, deal w/ as __iter__
			else:

				# skip type==np.ndarray : ind.tolist()
				# leads to hasattr(__iter__)
				return self[ind.tolist()]

		# if ind is iterable, build item
		elif hasattr(ind, '__iter__'):

			out = None

			# case: extract a m-fiber along samples
			if len(ind) == self.ndim:

				out = np.ones(self.cdim)

				for i in range(len(ind)):

					out *= self.cores[i][:, ind[i]]

				return out

			# case: extract an element of the tensor
			if len(ind) == self.order:

				out = 1.

				for i in range(len(self.cores)):

					out *= self.cores[i][ind[0], ind[i+1]]

				return out
			
			raise IndexError(f'Index {ind} is not in a recognized format')
		
		# else is an error: raise IndexError
		else:

			raise IndexError(f'Index {ind} is not consistent with tensor shape {self.shape}')


	def __iter__(self):
		"""
		iterable on rows

		TODO: allow to loop on elements of rows if self.cim==(1,)
		make efficient by building elements at each call
		(not call iter from outer product)
		"""
		i = 0

		while i < self.cdim[0]:

			yield self[i]

			i += 1
		

	def __neg__(self):
		"""
		compute the negated m-tensor by negating first core
		
		"""

		# negate only first element of cores
		negself = self.copy()
		negself.cores[0] = - 1 * negself.cores[0]

		return negself


	def __mul__(self, a):
		"""
		Element-wise multiplication between m-tensor and vector or scalar

		"""

		mulself = self.copy()

		# term-by-term mult along m-fiber with vector or nd-array
		if isinstance(a, np.ndarray):

			if a.shape != mulself.cdim :

				raise Exception(f'Inconsistent shape {a.shape} for term by terms product along m-fiber for {mulself.shape}')

			for i in range(a.shape[0]):

				mulself.cores[0][i, :] = a[i]*mulself.cores[0][i, :]

			return mulself
		
		# term-by-term mult along m-fiber with vector or nd-array
		if isinstance(a, MTensor):

			if a.shape != self.shape :

				raise Exception(f'Inconsistent shape {a.shape} for term by terms product for {mulself.shape}')

			for i in range(self.ndim):
			
				mulself.cores[i] *= a.cores[i]

			return mulself

		else:

			# scalar multiplication
			mulself.cores[0] *= a

			return mulself
	

	def __rmul__(self, a):
		"""
		Does not work with a==np.ndarray with a.shape==self.cdim
		overridden with array mul

		"""

		return self.__mul__(a)


	def __truediv__(self, a):
		"""
		Element-wise division

		"""

		divself = self.copy()

		# term-by-term mult along m-fiber with vector or nd-array
		if isinstance(a, np.ndarray):

			if a.shape != divself.cdim :

				raise Exception(f'Inconsistent shape {a.shape} for term by terms division along m-fiber for {divself.shape}')

			for i in range(a.shape[0]):

				divself.cores[0][i, :] = divself.cores[0][i, :]/a[i]

			return divself
		
		# term-by-term mult along m-fiber with vector or nd-array
		if isinstance(a, MTensor):

			if a.shape != self.shape :

				raise Exception(f'Inconsistent shape {a.shape} for term by terms divison for {divself.shape}')

			for i in range(self.ndim):
			
				divself.cores[i] /= a.cores[i]

			return divself

		else:

			# scalar multiplication
			divself.cores[0] /= a

			return divself
	

	def __rtruediv__(self, a):
		"""
		default back to __truediv__
		TODO
		
		"""

		return self.__truediv__(a)


	def contract_r(self):
		"""
		Returns the r-dim contraction of the MTensor

		result will be a vector of length c-dim
		"""

		return self @ ones((1,)+self.rdim)


	def contract_c(self):
		"""
		Returns the c-dim contraction of the MTensor

		result will be a tensor of shape rdim
		
		"""

		# initialize output
		res = np.zeros(self.rdim)

		# need to loop on multiindex in rdim (req. product)
		for i in range(self.cdim[0]):

			# sum along c-dim
			res += self[i].full()

		return res


	def dot(self, B : np.ndarray):
		"""
		sets the self.dot(B) operator

		here B is supposed to be the shape of self
		excluding the first (sampling) dimension

		work to include the case with tensor including the m-fiber:
		id to matrix on matrix mult. Need to build m1xm2 matrix in the end

		TWO DIFFERENCIATION stages : identify if B is a full tensor
		(ie isinstance np.ndarray) or an instance of MTensor
		2nd : identify matrix or vector result 
		
		this operation might be combinatorial in complexity

		TODO:
			- rmul by a being a 2D np.ndarray
			- rmul by a being a 1D np.ndarray
			- rmul by a being a ND np.ndarray (? less important in use)
		
		how to manage complexity of mtensor.T @ mtensor
		product of two mtensors always falls back to mtensor@mtensor.T
		"""
		
		# 1st step: identify type of B for summation method
		# if B is a np.ndarray
		if isinstance(B, np.ndarray):

			# vector ouput
			if B.ndim == self.ndim:

				m = self.cdim[0]

				res = np.zeros(m)

				# coputes full summation over each p_k-fiber of the tensor
				# tensor is implicitely fully constructed
				# by each call to self.__getitem__
					
				# equivalent low data operation on each mode of the tensor?
				for k in range(m):

					res[k] = np.tensordot(self[k], B, axes=[ # why B[0] initially ?
						range(self.ndim),# range(self.ndim-1, -1, -1) # TODO check why this does not work
						tuple([p - 1 for p in list(range(self.ndim))[::-1]])
					])
				
				return res

			# matrix output
			elif B.ndim == self.order:

				m1 = self.cdim[0]
				m2 = B.shape[-1]

				axs = [
					range(-1, -self.order, -1),
					range(self.ndim)
				]

				res = np.zeros((m1, m2))
			
				for k1 in range(m1):

					tmp = self[k1]

					# equivalent low data operation on each mode of the tensor?
					for k2 in range(m2):

						res[k1, k2] = np.tensordot(tmp, (B.T[k2]).T, axes=axs)

				return res

			else:

				raise Exception(f'Unexpected shape {B.shape} for operand in tensor contraction with tensor of shape {self.shape}')

		# if B is an instance of MTensor:
		# compute light weight product - implies transpose of MTensor instance
		elif isinstance(B, MTensor):

			# check shape of B is coherent with self:
			if B.shape[1:] != self.shape[1:]:

				raise Exception(f'Shape {B.shape} incoherent for dot product with tensor of shape {self.shape}')

			res = np.ones((self.shape[0], B.shape[0]))

			for i in range(len(self.shape[1:])):

				res *= self.cores[i]@B.cores[i].T
					
			return res


	def __matmul__(self, B : np.ndarray):

		return self.dot(B)


	def norm(self, axis : int = 0):
		"""
		Computes l2 norm of self

		if axis = 1 returns thhe vector of size c-dim
			containing the norm of each row
		"""

		if axis == 1:

			return np.sqrt( (self*self).contract_r() )

		return np.sqrt( (self*self).contract_r().sum() )


	def svd(self, s_only : bool = False, compute_v : bool = False):

		if s_only:

			S = np.linalg.svd(self@self, compute_uv=False)

			return np.sqrt(S)

		if compute_v:

			U, S = np.linalg.svd(self@self)[:2]

			S = np.sqrt(S)

			VT = ((1/S)*U).T @ self

			return U, S, VT
		
		U, S = np.linalg.svd(self@self)[:2]

		return U, np.sqrt(S)

		
	def pca(self, s_only : bool = False, compute_v : bool = False):

		m = self.cdim[0]

		tmp_mtx = 1/m*np.ones((m,m))

		# center
		P = self@self
		P = P - 2*tmp_mtx@P + tmp_mtx@P@tmp_mtx

		if s_only:

			S = np.linalg.svd(P, compute_uv=False)

			return np.sqrt(S)

		if compute_v:

			U, S = np.linalg.svd(P)[:2]

			S = np.sqrt(S)

			VT = ((1/S)*U).T @ self

			return U, S, VT
		
		U, S = np.linalg.svd(P)[:2]

		return U, np.sqrt(S)


	def alid(self, tol : float = 1e-3, greedy : bool = True, compute_w : bool = True):
		"""
		Computes the Almost linearly independent decomposition of self

		if compute_w is true, returns 
		"""

		# greedy ALID
		if greedy:

			# build self_ALI incrementally
			self_ALI = MTensor([c[0] for c in self.cores])

			# initialize cholesky factor [make it a 2D array]
			L = np.sqrt((self_ALI@self_ALI))

			# initialize weights
			if compute_w: W = np.array([[1.]])

			# stream like data loop
			for i in range(1, self.cdim[0]):

				# evaluate phi(sample_i)
				row_i = MTensor([c[i] for c in self.cores])

				b = (self_ALI@row_i)[:, 0]
				n = (row_i@row_i)[0, 0]

				# evaluate coef minimizing dist in feature space
				s = solve_triangular(L, b, lower=True)

				# evaluate dist in feature space
				dist = n - s@s

				# depending on criterion keep or not [is row_i ALD]
				if dist > tol:

					# append row_i to phi
					self_ALI.append(row_i.cores, axis=-1)

					# update cholesky factor
					L = np.vstack((np.c_[L, np.zeros(L.shape[0])], np.r_[s.T, np.sqrt(dist)]))

					# update weights with 1.
					if compute_w:

						W = np.vstack((np.c_[W, np.zeros(W.shape[0])], np.r_[np.zeros(W.shape[1]), 1.]))

				elif compute_w:

					# simply update weights
					W = np.vstack((W, solve_triangular(L.T, s)))

		# optimal ALID
		else:

			# to build delta_0, build proj
			P = self@self

			# build delta_0 matrix to evaluate dist
			delta = np.array([[P[i,i]-(P[i,j]**2/P[j,j]) for i in range(self.cdim[0])] for j in range(self.cdim[0])])

			# index of minimizing column
			ind = np.argmin(delta.sum(axis=0))

			# initialize ALID
			self_ALI = MTensor([c[ind] for c in self.cores])

			# check if all dist fall under tol
			if np.all(delta[:,ind]<tol):

				if compute_w: 
					
					W = (1/self_ALI.norm()**2)(self@self_ALI)

					return self_ALI, W

				else: return self_ALI

			# initialize Cholesky factor
			L = np.array([[self_ALI.norm()]])
			
			# build mask (T_bar in paper)
			msk = np.ones(self.cdim, dtype=bool)
			msk[ind] = False

			# loop on rows
			for k in range(1, self.cdim[0]):

				# build vector of norms of rows
				n = np.array([row.norm() for row in self[msk]])

				# build matrix B
				B = self_ALI@self[msk]

				# build matrix S
				S = solve_triangular(L, B)

				# build matrix delta
				delta = ( n*np.ones((self.cdim[0]-k, self.cdim[0]-k)) ).T - S.T@S
					
				# index of minimizing column
				ind = np.argmin(delta.sum(axis=0))

				# append row to ALID
				self_ALI.append([c[ind] for c in self[msk].cores])

				# compute vector s
				s = S[:, ind]

				# update Cholesky factor
				L = np.vstack((np.c_[L, np.zeros(L.shape[0])], np.r_[s.T, np.sqrt(self[ind].norm()**2 - s@s)]))

				# update mask (restrict T_bar)
				msk[ind] = False

				# check if all dist fall under tol
				if np.all(delta[:,ind]<tol):

					if compute_w: 
						
						# inverse L
						L_inv = np.linalg.inv(L)

						# compute weights
						W = (self@self_ALI)@L_inv.T@L_inv
				
					# break loop and return
					break
		
		# -- RETURN --
		if compute_w:

			return self_ALI, W
		
		else:

			return self_ALI


	def solve(self, rhs : np.ndarray, regul : float = 0., separated : bool = False):
		"""
		Computes the regression coefficients C that solve 
			self @ C = rhs

		regul (float - optional) : Tikhonov regularization

		"""
		# separated axes case
		if separated:

			if rhs.ndim == 1:

				slv = self.pinv() * rhs

				C = np.zeros( self.rdim )

				for pw in product( *[range( k ) for k in self.rdim] ):

					C[ *pw ] = np.sum( slv[ *pw ] )
		
				# need to explain this factor!!! see paper: mode m unfolding
				return (self.cdim[0]**(self.ndim-1)) * C

			elif rhs.ndim > 1:

				C = np.empty( self.rdim + rhs.shape[1:] ).T

				for ic in product(*[range(ni) for ni in rhs.shape[1:]]):

					# iteratively solve for vector rhs
					C[*ic[::-1]] = self.solve(rhs[:,*ic], regul, True).T

				return C.T
		
		# otherwise general solving procedure
		# compute Cholesky decomposition of self@self (more stable)
		L = np.linalg.cholesky(self@self + regul*np.eye(self.cdim))

		Z = solve_triangular(L.T, solve_triangular(L, rhs, lower=True))

		return self@Z

	
	def append(self, x : np.ndarray | list[np.ndarray] , axis : int | None = None):
		"""
		method to append data to the tensor

		axis is the mode to append data to

		is axis == None, x is added as a core (or cores if x is a list of cores)

		if axis == 0, x should be a list of np.ndarray with consistent col size
		with len self.ndim. Each element will be appended at the bottom of each core
		"""

		# Append new core(s): axis==None
		if axis == None:

			# if input is a list of new cores
			if isinstance(x, list):

				if not np.all([c.shape[0] == self.cdim[0] for c in x]):

					raise Exception(f"All cores should share same first shape (col dim)")				

				self.cores += x
			
			# if input is a single new core
			elif isinstance(x, np.ndarray):

				if not x.shape[0] == self.cdim[0]:

					raise Exception(f'Appended core should have column size {self.cdim[0]}, not {x.shape[0]}')

				self.cores.append( x )
		
		# Append data along m-fiber (bottom of each core): axis==0
		if axis == -1:

			for i in range(self.ndim):

				self.cores[i] = np.vstack( (self.cores[i], x[i]) )

		# Append data along i-th core: axis==i
		elif axis >= 0 and axis <= self.ndim:

			self.cores[axis] = np.c_[ self.cores[axis], x ]

		elif axis > self.ndim :

			raise IndexError(f'Axis {axis} is out of bound with ndim {self.ndim}')

		# then update dims of tensor
		# ndim (ndarray equiv)
		self.ndim = len(self.cores)

		# order of the tensor
		self.order = self.ndim + 1

		# columns dimensions (m-mode fibers shape)
		self.cdim = (self.cores[0].shape[0],)

		# rows dimensions (p-modes fibers shapes)
		self.rdim = tuple([ core.shape[1] for core in self.cores ])

		# define the shape of full self
		self.shape = self.cdim + self.rdim


		return self


#===============================================================


# initialize a formatted tensor from shape
def zeros( shape : tuple = (1, 1, 1) ):
	"""
	Returns a tensor of class MTensor of shape 'shape' filled with zeros
	"""

	cores = [np.zeros( (shape[0], shape[i]) ) for i in range(1, len(shape))]

	return MTensor( cores )


def ones( shape : tuple = (1, 1, 1) ):
	"""
	Returns a tensor of class MTensor of shape 'shape' filled with ones
	"""

	cores = [np.ones( (shape[0], shape[i]) ) for i in range(1, len(shape))]

	return MTensor( cores )


def randn( shape : tuple = (1, 1, 1) ):
	"""
	Returns a tensor of class MTensor of shape 'shape' filled with random floats
	"""

	cores = [np.random.randn( shape[0], shape[i] ) for i in range(1, len(shape))]

	return MTensor( cores )


# ==============================================================

# define a class to deal with sparse cores tensor calc
class SparseMTensor:
	"""
	Definition of the sparse version of the m-tensor format

	Arguments: (to initialize)
	----------
	cores (list[sp.sparse_array] - required) : list or array of cores of the tensor to build
		default: []

	Methods:
	--------
	copy		: builds a copy of self
	full		: builds the full tensor as np.ndarray
	pinv		: computes self pseudoinverse
	__repr__	: str representation of self
	__getitem__	: item getter
	__iter__	: iterable behaviour of the m-tensor
	__neg__		: negative self
	__mul__		: scalar multiplication and Hadamard product (*)
	__rmul__	: same as before from the left
	__truediv__	: scalar division and Hadamard division (/)
	__rtruediv__: same as before from the left
	contract_r	: m-tensor row-dim contraction
	contract_c	: m-tensor column-dim contraction
	dot			: m-product and compatible tensor-dot products
	__matmul__	: dot (@)
	norm		: norm of self
	svd			: Singular Value Decomposition
	pca			: Principal Component Analysis
	alid		: Almost Linearly Independent Decomposition
	solve		: solves for X in 'self @ X = y'
	append		: appends cores or rows to self
	"""


	def __init__( self, cores : list[sp.csr_array] = []):

		# note bsr_arrays all have ndim=2, if any other type
		if np.all([not isinstance(c, sp.csr_array) for c in cores]):

			try :

				cores_ = [sp.csr_array(c) for c in cores]

			except:

				err_msg = f'Wrong type encountered in sparse MTensor init:\n{[type(c) for c in cores]}\n'
				err_msg += f"All cores should be convertible to scipy.sparse.csr_array"
				raise Exception(err_msg)

		else:

			cores_ = cores

		if not np.all([c.shape[0] == cores_[0].shape[0] for c in cores_]):

			raise Exception(f"All cores should share same first shape (col dim)")

		# list of 2D np.ndarray : the cores should share same shape[0]
		self.cores = cores_

		# ndim (ndarray equiv)
		self.ndim = len(self.cores)

		# order of the tensor 
		self.order = self.ndim + 1

		# columns dimensions (m-mode fibers shape)
		# how to build with more than 1D ?
		self.cdim = (self.cores[0].shape[0],)

		# rows dimensions (p-modes fibers shapes)
		self.rdim = tuple([ core.shape[1] for core in self.cores ])

		# define the shape of full self
		self.shape = self.cdim + self.rdim


	def copy(self):
		"""
		Return a copy of self
		"""

		return SparseMTensor([c.copy() for c in self.cores])


	def full(self):
		"""
		Returns the full version of the tensor

		Full tensor construction is based on the mode tensor product

		NO ND-array with sparse structure in scipy.sparse, output full
		"""

		return mode_tensor_product( *[c.toarray() for c in self.cores] )


	def pinv(self, separated=False):
		"""
		computes the transposed cores of the pseudoinverse of self

		the inverse of a sparse core will most likely nor be sparse
		the computation therfore returns a full mtensor, not sparse

		separated (bool - optional ; default: False) :
			is separated pinv provided is the separated axes pinv
		"""
		# separated axes pinv
		if separated:

			cores = [sp.linalg.inv(c).T for c in self.cores]

			return MTensor(cores)
		
		# generalized pinv
		else:

			raise NotImplementedError
			# this result will not be a m-tensor
			# is that product even implemented?? (__rmatmul__ by np.ndarray)
			# --> see __rmatmul__ : answer is NOPE!
			return self @ np.linalg.pinv(self@self)


	def __repr__(self) -> str:
		
		repr = f'SparseMTensor with shape {self.shape}\nWith {self.ndim} cores:\n'
		
		for core in self.cores:

			repr += core.toarray().__repr__() + '\n'

		return repr


	# needs sparse update
	def __getitem__(self, ind):

		# int index is a row of self
		if isinstance(ind, int):

			if ind >= self.cdim[0] :

				raise IndexError(f'Index {ind} out of range with size {self.cdim}')
			
			# if full is needed use self[ind].full()
			return SparseMTensor([c[ind: ind+1] for c in self.cores])
			
		
		# if index is given as an array
		elif isinstance(ind, np.ndarray):

			# apply mask to cdim (select rows)
			if ind.dtype == bool and ind.shape == self.cdim:

				return SparseMTensor([c[ind] for c in self.cores])

			# apply mask to rdim, result is flattened in a (m, N_true) 2D array
			if ind.dtype == bool and ind.shape == self.rdim:
			
				# initialize with shape (m, nnz(mask))
				out = np.zeros((self.cdim[0], np.count_nonzero(ind)))

				# first retrieve coordinates from mask
				indx = np.where(ind)

				# here populate m-fibers (columns)
				for i in range(len(indx[0])):

					# populate full m-fiber reorienting case use in __getitem__:
	 				# using self.__getitem__(): case hasattr(ind,'__iter__'), w/ len(cores)
					out[:, i] = self[*[indx[j][i] for j in range(len(indx))]]

				return out
			
			# if array is not boolean, deal w/ as __iter__
			else:

				# skip type==np.ndarray : ind.tolist()
				# leads to hasattr(__iter__)
				return self[ind.tolist()]

		# if ind is iterable, build item
		elif hasattr(ind, '__iter__'):

			out = None

			# case: extract a m-fiber along samples
			if len(ind) == self.ndim:

				out = np.ones(self.cdim)

				for i in range(len(ind)):

					out *= self.cores[i][:, ind[i]]

				return out

			# case: extract an element of the tensor
			if len(ind) == self.order:

				out = 1.

				for i in range(len(self.cores)):

					out *= self.cores[i][ind[0], ind[i+1]]

				return out
			
			raise IndexError(f'Index {ind} is not in a recognized format')
		
		# else is an error: raise IndexError
		else:

			raise IndexError(f'Index {ind} is not consistent with tensor shape {self.shape}')


	def __iter__(self):
		"""
		iterable on rows

		TODO: allow to loop on elements of rows if self.cim==(1,)
		make efficient by building elements at each call
		(not call iter from outer product)
		"""
		i = 0

		while i < self.cdim[0]:

			yield self[i]

			i += 1
		

	def __neg__(self):
		"""
		compute the negated m-tensor by negating first core
		
		"""

		# negate only first element of cores
		negself = self.copy()
		negself.cores[0] = - 1 * negself.cores[0]

		return negself


	def __mul__(self, a):
		"""
		Element-wise multiplication between m-tensor and vector or scalar

		"""

		mulself = self.copy()

		# term-by-term mult along m-fiber with vector or nd-array
		if isinstance(a, np.ndarray):

			if a.shape != mulself.cdim :

				raise Exception(f'Inconsistent shape {a.shape} for term by terms product along m-fiber for {mulself.shape}')

			mulself.cores[0] = ( a*(mulself.cores[0].T) ).T

			return mulself
		
		# term-by-term mult along m-fiber with vector or nd-array
		if isinstance(a, (MTensor, SparseMTensor)):

			if a.shape != self.shape :

				raise Exception(f'Inconsistent shape {a.shape} for term by terms product for {mulself.shape}')

			for i in range(self.ndim):
			
				mulself.cores[i] *= a.cores[i]

			return mulself

		else:

			# scalar multiplication
			mulself.cores[0] *= a

			return mulself
	

	def __rmul__(self, a):
		"""
		Does not work with a==np.ndarray with a.shape==self.cdim
		overridden with array mul

		"""

		return self.__mul__(a)


	def __truediv__(self, a):
		"""
		Element-wise division

		"""

		divself = self.copy()

		# term-by-term mult along m-fiber with vector or nd-array
		if isinstance(a, np.ndarray):

			if a.shape != divself.cdim :

				raise Exception(f'Inconsistent shape {a.shape} for term by terms division along m-fiber for {divself.shape}')

			divself.cores[0] = (divself.cores[0].T/a).T

			return divself
		
		# term-by-term mult along m-fiber with vector or nd-array
		if isinstance(a, MTensor):

			if a.shape != self.shape :

				raise Exception(f'Inconsistent shape {a.shape} for term by terms divison for {divself.shape}')

			for i in range(self.ndim):
			
				divself.cores[i] /= a.cores[i]

			return divself

		else:

			# scalar multiplication
			divself.cores[0] /= a

			return divself
	

	def __rtruediv__(self, a):
		"""
		default back to __truediv__
		TODO
		
		"""

		return self.__truediv__(a)


	def contract_r(self):
		"""
		Returns the r-dim contraction of the MTensor

		result will be a vector of length c-dim
		"""

		return self@ones(self.rdim)


	def contract_c(self):
		"""
		Returns the c-dim contraction of the MTensor

		result will be a tensor of shape rdim
		
		"""

		# initialize output
		res = np.zeros(self.rdim)

		# need to loop on multiindex in rdim (req. product)
		for i in range(self.cdim[0]):

			# sum along c-dim
			res += self[i].full()

		return res


	def dot(self, B : np.ndarray):
		"""
		sets the self.dot(B) operator

		here B is supposed to be the shape of self
		excluding the first (sampling) dimension

		work to include the case with tensor including the m-fiber:
		id to matrix on matrix mult. Need to build m1xm2 matrix in the end

		TWO DIFFERENCIATION stages : identify if B is a full tensor
		(ie isinstance np.ndarray) or an instance of MTensor
		2nd : identify matrix or vector result 
		
		this operation might be combinatorial in complexity

		TODO:
			- rmul by a being a 2D np.ndarray
			- rmul by a being a 1D np.ndarray
			- rmul by a being a ND np.ndarray (? less important in use)
		
		how to manage complexity of mtensor.T @ mtensor
		product of two mtensors always falls back to mtensor@mtensor.T
		"""
		
		# 1st step: identify type of B for summation method
		# if B is a np.ndarray
		if isinstance(B, np.ndarray):

			# vector ouput
			if B.ndim == self.ndim:

				m = self.cdim[0]

				res = np.zeros(m)

				# coputes full summation over each p_k-fiber of the tensor
				# tensor is implicitely fully constructed
				# by each call to self.__getitem__
					
				# equivalent low data operation on each mode of the tensor?
				for k in range(m):

					res[k] = np.tensordot(self[k], B, axes=[ # why B[0] initially ?
						range(self.ndim),# range(self.ndim-1, -1, -1) # TODO check why this does not work
						tuple([p - 1 for p in list(range(self.ndim))[::-1]])
					])
				
				return res

			# matrix output
			elif B.ndim == self.order:

				m1 = self.cdim[0]
				m2 = B.shape[-1]

				axs = [
					range(-1, -self.order, -1),
					range(self.ndim)
				]

				res = np.zeros((m1, m2))
			
				for k1 in range(m1):

					tmp = self[k1]

					# equivalent low data operation on each mode of the tensor?
					for k2 in range(m2):

						res[k1, k2] = np.tensordot(tmp, (B.T[k2]).T, axes=axs)

				return res

			else:

				raise Exception(f'Unexpected shape {B.shape} for operand in tensor contraction with tensor of shape {self.shape}')

		# if B is an instance of MTensor:
		# compute light weight product - implies transpose of MTensor instance
		elif isinstance(B, (MTensor, SparseMTensor)):

			# check shape of B is coherent with self:
			if B.shape[1:] != self.shape[1:]:

				raise Exception(f'Shape {B.shape} incoherent for dot product with tensor of shape {self.shape}')

			res = self.cores[0]@B.cores[0].T

			for i in range(1,len(self.shape[1:])):

				res *= self.cores[i]@B.cores[i].T
					
			return res


	def __matmul__(self, B : np.ndarray):

		return self.dot(B)


	def norm(self, axis : int = 0):
		"""
		Computes l2 norm of self

		if axis = 1 returns thhe vector of size c-dim
			containing the norm of each row
		"""

		if axis == 1:

			return np.sqrt( (self*self).contract_r() )

		return np.sqrt( (self*self).contract_r().sum() )


	def svd(self, s_only : bool = False, compute_v : bool = False):

		if s_only:

			S = np.linalg.svd(self@self, compute_uv=False)

			return np.sqrt(S)

		if compute_v:

			U, S = np.linalg.svd(self@self)[:2]

			S = np.sqrt(S)

			VT = ((1/S)*U).T @ self

			return U, S, VT
		
		U, S = np.linalg.svd(self@self)[:2]

		return U, np.sqrt(S)

		
	def pca(self, s_only : bool = False, compute_v : bool = False):

		m = self.cdim[0]

		tmp_mtx = 1/m*np.ones((m,m))

		# center
		P = self@self
		P = P - 2*tmp_mtx@P + tmp_mtx@P@tmp_mtx

		if s_only:

			S = np.linalg.svd(P, compute_uv=False)

			return np.sqrt(S)

		if compute_v:

			U, S = np.linalg.svd(P)[:2]

			S = np.sqrt(S)

			VT = ((1/S)*U).T @ self

			return U, S, VT
		
		U, S = np.linalg.svd(P)[:2]

		return U, np.sqrt(S)


	def alid(self, tol : float = 1e-3, greedy : bool = True, compute_w : bool = True):
		"""
		Computes the Almost linearly independent decomposition of self

		if compute_w is true, returns 
		"""

		# greedy ALID
		if greedy:

			# build self_ALI incrementally
			self_ALI = MTensor([c[0] for c in self.cores])

			# initialize cholesky factor [make it a 2D array]
			L = np.sqrt((self_ALI@self_ALI))

			# initialize weights
			if compute_w: W = np.array([[1.]])

			# stream like data loop
			for i in range(1, self.cdim[0]):

				# evaluate phi(sample_i)
				row_i = MTensor([c[i] for c in self.cores])

				b = (self_ALI@row_i)[:, 0]
				n = (row_i@row_i)[0, 0]

				# evaluate coef minimizing dist in feature space
				s = solve_triangular(L, b, lower=True)

				# evaluate dist in feature space
				dist = n - s@s

				# depending on criterion keep or not [is row_i ALD]
				if dist > tol:

					# append row_i to phi
					self_ALI.append(row_i.cores, axis=-1)

					# update cholesky factor
					L = np.vstack((np.c_[L, np.zeros(L.shape[0])], np.r_[s.T, np.sqrt(dist)]))

					# update weights with 1.
					if compute_w:

						W = np.vstack((np.c_[W, np.zeros(W.shape[0])], np.r_[np.zeros(W.shape[1]), 1.]))

				elif compute_w:

					# simply update weights
					W = np.vstack((W, solve_triangular(L.T, s)))

		# optimal ALID
		else:

			# to build delta_0, build proj
			P = self@self

			# build delta_0 matrix to evaluate dist
			delta = np.array([[P[i,i]-(P[i,j]**2/P[j,j]) for i in range(self.cdim[0])] for j in range(self.cdim[0])])

			# index of minimizing column
			ind = np.argmin(delta.sum(axis=0))

			# initialize ALID
			self_ALI = MTensor([c[ind] for c in self.cores])

			# check if all dist fall under tol
			if np.all(delta[:,ind]<tol):

				if compute_w: 
					
					W = (1/self_ALI.norm()**2)(self@self_ALI)

					return self_ALI, W

				else: return self_ALI

			# initialize Cholesky factor
			L = np.array([[self_ALI.norm()]])
			
			# build mask (T_bar in paper)
			msk = np.ones(self.cdim, dtype=bool)
			msk[ind] = False

			# loop on rows
			for k in range(1, self.cdim[0]):

				# build vector of norms of rows
				n = np.array([row.norm() for row in self[msk]])

				# build matrix B
				B = self_ALI@self[msk]

				# build matrix S
				S = solve_triangular(L, B)

				# build matrix delta
				delta = ( n*np.ones((self.cdim[0]-k, self.cdim[0]-k)) ).T - S.T@S
					
				# index of minimizing column
				ind = np.argmin(delta.sum(axis=0))

				# append row to ALID
				self_ALI.append([c[ind] for c in self[msk].cores])

				# compute vector s
				s = S[:, ind]

				# update Cholesky factor
				L = np.vstack((np.c_[L, np.zeros(L.shape[0])], np.r_[s.T, np.sqrt(self[ind].norm()**2 - s@s)]))

				# update mask (restrict T_bar)
				msk[ind] = False

				# check if all dist fall under tol
				if np.all(delta[:,ind]<tol):

					if compute_w: 
						
						# inverse L
						L_inv = np.linalg.inv(L)

						# compute weights
						W = (self@self_ALI)@L_inv.T@L_inv
				
					# break loop and return
					break
		
		# -- RETURN --
		if compute_w:

			return self_ALI, W
		
		else:

			return self_ALI


	def solve(self, rhs : np.ndarray, regul : float = 0., separated : bool = False):
		"""
		Computes the regression coefficients C that solve 
			self @ C = rhs

		regul (float - optional) : Tikhonov regularization

		"""
		# separated axes case
		if separated:

			if rhs.ndim == 1:

				slv = self.pinv() * rhs

				C = np.zeros( self.rdim )

				for pw in product( *[range( k ) for k in self.rdim] ):

					C[ *pw ] = np.sum( slv[ *pw ] )
		
				# need to explain this factor!!! see paper: mode m unfolding
				return (self.cdim[0]**(self.ndim-1)) * C

			elif rhs.ndim > 1:

				C = np.empty( self.rdim + rhs.shape[1:] ).T

				for ic in product(*[range(ni) for ni in rhs.shape[1:]]):

					# iteratively solve for vector rhs
					C[*ic[::-1]] = self.solve(rhs[:,*ic], regul, True).T

				return C.T
		
		# otherwise general solving procedure
		# compute Cholesky decomposition of self@self (more stable)
		L = np.linalg.cholesky(self@self + regul*sp.eye_array(self.cdim[0]))

		Z = solve_triangular(L.T, solve_triangular(L, rhs, lower=True))

		return self@Z

	
	# update for sparse
	def append(self, x : np.ndarray | list[np.ndarray] , axis : int | None = None):
		"""
		method to append data to the tensor

		axis is the mode to append data to

		is axis == None, x is added as a core (or cores if x is a list of cores)

		if axis == 0, x should be a list of np.ndarray with consistent col size
		with len self.ndim. Each element will be appended at the bottom of each core
		"""

		# Append new core(s): axis==None
		if axis == None:

			# if input is a list of new cores
			if isinstance(x, list):

				if not np.all([c.shape[0] == self.cdim[0] for c in x]):

					raise Exception(f"All cores should share same first shape (col dim)")				

				self.cores += x
			
			# if input is a single new core
			elif isinstance(x, np.ndarray):

				if not x.shape[0] == self.cdim[0]:

					raise Exception(f'Appended core should have column size {self.cdim[0]}, not {x.shape[0]}')

				self.cores.append( x )
		
		# Append data along m-fiber (bottom of each core): axis==0
		if axis == -1:

			for i in range(self.ndim):

				self.cores[i] = sp.vstack( (self.cores[i], x[i]) )

		# TODO: update for sparse
		# Append data along i-th core: axis==i
		elif axis >= 0 and axis <= self.ndim:

			self.cores[axis] = np.c_[ self.cores[axis], x ]

		elif axis > self.ndim :

			raise IndexError(f'Axis {axis} is out of bound with ndim {self.ndim}')

		# then update dims of tensor
		# ndim (ndarray equiv)
		self.ndim = len(self.cores)

		# order of the tensor
		self.order = self.ndim + 1

		# columns dimensions (m-mode fibers shape)
		self.cdim = (self.cores[0].shape[0],)

		# rows dimensions (p-modes fibers shapes)
		self.rdim = tuple([ core.shape[1] for core in self.cores ])

		# define the shape of full self
		self.shape = self.cdim + self.rdim


		return self


#===============================================================
