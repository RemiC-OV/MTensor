"""
Tensor operator based on the m-tensor format

TODO: regularization based on ridge regression on each dimension
TODO: augment possibilites to enlarge library and differentiate each axis

We implement the m-tensor format for multidimensional regression

TODO: implement differentiation and integration of the cores

TODO: safe generalized sparse regression framework while limitting the order
of basis functions

"""


import numpy as np 
from itertools import product

#===============================================================

rng = np.random.default_rng()

def sample(size : int = 10, parameters = None):

	# dimension of the parametric space
	dim = len( parameters )

	# initialize level's sample
	listMuI = []

	# loop on size of sample
	for i in range( size ):

		# initialize the parameter
		mu = {}

		# produce a random sample along each dimension
		for d in range( dim ):

			prm = parameters[ d ]
			mu[ prm["label"] ] = rng.uniform( prm["interval"][0], prm["interval"][1], 1 )[0]

		listMuI.append( mu )

	return listMuI


def mu_( mu ):
	"""
	transforms mu as a dict with param labels 
	to an iterable (list of values)
	"""

	itMu = []

	for label in mu:

		itMu.append( mu[label] )

	return itMu


def sparse_grid_subsample(intervals, p, t=0):
    """
    intervals : each 1D subdomains (iterable of iterables)
        len of intervals should be D
    t : oversampling parameter

    based on full sampling : for each axis of the grid, exclude 
    after a sample is made along that axis

    should store data linear in dimension

    TODO: axes shuffling approach
    ie: define the sampling along each axis based on the largest p_i
    then instead of subsampling in a large grid in high dimension
    suffle each linspaced 1D sample and associate arrays:
    ends up with linear cost with subsample

    TODO: theoretical comparison with randomized subsampling
    check A NOUY for th. background on RNLA

    Check any chance for alignment (singular system)

    TODO: check use of sparse grid subsample with sparse regression
    technique
    """

    # define D linspaced axes with each (p+1+t) elements
    axes = [np.linspace(e[0], e[1], p+1+t) for e in intervals]

    for ax in axes:

        np.random.shuffle(ax)

    return np.c_[*axes]

#===============================================================

def smolyak_level_implicit(level, rdim):
	"""
	Combinatoric iterator implementing smolyak's rule

	only outputs combinations of len(rdim) elements 
	whose sum amounts to level
	"""

	r = level if level < rdim[0] else rdim[0]
	n = len(rdim)

	# archetype (1|n) of distribution of 1 over n
	# considers all(rdim >= 1)
	if r == 1:

		for i in range(n):

			out = [0]*n
			out[i] = 1

			yield tuple(out)

	# archetype (r|2) of distribution of r over 2
	elif n == 2:

		for i in range(r+1):

			yield (r-i, i)


	# else find achetype (1|n) or (r|2)
	else:

		yield tuple([r]+[0]*(n-1))
					
		for i in range(r-1, -1, -1):

			for e in smolyak_level_implicit(r-i, rdim[1:]):

				yield (i,)+e


def smolyak_implicit(levelmax, rdim):

	for deg in range(levelmax+1):

		for pw in smolyak_level_implicit(deg, rdim):

			yield pw


#===============================================================

def basis_function( x : float, n : int = 0, basis : str = 'poly' ):
	"""
	x : float - parameter value (1D)
	n : int - order of the function 
	basis : str -	  'monom'|'cos'|'exp'...
	build basis function based on param
	and order of basis function

	Note:
	trigonometric functions take degree as input
	"""

	# coherently, all bases at order 0 should return 1
	if n == 0:

		return 1.
	
	if basis == 'poly':

		return x ** n
	
	if basis == 'invpoly':

		return x ** -n
	
	if basis == 'cos':

		return np.cos( n * np.pi * x / 180 )
	
	if basis == 'sin':

		return np.sin( n * np.pi * x / 180 ) + 1
	
	# 'binary' bases do not take n into account
	if basis == 'exp':

		return np.exp( x )
	
	if basis == 'log':

		if x <= 0.:

			return 1.

		return np.log( x )


#===============================================================

def mode_tensor_product( *argv ):
	"""
	Build the rank-1 tensor for each row of input data

	TODO: build special case of single row input
	generalize use to single vectors and vector-matrix
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

	P = [ arg.shape[1] for arg in args ]

	out = []

	for i in range(m):

		tmp = np.multiply.outer( args[0][i], args[1][i] )

		for j in range( 2, len(argv) ):

			tmp = np.multiply.outer( tmp, args[j][i] )
		
		out.append(tmp)

	return np.array(out)


def phi1D(sample, deg : int = 0, dim : int = 0, basis = 'poly'):
		"""
		Builds the matrix for a basis in 1D

		Arguments:
		----------
		deg ( int ) : max degree to reach in the specified basis
		dim ( int ) : dimension of the parametric space in which to build the matrix
		basis ( str ) : string representing the basis in which to build the matrix

		Returns:
		----------
		mat (2D-ndarray) : matrix containing the sample in specified basis in specified dimension
		"""

		m = len(sample)
		
		mat = np.zeros( (m, deg+1) )

		for i in range(deg+1):

			for j in range(m):

				mu = mu_( sample[j] )

				mat[ j, i ] = basis_function( mu[dim], i, basis )

		return mat


#===============================================================


def sparse_reg( muDict, coefs, pws, bases = ['poly'] ):

	res = 0

	mu = mu_(muDict)

	for i in range(len(coefs)):

		resTmp = coefs[i]

		for j in range(len(pws)):

			d = j // len(bases)

			b = bases[ j%len(bases) ]

			resTmp *= basis_function( mu[d], pws[j][i], b )
		
		res += resTmp

	return res


def sparse_reg_implicit( muDict, c, pws, deg, bases = ['poly'] ):

	# format parameter
	mu = mu_(muDict)

	# initialize num
	N = 0
	# initialize denom
	D = 0

	# compute numerator component
	for i in range(len(c)):

		resTmp = c[i]

		for j in range(len(pws)):

			d = j // len(bases)

			b = bases[ j%len(bases) ]

			# if deg is higher than :deg:, modulo deg+1 and consider denom data
			pw = pws[j][i]%(deg+1) if (pws[j][i] > deg and j == 0) else pws[j][i]

			resTmp *= basis_function( mu[d], pw, b )
		
		if pws[0][i] > deg:

			D -= resTmp
		
		else:

			N += resTmp
	
	if D == 0:

		return N

	return N/D


#===============================================================


# define a class to deal with tensor calc
class Tensor:
	"""
	Specific tensor format

	Arguments:
	----------
	cores (list[np.ndarray] | np.ndarray - required) : list or array of cores of the tensor to build
		default: []

	Methods:
	--------
	.full() : outputs the full tensor as np.ndarray
	.full_inv() : outputs the full pseudo inverse of the tensor as np.ndarray
	.__repr__() : displays the output of self.full()
	.__getitem__(ind) : depending on ind, outputs elements of the tensor
	.__matmul__(B) : computes the contracted product on rdim with B
	._phi1D(deg, dim, basis) : builds the 2D array for sampled data in dim in basis at deg
	._build_cores() : builds the cores of tensor based on input data of init
	._build_inverse() : builds the inverse cores of tensor basde on input data if init
	.solve(rhs) : outputs the regression coefs C of least square self @ C = rhs
	.append(mu) : appends mu (or [mus]) to tensor and re-computes cores and inverse

	"""

	def __init__( self, cores : list[np.ndarray] = []):

		# deal with case where all input cores are vectors
		if np.all([c.ndim == 1 for c in cores]):

			cores_ = [np.array([c]) for c in cores]

		else:

			cores_ = cores

		if not np.all([c.shape[0] == cores[0].shape[0] for c in cores]):

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

		# define transposition
		self.IS_TRANSPOSED = False


	def copy(self):
		"""
		return copy of self
		"""

		return Tensor([c.copy() for c in self.cores])


	def full(self):
		"""
		Returns the full version of the tensor

		Full tensor construction is based on the mode tensor product
		This product builds a tensor product for each matrix in the 
		cores. This leads to a rank-1 tensor for each sample
		"""

		return mode_tensor_product( *self.cores )


	def pinv(self, regul : float = 0.):
		"""
		computes the transposed cores of the pseudoinverse of self

		regul (float - optional) :
			if regul < 0. : spectral regulation in pinv : all sig < -regul are truncated
			if regul = 0. : regular solver
			if regul > 0. : ridge regression with lambda=regul
		"""

		# standard approach for regul == 0.
		if not regul:

			cores = [np.linalg.pinv(c).T for c in self.cores]
		
		else:

			cores = []

			for core in self.cores:

				U, S, VT = np.linalg.svd(core, full_matrices=False)

				# here treat s based on regul
				# case 1 : regul > 0. :: tikhonov
				if regul > 0. :
					print('here')
					# define lambda: regulation parameter
					lmb = regul

					for si in S:

						print(si)

						si = si / (si**2 + lmb**2)

						print(si)

				# case 2 : regul < 0. :: spectral
				else :

					# define tau: threshold for singular values
					tau = -regul
					
					for si in S:

						print(si)

						si = 1/si if si > tau else 0.

						print(si)

				cores.append(np.dot(VT.T * S, U.T).T)

		return Tensor(cores)


	def full_inv(self):
		"""
		Returns the full version of the inverse of the tensor
		"""

		return mode_tensor_product( *[c for c in self.pinv().cores] ).T


	def __repr__(self) -> str:
		
		repr = f'Tensor with shape {self.shape}\nWith {self.ndim} cores:\n'
		
		for core in self.cores:

			repr += core.__repr__() + '\n'

		return repr


	def __getitem__(self, ind):

		# int index gives tensor for ind's sampled value
		# compute m-tensor product for ind's sample
		if isinstance(ind, int):

			if ind >= self.cdim[0] :

				raise IndexError(f'Index {ind} out of range with size {self.cdim}')
			
			out = np.multiply.outer( self.cores[0][ind], self.cores[1][ind] )

			for j in range( 2, len(self.cores) ):

				out = np.multiply.outer( out, self.cores[j][ind] )
			
			return out
		
		# if index is given as an array
		elif isinstance(ind, np.ndarray):

			# apply mask to cdim (select on sample)
			if ind.dtype == bool and ind.shape == self.cdim: #? was tested ?

				out = np.zeros( (np.count_nonzero(ind), *self.rdim) )

				j = 0

				for i in range(len(ind)):

					if not ind[i]: continue

					out[j] = np.multiply.outer( self.cores[0][ind[i]], self.cores[1][ind[i]] )

					for k in range( 2, len(self.cores) ):

						out[j] = np.multiply.outer( out[j], self.cores[k][ind[i]] )

				return out

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


	def __neg__(self):

		# negate only first element of cores
		negself = self.copy()
		negself.cores[0] = - 1 * negself.cores[0]

		return negself


	def __mul__(self, a):
		"""
		Element-wise multiplication between m-tensor and vector or scalar

		TODO: add hadamard product of two m-tensors
		"""

		mulself = self.copy()

		# term-by-term mult along m-fiber with vector or nd-array
		if type(a) == np.ndarray:

			if a.shape != mulself.cdim :

				raise Exception(f'Inconsistent shape {a.shape} for term by terms product along m-fiber for {mulself.shape}')

			for i in range(a.shape[0]):

				mulself.cores[0][i, :] = a[i]*mulself.cores[0][i, :]

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

		TODO: adaptn as was done in mul
		"""

		divself = self.copy()

		if a < 0:

			divself = -divself

		# capitalize on properties of np.ndarray wrt mult
		for i in range(divself.ndim):

			divself.cores[i] /= abs(a)**(1/divself.ndim)

		return self
	

	def __rtruediv__(self, a):

		divself = self.copy()
		
		if a < 0:

			divself = -divself

		# capitalize on properties of np.ndarray wrt mult
		for i in range(divself.ndim):

			self.cores[i] = abs(a)**(1/divself.ndim) / divself.cores[i]

		return self


	def transpose(self):

		transposed_self = self.copy()
		transposed_self.IS_TRANSPOSED = True

		return transposed_self


	def contract_r(self):
		"""
		Returns the r-dim contraction of the Tensor

		result will be a vector of length c-dim
		
		TODO: find more efficient way to sum over outer product of elements
		of cores without the need for __getitem__
		which will compute the outer product

		use m-tensor product with to.ones ?
		"""

		m = self.cdim[0]

		# initialize output a m long vector of zeros
		res = np.zeros(m)

		# loop along m-fiber of self
		for k in range(m):

			# sum all elements of the tensor for k-th element
			# along m-fiber
			res[k] = self[k].sum()

		return res


	def contract_c(self):
		"""
		Returns the c-dim contraction of the Tensor

		result will be a tensor of shape rdim
		
		TODO: fiend more efficient way to sum over r-dim elements
		this method calls __getitem__ and computes the cartesian
		product of r-dim in order to loop on each m-fiber of the tensor
		"""

		# initialize output
		res = np.empty(self.rdim)

		# need to loop on multiindex in rdim (req. product)
		for ic in product(*[range(pi) for pi in self.rdim]):

			# sum along c-dim
			res[*ic] = self[*ic].sum()

		return res


	def dot(self, B : np.ndarray):
		"""
		sets the self.dot(B) operator

		here B is supposed to be the shape of self
		excluding the first (sampling) dimension

		work to include the case with tensor including the m-fiber:
		id to matrix on matrix mult. Need to build m1xm2 matrix in the end

		TWO DIFFERENCIATION stages : identify if B is a full tensor
		(ie isinstance np.ndarray) or an instance of Tensor
		2nd : identify matrix or vector result 
		 
		TODO:
		review summation indices
		add cases

		TODO: add case phi.T@phi
		currently tensor on tensor product systematically leads to 
		m_1xm_2 or m_2xm_1 result
		we wish to also compute the p_N x p_N-1 x ... x p_1 x p_1 x p_2 ... x p_N
		to do so, m-tensors should have a self.transposed state variable
		else we cannot identify
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

		# if B is an instance of Tensor:
		# compute light weight product - implies transpose of Tensor instance
		elif isinstance(B, Tensor):

			# check shape of B is coherent with self:
			if B.shape[1:] != self.shape[1:]:

				raise Exception(f'Shape {B.shape} incoherent for dot product with tensor of shape {self.shape}')

			# old solution
			#print(f'Tensor on Tensor contraction not fully implemented')
			#return self.dot(B.full().T)
			
			"""
			res = np.empty(self.cdim+ B.cdim)

			# loop along m-fiber of self with k1
			for k1 in range(self.cdim[0]):

				# loop along m-fiber of B with k2
				for k2 in range(B.cdim[0]):

					# build a Tensor with shape (1, :self.shape[1:])
					sumTensor = ones((1,)+self.shape[1:])

					# loop on cores of self and B
					for i in range(len(sumTensor.cores)):

						c = sumTensor.cores[i]
						c *= self.cores[i][k1]
						c *= B.cores[i][k2]

					# contract the 1-Tensor in res[k1, k2]
					res[k1, k2] = sumTensor.contract_r()#[0]?
			"""

			res = np.ones((self.shape[0], B.shape[0]))

			for i in range(len(self.shape[1:])):

				res *= self.cores[i]@B.cores[i].T
					
			return res


	def __matmul__(self, B : np.ndarray):

		return self.dot(B)


	def regul(self, tol : float = 1e-3):
		"""
		TEST

		Regularization strategy for m-tensors based on orthogonal projection
		onto the span of a multidimensional basis applied to all dimensions

		This approach corresponds to a multiway spectral truncation

		We add a constraint on infinity norm of residual from orth proj : 
		||r||_infty <= 1.
		This way no augmentation over tol in the multidimensional
		construction of the tensor, there is no cumulative error above tol
		for residuals from orthogonal projection onto our restricted basis

		Parameters:
		-----------
		tol (float - optional) : a regularization tolerance

		Returns:
		--------
		a regularized copy of self

		"""

		# instead of acting in-place return a copy
		T = self.copy()

		# initialize a single vector basis with first core
		B = T.cores[0][:, 0]/np.linalg.norm(T.cores[0][:, 0])

		# build the projector
		P = np.eye(B.shape[0]) - B@B.T

		# loop on each core and update Basis wrt tol
		for i in range(0, T.ndim):

			Ai = T.cores[i]

			# In core Ai loop on col vectors Aij
			for j in range(Ai.shape[1]):

				# Project to compute part of Aij out of span(B)
				r = P@Ai[:, j]

				# compute l-2 norm (tol)
				nr = np.linalg.norm(r)

				# compute infty-norm (multidim cumulative error)
				nr_inf = max(np.absolute(r))
				#print(nr, nr_inf)

				# confront to tol and keep infty-norm error <= 1.
				if nr > tol or nr_inf > 1.:

					# append the normalized residual to the basis
					# r is readily computed to be orthogonal to B
					B = np.c_[B, r/nr]

					# update projectors with enriched basis
					Q = B@B.T
					P = np.eye(B.shape[0]) - Q
			
		# once each core was confronted to the basis to enrich it,
		# we project them onto the span of B to restrict them
		# this is the regularization step
		for i in range(T.ndim):

			T.cores[i] = Q @ T.cores[i]

		return T


	def solve(self, rhs : np.ndarray, regul : float = 0., smolyak_level : int = None, d_axis_0 = False):
		"""
		Computes the regression coefficients that solve 
			T x_m C = rhs

		if smolyak_level, check for multi-dimensionnal index
		and only computes solution for mdIndex <= smolyak_level

		regul (float - optional) :
			if regul < 0. : spectral regulation in pinv
			if regul = 0. : regular solver
			if regul > 0. : ridge regression with lambda=regul

		TODO: change the way this is taken into account
		d_axis_0: is True, axis 0 is considered doubled in size, thus 
		smolyak_level is not treated the same but it is repeated along axis 0

		TODO:Currently returns a full array, this means too much data
		when the dimension of the problem augments. The idea is to seek
		a solution with same tensor format

		TODO: if rhs is a matrix (mxn), iteratively use the same method
		n times to solve for each column and return n tensors

		TODO: implement tikhonov and spectral regul: take them
		into account at the stage of pinv
		"""
		# lack initial check of input data!!!

		# initial computation of self.pinv()
		selfinv = self.pinv(regul=regul)

		# need to make dependent on the shape of rhs
		# can either act on self.__mul__()
		# or iteratively call the untouched __mul__
		# on each column of the rhs
		if rhs.ndim == 1:

			slv = selfinv * rhs

			C = np.zeros( self.rdim )

			if isinstance(smolyak_level, int):

				for pw in smolyak_implicit(smolyak_level, self.rdim):

					C[ *pw ] = np.sum( slv[*pw] )

					if d_axis_0:

						C[ pw[0]+smolyak_level+1, *pw[1:] ] = np.sum( slv[pw[0]+smolyak_level+1, *pw[1:] ] )

			else:

				for pw in product( *[range( k ) for k in self.rdim] ):

					C[ *pw ] = np.sum( slv[ *pw ] )
		
			# need to explain this factor!!! see paper: mode m unfolding
			return (self.cdim[0]**(self.ndim-1)) * C

		elif rhs.ndim > 1:

			C = np.empty( self.rdim + rhs.shape[1:] ).T

			for ic in product(*[range(ni) for ni in rhs.shape[1:]]):

				# iteratively solve for vector rhs
				C[*ic[::-1]] = self.solve(rhs[:,*ic], regul, smolyak_level, d_axis_0).T

			# in that case remove factor: already dealt with
			return C.T
	

	def sparse_solve(self, rhs : np.ndarray, threshold : float, maxit : int = 10, smolyak_level : int = None):
		"""
		Computes the sequentially thresholded regression

		Do not use! Instead use Tensor.solve and then Tensor.sparsify_solution
		"""

		# initialize solution with [tensorized] 'least square'
		_c = self.solve(rhs, smolyak_level)

		# build mask from threshold
		msk = np.ma.masked_greater_equal(np.absolute(_c), threshold, copy=True).mask

		# build matrix from tensor restricted to mask
		_T = self[msk]

		# STLS
		for i in range(maxit):

			# compute lstsq solution
			_c = np.linalg.pinv(_T)@rhs

			# update mask from threshold
			mask = np.ma.masked_greater_equal(np.absolute(_c), threshold, copy=True).mask
			
			if np.all(mask):

				break

			# here update mask_0 by distributing mask onto true elements of maks_0
			msk[msk] *= mask

			# update restricted regression matrix
			_T = _T[:, mask]

		# we can output mask or directly coordinates of nnz coefs
		# with either msk or np.nonzero(msk)
		return _c, msk.nonzero()


	def sparsify_solution(self, coefs: np.ndarray, rhs : np.ndarray, threshold : float):
		"""
		Computes the sequentially thresholded regression from given initial guess
		"""

		# build mask from threshold
		msk = np.absolute(coefs) > threshold

		# for the case where no coef is kept
		if isinstance(msk, np.bool_):

			raise ValueError(f'Threshold {threshold} too high')

		# build matrix from tensor restricted to mask
		_T = self[msk]

		# STLS
		while True:

			# compute lstsq solution
			#_c = np.linalg.pinv(_T)@rhs

			# no pinv is more efficient -> linalg.lstsq
			_c, resid, rnk, svs = np.linalg.lstsq(_T, rhs, rcond=None)

			# update mask from threshold
			mask = np.absolute(_c) > threshold
			
			# exit if mask does not change
			if np.all(mask):

				break

			# here update mask_0 by distributing mask onto true elements of mask_0
			msk[msk] *= mask

			# here check if there are still true in mask
			if not np.any(msk):

				# if not any true: raise error
				raise ValueError(f'Threshold {threshold} too high')

			# update restricted regression matrix
			_T = _T[:, mask]
		
		#print(msk)

		# we can output mask or directly coordinates of nnz coefs
		# with either msk or np.nonzero(msk)
		return _c, list(msk.nonzero())


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
	Returns a tensor of class Tensor of shape 'shape' filled with zeros
	"""

	cores = [np.zeros( (shape[0], shape[i]) ) for i in range(1, len(shape))]

	return Tensor( cores )


def ones( shape : tuple = (1, 1, 1) ):
	"""
	Returns a tensor of class Tensor of shape 'shape' filled with ones
	"""

	cores = [np.ones( (shape[0], shape[i]) ) for i in range(1, len(shape))]

	return Tensor( cores )


def randn( shape : tuple = (1, 1, 1) ):
	"""
	Returns a tensor of class Tensor of shape 'shape' filled with random floats
	"""

	cores = [np.random.randn( shape[0], shape[i] ) for i in range(1, len(shape))]

	return Tensor( cores )
