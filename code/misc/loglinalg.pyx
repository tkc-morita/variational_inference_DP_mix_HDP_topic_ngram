# coding: utf-8

import numpy as np
cimport numpy as np
import scipy.misc as spm
import cython
from cython.parallel cimport prange
from cpython cimport bool


def logmatmul(mat1, mat2):
	return _logmatmul(mat1.astype(np.float64), mat2.astype(np.float64))
	
def logdotprod(vect1, vect2):
	return _logdotprod(vect1.astype(np.float64), vect2.astype(np.float64))
	
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
cdef np.ndarray[np.float64_t,ndim=2] _logmatmul(
										np.ndarray[np.float64_t,ndim=2] mat1,
										np.ndarray[np.float64_t,ndim=2]mat2
										):
	cdef int row_size = mat1.shape[0]
	cdef int col_size = mat2.shape[1]
	cdef np.ndarray[np.float64_t,ndim=2] output = np.zeros((row_size,col_size))
	cdef int row_id, col_id
	with nogil:
		for row_id in prange(row_size):
			for col_id in xrange(col_size):
				with gil:
					output[row_id, col_id]=_logdotprod(mat1[row_id,:], mat2[:,col_id])
	return output
	
	
cdef np.float64_t _logdotprod(
							np.ndarray[np.float64_t,ndim=1] vect1,
							np.ndarray[np.float64_t,ndim=1] vect2
							):
	return spm.logsumexp(vect1+vect2)