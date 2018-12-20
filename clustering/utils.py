import numpy as np
import scipy.io
import h5py
from scipy.spatial.distance import pdist, squareform

def load_dot_mat(path, db_name):
    try:
        mat = scipy.io.loadmat(path)
    except NotImplementedError:
        mat = h5py.File(path)

    return np.array(mat[db_name]).transpose()

def gaussian_kernel(X, sigma):
	pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
	return scipy.exp(-pairwise_sq_dists / sigma**2)

# 30% slower
def vectorized_RBF_kernel(X, sigma):
    # % This is equivalent to computing the kernel on every pair of examples
    X2 = np.sum(np.multiply(X, X), 1) # sum colums of the matrix
    K0 = X2 + X2.T - 2 * X * X.T
    K = np.power(np.exp(-1.0 / sigma**2), K0)
    return K