def colvec(x):
    """
    Transform 1D np.array of size (n,) to 2D np.array of size (n, 1)
    :param x: np.array
    :return: np.array
    """
    if x is not None:
        return x.reshape((x.size, 1))
    else:
        raise ValueError

import numpy as np
from numba import guvectorize

@guvectorize(['void(float64[:,:], float64[:,:], float64[:,:])',
              'void(float32[:,:], float32[:,:], float32[:,:])'],
              '(m,n),(n,p)->(m,p)', target='cuda')
def matmul(A, B, C):
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
