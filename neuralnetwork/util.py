import numpy as np


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


def vectorized_result(j, n_bin):
    """
    Transform categorical data into vector form
    ex> 3 -> [0,0,1,0,0]
    :param j: int
    :return: np.Array
    """
    e = np.zeros((n_bin, 1))
    e[j] = 1.0
    return e