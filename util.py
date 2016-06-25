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
