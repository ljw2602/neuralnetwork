import numpy as np

from util import colvec

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoidprime(z):
    """
    (i, j)th element is d(a_i)/d(z_j)
    diagonal elements are a_i * (1 - a_i)
    off-diagonal elements are zero
    :param z: int or np.array
    :return: int or np.array
    """
    if isinstance(z, int):
        return sigmoid(z) * (1-sigmoid(z))
    elif isinstance(z, np.ndarray):
        return np.diag(sigmoid(z) * (1-sigmoid(z)))
    else:
        raise TypeError


def relu(z):
    if isinstance(z, int):
        return np.maximum(z, 0)
    elif isinstance(z, np.ndarray):
        return np.maximum(z, np.zeros(z.shape))


def reluprime(z):
    """
    (i, j)th element is d(a_i)/d(z_j)
    diagonal elements are int(z_i > 0)
    off-diagonal elements are zero
    :param z: int or np.array
    :return: int or np.array
    """
    if isinstance(z, int):
        return 1 if z > 0 else 0
    elif isinstance(z, np.ndarray):
        return np.diag(np.greater(z, np.zeros(z.shape)).astype(int))
    else:
        raise TypeError


def softmax(z):
    if isinstance(z, np.ndarray):
        return np.exp(z) / np.sum(np.exp(z))
    else:
        raise TypeError


def softmaxprime(z):
    """
    (i, j)th element is d(a_i)/d(z_j)
    diagonal elements are a_i * (1 - a_i)
    off-diagonal elements are a_i * a_j
    :param z: np.array
    :return: np.array
    """
    if isinstance(z, np.ndarray):
        a = softmax(z)
        return -np.dot(colvec(a), colvec(a).T) + np.diag(a**2) + np.diag(a*(1-a))
    else:
        raise TypeError


if __name__ == "__main__":

    z = np.array([-2, 0, 2])

    print(sigmoid(z))
    print([sigmoid(-2), sigmoid(0), sigmoid(2)])

    print(sigmoidprime(z))
    print([sigmoidprime(-2), sigmoidprime(0), sigmoidprime(2)])

    print(relu(z))
    print([relu(-2), relu(0), relu(2)])

    print(reluprime(z))
    print([reluprime(-2), reluprime(0), reluprime(2)])
