import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidprime(x):
    return sigmoid(x) * (1-sigmoid(x))


def relu(x):
    if isinstance(x, int):
        return np.maximum(x, 0)
    elif isinstance(x, np.ndarray):
        return np.maximum(x, np.zeros(x.shape))


def reluprime(x):
    if isinstance(x, int):
        return 1 if x > 0 else 0
    elif isinstance(x, np.ndarray):
        return np.greater(x, np.zeros(x.shape)).astype(int)
    else:
        raise TypeError


def softmax(x):
    if isinstance(x, np.ndarray):
        return np.exp(x) / np.sum(np.exp(x))
    else:
        raise TypeError


def softmaxprime(x):
    if isinstance(x, np.ndarray):
        raise NotImplementedError
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
