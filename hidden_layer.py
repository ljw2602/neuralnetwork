import numpy as np
from abc import ABCMeta, abstractmethod

import activation_function as fn


class HiddenLayer(object):
    __metaclass__ = ABCMeta

    def __init__(self, size_):
        self._size = size_
        self._z = None  # will be assigned in sigma()
        self._delta = None

    def size(self):
        """
        Return size
        :return: int
        """
        return self._size

    def feedforward(self, z):
        """
        Save z, and return a = sigma(z)
        :param z: np.array
        :return: np.array
        """
        self._z = z
        return self.sigma()

    def backpropagate(self, delta_right, w_right):
        """
        Calculate, save, and return delta = dC/dz = dot(w_right, delta_right) * sigmaprime(z)
        :param delta_right: np.array, delta from (l+1)th layer
        :param w_right: np.array, w connecting (l)th and (l+1)th layer
        :return: np.array
        """
        self._delta = np.dot(w_right.T, delta_right) * self.sigmaprime(self._z)
        return self._delta

    def a(self):
        """
        Return a = sigma(z)
        :return: np.array
        """
        return self.sigma()

    def delta(self):
        """
        Return delta
        :return: np.array
        """
        return self._delta

    @abstractmethod
    def sigma(self):
        """
        Return a = sigma(z)
        :return: np.array
        """
        raise NotImplementedError

    @abstractmethod
    def sigmaprime(self):
        """
        Returns da/dz = d(sigma(z))/dz
        :return: np.array
        """
        raise NotImplementedError


class Sigmoid(HiddenLayer):

    def __init__(self, size_):
        super(Sigmoid, self).__init__(size_)

    def sigma(self):
        return fn.sigmoid(self._z)

    def sigmaprime(self):
        return fn.sigmoidprime(self._z)


class Relu(HiddenLayer):

    def __init__(self, size_):
        super(Relu, self).__init__(size_)

    def sigma(self):
        return fn.relu(self._z)

    def sigmaprime(self):
        return fn.reluprime(self._z)


if __name__ == "__main__":

    z = np.array([-2, 0, 2])

    a = Sigmoid(3)
    print(a.feedforward(z))
    print(a.sigma())
    print(a.sigmaprime())

    b = Relu(3)
    print(b.feedforward(z))
    print(b.sigma())
    print(b.sigmaprime())
