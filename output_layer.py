import numpy as np
from abc import ABCMeta, abstractmethod


import activation_function as fn


class OutputLayer(object):
    __metaclass__ = ABCMeta

    def __init__(self, size_, cost_):
        self._size = size_
        self._z = None
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
        This is different to sigmaprime() in HiddenLayer because it is 2D
        (i,j) element is d(a_i)/d(z_j) = d(sigma(z_i))/d(z_j) <-- ?? check
        :return: 2D np.array
        """
        raise NotImplementedError


class SigmoidOutput(OutputLayer):

    def __init__(self, size_, cost_):
        super(SigmoidOutput, self).__init__(size_, cost_)

    def sigma(self):
        return fn.sigmoid(self._z)

    def sigmaprime(self):
        return fn.sigmoidprime(self._z)

    def delta(self, z, y):
        return SigmoidOutput.sigma(z)-y


class SoftmaxOutput(OutputLayer):

    def __init__(self, size_, cost_):
        super(SoftmaxOutput, self).__init__(size_, cost_)

    def sigma(self, z):
        return fn.softmax(z)

    def sigmaprime(self, z):
        raise NotImplementedError

    def delta(self, z, y):
        return SoftmaxOutput.sigma(z)-y


if __name__ == "__main__":

    z = np.array([-2, 0, 2])
    y = np.array([0, 1, 1])

    a = SigmoidOutput(3)
    print(a.delta(z, y))

    b = SoftmaxOutput(3)
    print(b.delta(z, y))
