import numpy as np
from abc import ABCMeta, abstractmethod


class OutputLayer(object):
    __metaclass__ = ABCMeta

    def __init__(self, size_):
        self.__size = size_

    def size(self):
        return self.__size

    @abstractmethod
    def sigma(self, z):
        """
        Returns a = sigma(z)
        :param z: np.array
        :return: np.array
        """
        return

    @abstractmethod
    def delta(self, z, y):
        """
        Returns delta^L = dC/dz^L
        :param z: np.array
        :param y: np.array
        :return: np.array
        """
        return


class SigmoidOutput(OutputLayer):

    def __init__(self, size_):
        super(SigmoidOutput, self).__init__(size_)

    def sigma(self, z):
        return 1 / (1 + np.exp(-z))

    def delta(self, z, y):
        return SigmoidOutput.sigma(z)-y


class SoftmaxOutput(OutputLayer):

    def __init__(self, size_):
        super(SoftmaxOutput, self).__init__(size_)

    def sigma(self, z):
        return np.exp(z) / np.sum(np.exp(z))

    def delta(self, z, y):
        return SoftmaxOutput.sigma(z)-y


if __name__ == "__main__":

    z = np.array([-2, 0, 2])
    y = np.array([0, 1, 1])

    a = SigmoidOutput(3)
    print(a.delta(z, y))

    b = SoftmaxOutput(3)
    print(b.delta(z, y))
