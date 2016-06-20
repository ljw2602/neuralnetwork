import numpy as np
from abc import ABCMeta, abstractmethod


class HiddenLayer(object):
    __metaclass__ = ABCMeta

    def __init__(self, size_):
        self.__size = size_
        self.__z = None  # will be assigned in sigma()

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
    def sigmaprime(self):
        """
        Returns da/dz = d(sigma(z))/dz
        :return: np.array
        """
        return

    @abstractmethod
    def delta(self, w, delta2):
        """
        Return delta_{l} = dot(w_{l+1}.T, delta_{l+1}) * sigma'(z_{l}))
        :param w: np.array
        :param delta2: np.array
        :return: np.array
        """
        return


class Sigmoid(HiddenLayer):

    def __init__(self, size_):
        super(Sigmoid, self).__init__(size_)

    def sigma(self, z_):
        self.__z = z_
        return 1 / (1 + np.exp(-self.__z))

    def sigmaprime(self):
        return self.sigma(self.__z) * (1 - self.sigma(self.__z))

    def delta(self, w, delta2):
        return np.dot(w.T, delta2) * self.sigmaprime()


class Relu(HiddenLayer):

    def __init__(self, size_):
        super(Relu, self).__init__(size_)

    def sigma(self, z_):
        self.__z = z_
        return np.maximum(self.__z, np.zeros(self.__z.shape))

    def sigmaprime(self):
        return np.greater(self.__z, np.zeros(self.__z.shape)).astype(int)


if __name__ == "__main__":

    z = np.array([-2, 0, 2])

    a = Sigmoid(3)
    print(a.sigma(z))
    print(a.sigmaprime())

    b = Relu(3)
    print(b.sigma(z))
    print(b.sigmaprime())
