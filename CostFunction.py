import numpy as np
from abc import ABCMeta, abstractmethod


class CostFunction(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def cost(self, a, y):
        """
        Returns C = cost(a, y)
        :param a: np.array
        :param y: np.array
        :return: double
        """
        return


class Quadratic(CostFunction):

    @staticmethod
    def cost(a, y):
        return np.sum(np.square(a-y)) / 2.0 / y.shape[0]


class CrossEntropy(CostFunction):

    @staticmethod
    def cost(a, y):
        return np.sum(-(y*np.log(a) + (1.0-y)*np.log(1.0-a))) / y.shape[0]


class LogLikelihood(CostFunction):

    @staticmethod
    def cost(a, y):
        return np.sum(-y*np.log(a)) / y.shape[0]


if __name__ == "__main__":
    x = np.array([0.9, 0.2, 0.85])
    y = np.array([1, 0, 1])
    print(Quadratic.cost(x, y))
    print(CrossEntropy.cost(x, y))
    print(LogLikelihood.cost(x, y))
