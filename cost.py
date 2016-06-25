import numpy as np
from abc import ABCMeta, abstractmethod


class Cost(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def cost(self, a, y):
        raise NotImplementedError

    @abstractmethod
    def cprime(self, a, y):
        """
        Returns dC/da
        :param a: np.array
        :param y: np.array
        :return: np.array
        """
        raise NotImplementedError


class QuadraticCost(Cost):
    """
    Cost = 0.5*(y-a)^2
    """
    def cost(self, a, y):
        return 0.5*np.sum((y-a)*(y-a))

    def cprime(self, a, y):
        return (a-y)


class CrossEntropyCost(Cost):
    """
    Cost = -(y*log(a) + (1-y)*log(1-a))
    """
    def cost(self, a, y):
        return np.sum(-(y*np.log(a) + (1.0-y)*np.log(1.0-a)))

    def cprime(self, a, y):
        return (a-y)/a/(1.0-a)


class LogLikelihoodCost(Cost):
    """
    Cost = -(y*log(a))
    """
    def cost(self, a, y):
        return np.sum(-y*np.log(a))

    def cprime(self, a, y):
        return -y/a


if __name__ == "__main__":

    y = np.array([1, 0, 1])
    a = np.array([0.9, 0.2, 0.85])

    c1 = QuadraticCost()
    print(c1.cost(a, y))
    print(c1.cprime(a, y))

    c2 = CrossEntropyCost()
    print(c2.cost(a, y))
    print(c2.cprime(a, y))

    c3 = LogLikelihoodCost()
    print(c3.cost(a, y))
    print(c3.cprime(a, y))
