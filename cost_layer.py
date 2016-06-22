import numpy as np
from abc import ABCMeta, abstractmethod


class CostLayer(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._y = None
        self.cost = []

    def sety(self, y_):
        """
        Set y
        :return:
        """
        self._y = y_
        return

    @abstractmethod
    def cprime(self, a):
        """
        Returns dC/da
        :param a: np.array
        :return: np.array
        """
        raise NotImplementedError


class QuadraticCost(CostLayer):
    """
    Cost = 0.5*(y-a)^2
    """
    def cprime(a, y):
        raise NotImplementedError


class CrossEntropyCost(CostLayer):
    """
    Cost = -(y*log(a) + (1-y)*log(1-a))
    """
    def cprime(self, a):
        if self._y is not None:
            return -(a-self._y)/(a*(1-a))
        else:
            raise ValueError


class LogLikelihoodCost(CostLayer):
    """
    Cost = -(y*log(a))
    """
    def cprime(self, a):
        if self._y is not None:
            return -self._y/a
        else:
            raise ValueError


if __name__ == "__main__":

    y = np.array([1, 0, 1])
    a = np.array([0.9, 0.2, 0.85])

    c = LogLikelihoodCost()
    c.sety(y)
    print(c.cprime(a))

    d = CrossEntropyCost()
    d.sety(y)
    print(d.cprime(a))
