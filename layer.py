import numpy as np
from abc import ABCMeta, abstractmethod

import activation_function as fn


class Layer(object):
    __metaclass__ = ABCMeta

    def __init__(self, size_):
        self._size = size_
        self._z = None
        self._a = None
        self._delta = None

    def get_size(self):
        """
        Return size
        :return: int
        """
        if self._size is not None:
            return self._size
        else:
            raise ValueError

    def get_z(self):
        """
        Return z
        :return: np.array
        """
        if self._z is not None:
            return self._z
        else:
            raise ValueError

    def get_a(self):
        """
        Return a = sigma(z)
        :return: np.array
        """
        if self._a is not None:
            return self._a
        else:
            raise ValueError

    def get_delta(self):
        """
        Return delta
        :return: np.array
        """
        if self._delta is not None:
            return self._delta
        else:
            raise ValueError

    def feedforward(self, b, w, a):
        """
        Save z = b + w a_in, and return a_out = sigma(z)
        :param b: np.array
        :param w: np.array
        :param a: np.array
        :return: np.array
        """
        self._z = b + np.dot(w, a)
        self._a = self.sigma()
        return self._a

    def backpropagate(self, delta_right, w_right):
        """
        Calculate, save, and return delta = dC/dz = dot(w_right, delta_right) * sigmaprime(z)
        :param delta_right: np.array, delta from (l+1)th layer
        :param w_right: np.array, w connecting (l)th and (l+1)th layer
        :return: np.array
        """
        self._delta = np.dot(self.sigmaprime().T, np.dot(w_right.T, delta_right))
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
        (i,j)th element corresponds to d(a_{i})/d(z_{j})
        :return: np.array
        """
        raise NotImplementedError


class Sigmoid(Layer):

    def __init__(self, size_):
        super(Sigmoid, self).__init__(size_)

    def sigma(self):
        return fn.sigmoid(self._z)

    def sigmaprime(self):
        return fn.sigmoidprime(self._z)


class Relu(Layer):

    def __init__(self, size_):
        super(Relu, self).__init__(size_)

    def sigma(self):
        return fn.relu(self._z)

    def sigmaprime(self):
        return fn.reluprime(self._z)


class Softmax(Layer):

    def __init__(self, size_):
        super(Softmax, self).__init__(size_)

    def sigma(self):
        return fn.softmax(self._z)

    def sigmaprime(self):
        return fn.softmaxprime(self._z)


if __name__ == "__main__":

    from cost import QuadraticCost, CrossEntropyCost, LogLikelihoodCost
    from output_layer import SigmoidOutput, SoftmaxOutput

    b = np.array([-5, 1, 2])
    w = np.array([[1,2,3],[2,3,4],[-2,3,-5]])

    x = np.array([-2, 0, 2])
    y = np.array([0, 1, 0])

    l1 = Sigmoid(3)
    l2 = Relu(3)
    l3 = Softmax(3)
    # l4 = SoftmaxOutput(3, LogLikelihoodCost)
    l4 = SigmoidOutput(3, CrossEntropyCost)
    layers = [l1, l2, l3, l4]

    a = x  # x from network
    for l in layers:
        a = l.feedforward(b,w,a)
        print(a)

    cost, accuracy, delta = layers[-1].evaluate(y)  # y from network
    print(cost, accuracy)
    print(delta)

    # delta = layers[-2].backpropagate(delta, w)

    for l in layers[::-1][1:]:
        delta = l.backpropagate(delta, w)
        print(delta)

    # print(l1.feedforward(b, w, x))
    # print(l1.sigma())
    # print(l1.sigmaprime())
    #
    # print(l2.feedforward(b, w, x))
    # print(l2.sigma())
    # print(l2.sigmaprime())


    # l3 = SigmoidOutput(3)
    # print(l3.feedforward(b, w, x))
    # print(l3.sigma())
    # print(l3.sigmaprime())

    # from cost_layer import CrossEntropyCost, LogLikelihoodCost
    # print("---")
    # c1 = CrossEntropyCost()
    # print(c1.cprime(l3.sigma(), y))
    # print(c1.cost(l3.sigma(), y))
    #
    # c2 = LogLikelihoodCost()
    # print(c2.cprime(l3.sigma(), y))
    # print(c2.cost(l3.sigma(), y))