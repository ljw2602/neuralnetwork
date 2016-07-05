import numpy as np
from abc import ABCMeta, abstractmethod

import activation_function as fn
from util import matmul


class Layer(object):
    __metaclass__ = ABCMeta

    def __init__(self, size_, dropout_):
        self._size = size_
        assert 0.0 <= dropout_ < 1.0, "Dropout probability must be smaller than 1"
        self._dropout = dropout_
        self._remaining_size = max(1, int(self._size - self._size*self._dropout))
        self._remaining_nodes = None

    def get_size(self):
        """
        Return original node number
        :return: int
        """
        if self._size is not None:
            return self._size
        else:
            raise ValueError

    def get_remaining_size(self):
        """
        Return node number after dropout
        :return: int
        """
        if self._remaining_size is not None:
            return self._remaining_size
        else:
            raise ValueError

    def is_dropout(self):
        """
        Return if this layer uses dropout regularization
        :return: bool
        """
        if self._dropout is not None:
            return self._dropout != 0.0
        else:
            raise ValueError

    def reset_dropout(self):
        """
        Draw random numbers to choose what nodes to be remained after dropout
        :return: np.array
        """
        self._remaining_nodes = np.sort(np.random.choice(self._size, self._remaining_size, replace=False))
        return self._remaining_nodes

    def feedforward(self, b, w, a):
        """
        Save z = b + w a_in, and return a_out = sigma(z)
        :param b: np.array
        :param w: np.array
        :param a: np.array
        :return: np.array
        """
        #z = b + np.dot(w, a)
        z = b + matmul(w, a)
        a = self.sigma(z)
        return z, a

    def backpropagate(self, z, delta_right, w_right):
        """
        Calculate, save, and return delta = dC/dz = dot(w_right, delta_right) * sigmaprime(z)
        :param delta_right: np.array, delta from (l+1)th layer
        :param w_right: np.array, w connecting (l)th and (l+1)th layer
        :return: np.array
        """
        #return np.dot(self.sigmaprime(z).T, np.dot(w_right.T, delta_right))
        return matmul(self.sigmaprime(z).T, matmul(w_right.T, delta_right))

    @abstractmethod
    def sigma(self, z):
        """
        Return a = sigma(z)
        :return: np.array
        """
        raise NotImplementedError

    @abstractmethod
    def sigmaprime(self, z):
        """
        Returns da/dz = d(sigma(z))/dz
        (i,j)th element corresponds to d(a_{i})/d(z_{j})
        :return: np.array
        """
        raise NotImplementedError


class Sigmoid(Layer):

    def __init__(self, size_, dropout_):
        super(Sigmoid, self).__init__(size_, dropout_)

    def sigma(self, z):
        return fn.sigmoid(z)

    def sigmaprime(self, z):
        return fn.sigmoidprime(z)


class Relu(Layer):

    def __init__(self, size_, dropout_):
        super(Relu, self).__init__(size_, dropout_)

    def sigma(self, z):
        return fn.relu(z)

    def sigmaprime(self, z):
        return fn.reluprime(z)


class Softmax(Layer):

    def __init__(self, size_, dropout_):
        super(Softmax, self).__init__(size_, dropout_)

    def sigma(self, z):
        return fn.softmax(z)

    def sigmaprime(self, z):
        return fn.softmaxprime(z)


if __name__ == "__main__":

    from cost import QuadraticCost, CrossEntropyCost, LogLikelihoodCost
    from layer_output import SigmoidOutput, SoftmaxOutput

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
