import numpy as np

from layer import Sigmoid, Softmax

class SigmoidOutput(Sigmoid):

    def __init__(self, size_, cost_):
        super(SigmoidOutput, self).__init__(size_)
        self._cost = cost_()

    def delta_L(self, z_L, a_L, y):
        return np.dot(self.sigmaprime(z_L).T,  self._cost.cprime(a_L, y))

    def cost(self, a, y):
        return self._cost.cost(a, y)

    def accuracy(self, a, y):
        return int(np.argmax(a) == np.argmax(y))


class SoftmaxOutput(Softmax):

    def __init__(self, size_, cost_):
        super(SoftmaxOutput, self).__init__(size_)
        self._cost = cost_()

    def cost(self, a, y):
        return self._cost.cost(a, y)

    def accuracy(self, a, y):
        return int(np.argmax(a) == np.argmax(y))

    def delta_L(self, z_L, a_L, y):
        return np.dot(self.sigmaprime(z_L).T,  self._cost.cprime(a_L, y))


if __name__ == "__main__":

    z = np.array([-2, 0, 2])
    y = np.array([0, 1, 1])

    a = SigmoidOutput(3)
    print(a.delta(z, y))

    b = SoftmaxOutput(3)
    print(b.delta(z, y))
