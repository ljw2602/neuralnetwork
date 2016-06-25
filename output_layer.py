import numpy as np

from layer import Sigmoid, Softmax

class SigmoidOutput(Sigmoid):

    def __init__(self, size_, cost_):
        super(SigmoidOutput, self).__init__(size_)
        self._cost = cost_()

    def cost(self, y):
        return self._cost.cost(self._a, y)

    def accuracy(self, y):
        return int(np.argmax(self._a) == np.argmax(y))

    def delta_L(self, y):
        self._delta = np.dot(self.sigmaprime().T,  self._cost.cprime(self._a, y))
        return self._delta

    def evaluate(self, y):
        return self.cost(y), self.accuracy(y)


class SoftmaxOutput(Softmax):

    def __init__(self, size_, cost_):
        super(SoftmaxOutput, self).__init__(size_)
        self._cost = cost_()

    def cost(self, y):
        return self._cost.cost(self._a, y)

    def accuracy(self, y):
        return int(np.argmax(self._a) == np.argmax(y))

    def delta_L(self, y):
        self._delta = np.dot(self.sigmaprime().T,  self._cost.cprime(self._a, y))
        return self._delta

    def evaluate(self, y):
        return self.cost(y), self.accuracy(y), self.delta(y)


if __name__ == "__main__":

    z = np.array([-2, 0, 2])
    y = np.array([0, 1, 1])

    a = SigmoidOutput(3)
    print(a.delta(z, y))

    b = SoftmaxOutput(3)
    print(b.delta(z, y))
