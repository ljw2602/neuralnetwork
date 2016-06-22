import numpy as np


class Network(object):

    def __init__(self, layers_):
        self.num_layers = len(layers_)
        self.sizes, self.weights, self.biases = self.weight_initializer(layers_)

    def weight_initializer(self, layers):
        """
        Initialized weights and biases
        Weights is scaled by sqrt(N_inputs)
        :param layers: list of Layers
        :return: list of int, np.array, np.array
        """
        sizes = [layer.size() for layer in layers]
        weights = [np.random.rand(x, y)/np.sqrt(y) for x, y in zip(sizes[1:], sizes[:-1])]
        biases = [np.random.rand(x, 1) for x in sizes[1:]]
        return sizes, weights, biases

    def feedforward(self):
        return NotImplementedError

    def backpropagation(self):
        return NotImplementedError


if __name__ == "__main__":
    l1 = inputlayer(100, X)
    l2 = Relu(3)
    l3 = Sigmoid(5)
    l4 = outputlayer(10, XYcost())
    network(X, Y, cost, [l1, l2, l3, l4])  # call setx(), sety(), define W, b,

    a1 = l1.feedforward()

    z2 = np.dot(w1,a1)
    a2 = l2.feedforward(z2)

    z3 = np.dot(w2,a2)
    a3 = l3.feedforward(z3)

    z4 = np.dot(w3,a3)
    a4 = l4.feedforward(z4)

    y = cost(a4)

    delta4 = l4.backpropagate(y)

    delta3 = l3.backpropagate(delta4, w3)

    delta2 = l2.backpropagate(delta3, w2)

    update(w1, l1.a(), l2.delta())
    update(w2, l2.a(), l3.delta())
    update(w3, l3.a(), l4.delta())