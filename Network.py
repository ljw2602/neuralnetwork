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
    1