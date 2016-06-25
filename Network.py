import numpy as np

from util import colvec

class Network(object):

    def __init__(self, x_, y_, layers_):
        self._x = x_
        self._y = y_
        self._data = zip(x_, y_)
        self._layers = layers_
        self._num_layers = len(layers_)
        self._sizes, self._weights, self._biases = self.weight_initializer(layers_)

    def weight_initializer(self, layers):
        """
        Initialized weights and biases
        Weights is scaled by sqrt(N_inputs)
        :param layers: list of Layers
        :return: list of int, np.array, np.array
        """
        # sizes = [len(self._x[0])]
        sizes = np.array([layer.get_size() for layer in layers])
        weights = np.array([np.random.rand(x, y)/np.sqrt(y) for x, y in zip(sizes[1:], sizes[:-1])])
        biases = np.array([np.random.rand(x) for x in sizes[1:]])
        return sizes, weights, biases

    def feedforward(self):
        return NotImplementedError

    def backpropagate(self):
        return NotImplementedError


if __name__ == "__main__":
    # x = np.array([[-2, 0, 2], [-2, 0, 2]])
    # y = np.array([[0, 1, 0], [0, 1, 0]])
    #
    # from layer import Sigmoid, Relu, Softmax
    # from output_layer import SigmoidOutput, SoftmaxOutput
    # from cost import CrossEntropyCost, LogLikelihoodCost
    #
    # l1 = Sigmoid(3)
    # l2 = Relu(3)
    # l3 = Softmax(3)
    # # l4 = SoftmaxOutput(3, LogLikelihoodCost)
    # l4 = SigmoidOutput(3, CrossEntropyCost)
    # layers = np.array([l1, l2, l3, l4])
    #
    # nn = Network(x, y, layers)
    #
    # ########## TO BE DELETED LATER
    # b = np.array([-5, 1, 2])
    # w = np.array([[1,2,3],[2,3,4],[-2,3,-5]])
    #
    # for i in range(nn._num_layers):
    #     nn._weights[i] = w
    #     nn._biases[i] = b
    # ##########
    #
    # data = next(nn._data)  # later, put it in for loop of size m
    # x, y = data
    #
    # a = x  # x from network
    # for l in layers:
    #     a = l.feedforward(b, w, a)
    #
    # # for l in layers:
    # #     print(l.get_a())
    #
    # cost, accuracy = layers[-1].evaluate(y)  # y from network
    # # print(cost, accuracy)
    #
    # delta = layers[-1].delta_L(y)
    # for l in layers[::-1][1:]:
    #     delta = l.backpropagate(delta, w)
    #
    # # for l in layers[::-1]:
    # #     print(l.get_delta())
    #
    # for l in layers[1:]:
    #     l.get_delta()
    #
    # delta = layers[0].get_delta()
    # a = x
    #
    # Delta_w = np.dot(colvec(delta), colvec(a).T)
    # Delta_b = delta
    #
    # print(Delta_w)
    # print(Delta_b)
    #
    # for l1, l2 in zip(layers[:-1], layers[1:]):
    #     a = l1.get_a()
    #     delta = l2.get_delta()
    #
    #     Delta_w = np.dot(colvec(delta), colvec(a).T)
    #     Delta_b = delta
    #
    #     print(Delta_w)
    #     print(Delta_b)

    x = np.array([[-2, 0, 2], [-2, 0, 2]])
    y = np.array([[0, 1, 0], [0, 1, 0]])

    from layer import Sigmoid, Relu, Softmax
    from input_layer import Input
    from output_layer import SigmoidOutput, SoftmaxOutput
    from cost import CrossEntropyCost, LogLikelihoodCost

    l0 = Input(3)  # this should be size of x[0]
    l1 = Sigmoid(3)
    l2 = Relu(3)
    l3 = Softmax(3)
    # l4 = SoftmaxOutput(3, LogLikelihoodCost)
    l4 = SigmoidOutput(3, CrossEntropyCost)
    layers = np.array([l0, l1, l2, l3, l4])

    nn = Network(x, y, layers)

    ########## TO BE DELETED LATER
    b = np.array([-5, 1, 2])
    w = np.array([[1,2,3],[2,3,4],[-2,3,-5]])

    for i in range(nn._num_layers-1):
        nn._weights[i] = w
        nn._biases[i] = b
    ##########


    data = next(nn._data)  # later, put it in for loop of size m
    x, y = data

    nn._layers[0].set_a(x)

    a = nn._layers[0].get_a()  # x from network
    for l in layers[1:]:
        a = l.feedforward(b, w, a)

    cost, accuracy = layers[-1].evaluate(y)  # y from network

    delta = layers[-1].delta_L(y)
    for l in layers[::-1][1:]:
        delta = l.backpropagate(delta, w)

    Delta_w = []
    Delta_b = []
    for l1, l2 in zip(layers[:-1], layers[1:]):
        Delta_wi = np.dot(colvec(l2.get_delta()), colvec(l1.get_a()).T)
        Delta_bi = l2.get_delta()
        Delta_w.append(Delta_wi)
        Delta_b.append(Delta_bi)

    print(nn._weights)
    print(np.array(Delta_w))

    print(nn._biases)
    print(np.array(Delta_b))
