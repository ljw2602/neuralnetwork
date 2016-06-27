import numpy as np
import random

from util import colvec


class Network(object):
    """
    Fully connected neural network

    Supported hidden layer types: sigmoid, softmax, relu
    Supported output layer types: sigmoid, softmax
    Supported cost types: quadratic, cross-entropy, log-likelihood

    Supported hyper-parameters: eta, mini batch size, epoch, lambda
    (L2 regularization is assumed when lambda is passed)
    """
    def __init__(self, layers_, eta_, mini_batch_size_, epoch_, lam_ = None):
        self._layers = layers_
        self._num_layers = len(layers_)
        self._sizes, self._weights, self._biases = self.weight_initializer(layers_)

        # Hyperparameters
        self._eta = eta_
        self._mini_batch_size = mini_batch_size_
        self._epoch = epoch_
        self._lam = lam_

    def weight_initializer(self, layers):
        """
        Initialized weights and biases
        Weights is scaled by sqrt(N_inputs)
        :param layers: list of Layers
        :return: list of int, np.array, np.array
        """
        sizes = np.array([layer.get_size() for layer in layers])
        weights = np.array([np.random.rand(x, y)/np.sqrt(y) for x, y in zip(sizes[1:], sizes[:-1])])
        biases = np.array([colvec(np.random.rand(x)) for x in sizes[1:]])
        return sizes, weights, biases

    def feedforward(self, x):
        self._layers[0].set_a(x)

        a = self._layers[0].get_a()
        for i, ly in enumerate(self._layers[1:]):
            a = ly.feedforward(self._biases[i], self._weights[i], a)
        return

    def backpropagate(self, y):
        delta = self._layers[-1].delta_L(y)
        for i, ly in enumerate(self._layers[::-1][1:]):
            delta = ly.backpropagate(delta, self._weights[self._num_layers-2-i])
        return

    def update(self):
        """
        Calculate dC/dw from one sample
        :return:
        """
        delta_w = []
        delta_b = []
        for l1, l2 in zip(self._layers[:-1], self._layers[1:]):
            delta_w.append( np.dot(colvec(l2.get_delta()), colvec(l1.get_a()).T) )
            delta_b.append( l2.get_delta() )
        return delta_w, delta_b

    def run_minibatch(self, mini_batch):
        """
        Run one mini batch and update the weights
        :param mini_batch: list
        :return:
        """
        delta_w_sum = np.zeros(self._weights.shape)
        delta_b_sum = np.zeros(self._biases.shape)

        for x, y in mini_batch:
            self.feedforward(x)
            self.backpropagate(y)
            delta_w, delta_b = self.update()
            # delta_w_sum += delta_w
            # delta_b_sum += delta_b
            delta_w_sum = [dws+dw for dws, dw in zip(delta_w_sum, delta_w)]
            delta_b_sum = [dbs+db for dbs, db in zip(delta_b_sum, delta_b)]

        delta_w_sum = np.array(delta_w_sum)
        delta_b_sum = np.array(delta_b_sum)

        self._biases -= self._eta * delta_b_sum/len(mini_batch)
        if self._lam is not None:
            self._weights = (1.0 - self._eta*self._lam/len(mini_batch))*self._weights - self._eta*delta_w_sum/len(mini_batch)
        else:
            self._weights -= self._eta * delta_w_sum/len(mini_batch)
        return

    def sgd(self, training_data,
            monitor_training_data = False,
            evaluation_data = None,
            monitor_evaluation_data = False):

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for e in range(self._epoch):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+self._mini_batch_size] for k in range(0, len(training_data), self._mini_batch_size)]

            for mini_batch in mini_batches:
                self.run_minibatch(mini_batch)

            if monitor_training_data:
                total_cost, total_accuracy = self.evaluate(training_data)
                training_cost.append(total_cost)
                training_accuracy.append(total_accuracy)
            if monitor_evaluation_data and evaluation_data is not None:
                total_cost, total_accuracy = self.evaluate(evaluation_data)
                evaluation_cost.append(total_cost)
                evaluation_accuracy.append(total_accuracy)

        return training_cost, training_accuracy, evaluation_cost, evaluation_accuracy

    def evaluate(self, data):
        total_cost = 0.0
        total_accuracy = 0.0

        for x, y in data:
            self.feedforward(x)
            total_cost += self._layers[-1].cost(y)
            total_accuracy += self._layers[-1].accuracy(y)

        if self._lam is not None:
            # total_cost += (self._lam/2.0)*np.sum(self._weights**2)
            total_cost += (self._lam/2.0)*sum([x.sum() for x in (self._weights**2)])

        total_cost /= len(data)
        total_accuracy /= len(data)

        return total_cost, total_accuracy

if __name__ == "__main__":

    X = np.array([[-2, 0, 2], [-2, 0, 2]])
    Y = np.array([[0, 1, 0], [0, 1, 0]])
    data = list(zip(X, Y))


    from layer import Sigmoid, Relu, Softmax
    from layer_input import Input
    from layer_output import SigmoidOutput, SoftmaxOutput
    from cost import CrossEntropyCost, LogLikelihoodCost

    l0 = Input(3)  # this should be size of x[0]
    l1 = Sigmoid(3)
    l2 = Relu(3)
    l3 = Softmax(3)
    # l4 = SoftmaxOutput(3, LogLikelihoodCost)
    l4 = SigmoidOutput(3, CrossEntropyCost)
    layers = np.array([l0, l1, l2, l3, l4])

    epoch = 1
    mini_batch_size = 2
    eta = 0
    lam = 0.001

    nn = Network(layers, eta, mini_batch_size, epoch, lam)

    ########## TO BE DELETED LATER
    b = np.array([-5, 1, 2])
    w = np.array([[1,2,3],[2,3,4],[-2,3,-5]])

    for i in range(nn._num_layers-1):
        nn._weights[i] = w
        nn._biases[i] = colvec(b)
    ##########

    # nn.sgd(data)

    random.shuffle(data)
    mini_batches = [data[k:k+mini_batch_size] for k in range(0, len(data), mini_batch_size)]

    for mini_batch in mini_batches:
        # print(mini_batch)
        nn.run_minibatch(mini_batch)
        # print(nn._weights)
        # print(nn._biases)

    total_cost, total_accuracy = nn.evaluate(data)
    # total_cost = 0.0
    # total_accuracy = 0.0
    # for x, y in data:
    #     nn.feedforward(x)
    #     total_cost += nn._layers[-1].cost(y)
    #     total_accuracy += nn._layers[-1].accuracy(y)
    #     print(total_cost)
    #     print(total_accuracy)
    #
    # if lam is not None:
    #     total_cost += (lam/2.0)*np.sum(nn._weights**2)
    #
    # total_cost /= len(data)
    # total_accuracy /= len(data)
    #
    print(total_cost, total_accuracy)