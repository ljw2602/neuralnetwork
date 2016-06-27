def main():
    import numpy as np

    from data import load_data_wrapper
    from layer import Sigmoid, Relu, Softmax
    from layer_input import Input
    from layer_output import SigmoidOutput, SoftmaxOutput
    from cost import CrossEntropyCost, LogLikelihoodCost
    from network import Network

    # X = np.array([[[-2], [0], [2]], [[-2], [0], [2]]])
    # Y = np.array([[[0], [1], [0]], [[0], [1], [0]]])
    # data = list(zip(X, Y))

    training_data, validation_data, test_data = load_data_wrapper()
    training_data = training_data[:1000]

    l0 = Input(784)  # this should be size of x[0]
    l1 = Sigmoid(30)
    # l2 = Relu(3)
    # l3 = Softmax(3)
    # l4 = SoftmaxOutput(10, LogLikelihoodCost)
    l4 = SigmoidOutput(10, CrossEntropyCost)
    layers = np.array([l0, l1, l4])

    epoch = 400
    mini_batch_size = 10
    eta = 0.5
    lam = 0.1

    nn = Network(layers, eta, mini_batch_size, epoch, lam)

    ########## TO BE DELETED LATER
    # b = np.array([-5, 1, 2])
    # w = np.array([[1,2,3],[2,3,4],[-2,3,-5]])
    # from util import colvec
    # for i in range(nn._num_layers-1):
    #     nn._weights[i] = w
    #     nn._biases[i] = colvec(b)
    ##########

    training_cost, training_accuracy, evaluation_cost, evaluation_accuracy = nn.sgd(training_data, monitor_training_data = True, evaluation_data=validation_data, monitor_evaluation_data=True)
    import csv
    with open("training_cost.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(training_cost)
    with open("training_accuracy.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(training_accuracy)
    with open("evaluation_cost.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(evaluation_cost)
    with open("evaluation_accuracy.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(evaluation_accuracy)

    # print(training_cost)
    # print(training_accuracy)
    # print(evaluation_cost)
    # print(evaluation_accuracy)


main()