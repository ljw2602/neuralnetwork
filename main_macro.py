def main():
    import numpy as np
    from timeit import default_timer as timer

    from data_macro import load_data_wrapper
    from layer import Sigmoid, Relu, Softmax
    from layer_input import Input
    from layer_output import SigmoidOutput, SoftmaxOutput
    from cost import CrossEntropyCost, LogLikelihoodCost
    from network import Network

    # X = np.array([[[-2], [0], [2]], [[-2], [0], [2]]])
    # Y = np.array([[[0], [1], [0]], [[0], [1], [0]]])
    # training_data = list(zip(X, Y))
    # validation_data = list(zip(X, Y))

    training_data, validation_data, test_data = load_data_wrapper()
    # training_data = training_data[:100]

    l0 = Input(training_data[0][0].size)
    # l1 = Sigmoid(50, 0.5)
    l2 = Relu(50, 0.5)
    l3 = Relu(50, 0.5)
    # l1 = Relu(100, 0.0)
    # l3 = Softmax(10, 0.0)
    l4 = SoftmaxOutput(5, LogLikelihoodCost)
    # l4 = SigmoidOutput(5, CrossEntropyCost, 0.0)
    layers = np.array([l0, l2, l3, l4])

    epoch = 30
    mini_batch_size = 10
    eta = 0.0001
    lam = 0.0

    nn = Network(layers, eta, mini_batch_size, epoch, lam)

    ########## TO BE DELETED LATER
    # b = np.array([[-5], [1], [2]])
    # w = np.array([[1,2,3],[2,3,4],[-2,3,-5]])
    # from util import colvec
    # for i in range(nn._num_layers-1):
    #     nn._weights[i] = w
    #     nn._biases[i] = b
    ##########
    
    start = timer()
    training_cost, training_accuracy, \
    evaluation_cost, evaluation_accuracy, \
    test_cost, test_accuracy = nn.sgd(training_data,
                                      monitor_training_data = True,
                                      evaluation_data = validation_data,
                                      monitor_evaluation_data = True,
                                      test_data = test_data,
                                      )
    duration = timer() - start
    print("Total time is %f seconds." % duration)
    
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

    print(test_cost)
    print(test_accuracy*100.0)


main()
#import cProfile as profile
#profile.run("main()", sort="time")
