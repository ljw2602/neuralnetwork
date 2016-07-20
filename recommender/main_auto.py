def main():
    import numpy as np
    from timeit import default_timer as timer

    from recommender.data_auto import load_data_wrapper
    from neuralnetwork.layer import Sigmoid, Relu, Softmax
    from neuralnetwork.layer_input import Input
    from neuralnetwork.layer_output import SigmoidOutput, SoftmaxOutput
    from neuralnetwork.cost import CrossEntropyCost, LogLikelihoodCost
    from neuralnetwork.network import Network

    training_data, validation_data, test_data, n_bin = load_data_wrapper()
    # training_data = training_data[:100]

    l0 = Input(training_data[0][0].size)
    l1 = Relu(200, 0.0)  # or Sigmoid(50, 0.5), Softmax(50, 0.5)
    # l2 = Relu(100, 0.0)
    l3 = SigmoidOutput(n_bin, CrossEntropyCost)  # SoftmaxOutput(n_bin, LogLikelihoodCost)
    layers = np.array([l0, l1, l3])

    epoch = 1000
    mini_batch_size = 200
    eta = 0.05
    lam = 0.0

    nn = Network(layers, eta, mini_batch_size, epoch, lam)

    start = timer()
    training_cost, training_accuracy, \
    validation_cost, validation_accuracy, \
    test_cost, test_accuracy = nn.sgd(training_data,
                                      monitor_training_data = False,
                                      evaluation_data = validation_data,
                                      monitor_evaluation_data = True,
                                      test_data = test_data,
                                      output=True
                                      )

    duration = timer() - start
    print("Total time is %f seconds." % duration)
    if test_accuracy:
        print("Test accuracy is %.2f%%." % (test_accuracy * 100.0))
    else:
        print("Last validation accuracy is %.2f%%." % (validation_accuracy[-1] * 100.0))


main()
#import cProfile as profile
#profile.run("main()", sort="time")
