def main():
    import numpy as np
    from timeit import default_timer as timer

    from sp500.data_macro import load_data_wrapper
    from neuralnetwork.layer import Sigmoid, Relu, Softmax
    from neuralnetwork.layer_input import Input
    from neuralnetwork.layer_output import SigmoidOutput, SoftmaxOutput
    from neuralnetwork.cost import CrossEntropyCost, LogLikelihoodCost
    from neuralnetwork.network import Network

    training_data, validation_data, test_data = load_data_wrapper()
    # training_data = training_data[:100]

    l0 = Input(training_data[0][0].size)
    l1 = Relu(50, 0.5) # or Sigmoid(50, 0.5), Softmax(50, 0.5)
    l2 = Relu(50, 0.5)
    l3 = SoftmaxOutput(5, LogLikelihoodCost)  # or SigmoidOutput(5, CrossEntropyCost, 0.0)
    layers = np.array([l0, l1, l2, l3])

    epoch = 1
    mini_batch_size = 10
    eta = 0.0001
    lam = 0.0

    nn = Network(layers, eta, mini_batch_size, epoch, lam)

    start = timer()
    training_cost, training_accuracy, \
    validation_cost, validation_accuracy, \
    test_cost, test_accuracy = nn.sgd(training_data,
                                      monitor_training_data = True,
                                      evaluation_data = validation_data,
                                      monitor_evaluation_data = True,
                                      test_data = test_data,
                                      output=True
                                      )

    duration = timer() - start
    print("Total time is %f seconds." % duration)
    if test_accuracy:
        print("Test accuracy is %.2f %%." % (test_accuracy * 100.0))
    else:
        print("Last validation accuracy is %.2f %%." % (validation_accuracy[-1] * 100.0))


main()
#import cProfile as profile
#profile.run("main()", sort="time")
