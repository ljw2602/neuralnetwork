import numpy as np

from data import load_data_wrapper
from InputLayer import InputLayer
from HiddenLayer import Sigmoid, Relu
from OutputLayer import SigmoidOutput, SoftmaxOutput
from Network import Network

def run():
    training_data, validation_data, test_data = load_data_wrapper()

    net = [InputLayer(5),
           Sigmoid(10),
           Relu(20),
           SoftmaxOutput(3),
           ]
    nn = Network(net)
    print(nn.sizes)
    print([s.shape for s in nn.biases])
    print([s.shape for s in nn.weights])

run()