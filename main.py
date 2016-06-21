import numpy as np

from data import load_data_wrapper
from input_layer import InputLayer
from hidden_layer import Sigmoid, Relu
from output_layer import SigmoidOutput, SoftmaxOutput
from network import Network

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