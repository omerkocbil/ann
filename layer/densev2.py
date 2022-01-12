import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))))
from layer_initialization.initializer import initializer
from summation.initializer import summation as summate
from activation.initalizer import activation as activate
from model.networkv2 import Network

class Dense():

    def __init__(self, input_size=4, output_size=4, weight_init='uniform', bias_init='zero',
                 summation='sum', activation='relu'):

        self.input_size = input_size
        self.output_size = output_size

        self.weights = self.initialize_weight(weight_init)
        self.bias = self.initialize_bias(bias_init)
        self.summation = summate[summation]
        self.activation = activate[activation]


    def initialize_weight(self, weight_init):
        size = (self.input_size, self.output_size)
        weights =  initializer[weight_init].build(size[::-1])
        return weights

    def initialize_bias(self, bias_init):
        return initializer[bias_init].build() #TODO one bias per layer? not neuron?

    def __call__(self, input):
        w_x = np.multiply(input, self.weights)
        output = self.summation.forward(w_x)
        to_activition = output + self.bias
        output = self.activation.forward(to_activition)
        Network.graph.append([self, output, to_activition])
        return output