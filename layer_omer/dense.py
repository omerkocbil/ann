import os
import sys

sys.path.insert(0, os.path.join(os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))))
from init_omer.initializer import initializer
from summation_omer.initializer import summation as summate
from activation_omer.initalizer import activation as activate

class Dense():

    def __init__(self, neuron=4, weight_init='uniform', bias_init='zeros', 
                 summation='sum', activation='relu'):
        self.neuron = neuron
        self.weight_init = initializer[weight_init]
        self.bias_init = initializer[bias_init]
        self.summation = summate[summation]
        self.activation = activate[activation]
        