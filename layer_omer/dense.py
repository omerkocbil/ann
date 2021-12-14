import os
import sys
sys.path.insert(0, os.path.join(os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))))
from init_omer.initializer import initializer

class Dense():

    def __init__(self, neuron=4, weight_init='uniform', bias_init='zeros', 
                 summation='sum', activation='relu'):
        self.weight_init = initializer[weight_init]
        self.bias_init = initializer[bias_init]