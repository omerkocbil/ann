import numpy as np

import os
import sys

sys.path.insert(0, os.path.join(os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))))
from loss.initializer import loss as lose

class Network():

    def __init__(self):
        self.network = []
        self.weights, self.biases = [], []
    
    def add(self, layer):
        self.network.append(layer)
    
    def build(self):
        if not self.network:
            return "Please create a network"
        elif len(self.network) < 3:
            return "The network must contain at least 1 hidden layer"
        
        self.network[0].weight_init = None
        self.network[0].bias_init = None
        self.network[0].summation = None
        self.network[0].activation = None
        
        for layer in range(len(self.network)):
            if layer != 0:
                size = (self.network[layer-1].neuron, self.network[layer].neuron)
                weights = self.network[layer].weight_init.build(size[::-1])
                self.network[layer].weights = weights
                self.weights.append(weights)
                
                bias = self.network[layer].bias_init.build()
                self.network[layer].bias = bias
                self.biases.append(bias)
    
    def forward(self, X):
        if not self.weights:
            self.build()
        
        input = X
        for layer in range(len(self.network)):
            if layer == 0:
                continue
            
            w_x = np.multiply(input, self.network[layer].weights)    
            output = self.network[layer].summation.forward(w_x)
            output = output + self.network[layer].bias
            output = self.network[layer].activation.forward(output)
            
            if layer != len(self.network) - 1:
                input = output
        
        return output
    
    def config(self, loss=None, optimizer='sgd'):
        self.loss_class = None if not loss else lose[loss]
    
    def loss(self, y, yhat):
        return self.loss_class.calculate(y, yhat)