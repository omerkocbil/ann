import numpy as np

import os
import sys

sys.path.insert(0, os.path.join(os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))))
from loss.initializer import loss as lose

class Network():

    def __init__(self):
        self.network = []
        self.weights, self.biases, = [], []
        self.nets, self.outputs = [], []
    
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
                
                bias = self.network[layer].bias_init.build(self.network[layer].neuron)
                self.network[layer].bias = bias
                self.biases.append(bias)
        
        self.dweights = []
        for layer in range(len(self.weights)):
            self.dweights.append(np.full_like(self.weights.copy()[layer], 0))
        
        self.dbiases = []
        for layer in range(len(self.biases)):
            self.dbiases.append(np.full_like(self.biases.copy()[layer], 0))
    
    def forward(self, X):
        if not self.weights:
            self.build()
        
        input = X.copy()
        self.nets.append(input)
        self.outputs.append(input)
        for layer in range(len(self.network)):
            if layer == 0:
                continue
            
            w_x = np.multiply(input, self.network[layer].weights)    
            output = self.network[layer].summation.forward(w_x)
            
            output = output + self.network[layer].bias
            self.nets.append(output)
            
            output = self.network[layer].activation.forward(output)
            self.outputs.append(output)
            
            if layer != len(self.network) - 1:
                input = output
        
        return output
    
    def config(self, loss=None, optimizer=None):
        self.loss_class = None if not loss else lose[loss]
        self.optimizer = optimizer
    
    def loss(self, y, yhat):
        return self.loss_class.calculate(y, yhat)
    
    def backward(self, y, yhat):
        loss = []
        for index in range(self.network[-1].neuron):
            dloss = self.loss_class.backward(y[index], yhat[index])
            loss.append(dloss)

        for index, layer in zip(range(len(self.network)-1, 0, -1), reversed(self.network)):
            doutputs = [np.array(loss)]
            for current_layer in range(len(self.network)-1, index-1, -1):
                if current_layer != index:
                    doutputs.append(np.zeros(self.network[current_layer-1].neuron))
                    for to_neuron in range(self.network[current_layer-1].neuron):
                        for from_neuron in range(self.network[current_layer].neuron): 
                            dactivation = layer.activation.backward(self.nets[current_layer][from_neuron])
                            dsummation = layer.summation.backward(self.weights[current_layer-1][from_neuron][to_neuron])
                            doutputs[-1][to_neuron] += doutputs[-2][from_neuron] * dactivation * dsummation
                
                elif current_layer == index:  
                    for from_neuron in range(self.network[current_layer].neuron):   
                        for to_neuron in range(self.network[current_layer-1].neuron):
                            dactivation = layer.activation.backward(self.nets[current_layer][from_neuron])
                            dsummation = layer.summation.backward(self.outputs[current_layer-1][to_neuron])
                            self.dweights[current_layer-1][from_neuron][to_neuron] = doutputs[-1][from_neuron] * dactivation * dsummation
                        self.dbiases[current_layer-1][from_neuron] = doutputs[-1][from_neuron] * dactivation * 1
        
        self.weights, self.biases = self.update_params(weights=self.weights, dweights=self.dweights, 
                                                       biases=self.biases, dbiases=self.dbiases)
        
        return self.dweights, self.weights, self.dbiases, self.biases
    
    def update_params(self, *, weights, dweights, biases, dbiases):
        return self.optimizer.update_params(weights, dweights, biases, dbiases)
    
    def backward2(self, y):
        dinputs = self.loss_class.backward(y, self.outputs[-1])
        for i in range(len(self.network)-1):
            layer = self.network[-(i+1)]
            net = self.nets[-(i+1)]
            dvalues = np.multiply(dinputs, layer.activation.backward(net))
            dinputs = np.matmul(dvalues.T, layer.weights)
            
            outputs = np.expand_dims(self.outputs[-(i+2)], axis=1)
            dvalues = np.expand_dims(dvalues, axis=1)
            self.dweights[-(i+1)] = np.multiply(outputs.T, dvalues)
            self.dbiases[-(i+1)] = dvalues.reshape(len(dvalues),) * 1
        
        self.weights, self.biases = self.update_params(weights=self.weights, dweights=self.dweights, 
                                                       biases=self.biases, dbiases=self.dbiases)

        return self.dweights, self.weights, self.dbiases, self.biases