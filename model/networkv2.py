import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))))
from loss.initializer import loss as lose

class Network():
    graph = []

    def __init__(self):
        self.network = []
        self.weights, self.biases = [], []

    def forward(self):
        raise "Implement forward method for your model."

    def config(self, loss='mse', optimizer='sgd'):
        self.loss_class = lose[loss]
        self.optimizer_class = "PASS" #TODO add optimizer

    def loss(self, y, yhat):
        self.loss_value = self.loss_class.calculate(y, yhat)
        return self.loss_value

    def backward(self, loss):
        graph = Network.graph
        delta_1 = graph[-1][0].activation.backward(loss)
        for i in range(len(graph)):

            layer = graph[-(i+1)][0] #Dense class instance
            net = graph[-(i+1)][2] #value activiton function got while forward passin
            dvalue =  np.multiply(delta_1, layer.activation.backward(net))
            print(dvalue)
            delta_1 = np.matmul(dvalue.T, layer.weights)

            # print(delta_1)


        Network.graph = []