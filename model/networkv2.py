import os
import sys

sys.path.insert(0, os.path.join(os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))))
from loss.initializer import loss as lose

class Network():

    def __init__(self):
        self.network = []
        self.weights, self.biases = [], []

    def forward(self):
        raise "Implement forward method for your model."

    def config(self, loss='mse', optimizer='sgd'):
        self.loss_class = lose[loss]
        self.optimizer_class = "PASS" #TODO add optimizer

    def loss(self, y, yhat):
        return self.loss_class.calculate(y, yhat)