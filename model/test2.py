import os
import sys

sys.path.insert(0, os.path.join(os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))))
from model.network import Network #TODO this will change to new Network
from layer.densev2 import Dense
from loss.mean_squared_error import MeanSquaredError
# from optimizer.sgd import SGD

import numpy as np
np.random.seed(1)

class CustomModel(Network):

    def __init__(self):
        super().__init__()

        
        self.hidden_layer_1 = Dense(4,8, weight_init='uniform', bias_init='zero', summation='sum', activation='relu')
        self.output_layer = Dense(8,1, summation='sum', activation='relu')
        self.config(loss='mse', optimizer='sgd')


    def forward(self, X):
        x = self.hidden_layer_1(X)
        x = self.output_layer(x)

        return x



model = CustomModel()

X = np.array([1, 1, 1, 1])
y = 7.5

yhat = model.forward(X)
print(yhat)

loss = model.loss(y, yhat)
print(loss)
