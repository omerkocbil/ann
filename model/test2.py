import os
import sys

sys.path.insert(0, os.path.join(os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))))
from model.networkv2 import Network #TODO this will change to new Network
from layer.densev2 import Dense
from loss.mean_squared_error import MeanSquaredError

import numpy as np
np.random.seed(1)

class CustomModel(Network):

    def __init__(self):
        super().__init__()

        
        self.hidden_layer_1 = Dense(2,4, weight_init='uniform', bias_init='zero', summation='sum', activation='relu')
        self.hidden_layer_2 = Dense(4,2, weight_init='uniform', bias_init='zero', summation='sum', activation='relu')

        self.output_layer = Dense(2,1, summation='sum', activation='relu')
        self.config(loss='mse', optimizer='sgd')


    def forward(self, X):
        x = self.hidden_layer_1(X)
        x = self.hidden_layer_2(x)
        x = self.output_layer(x)

        return x



model = CustomModel()

X = np.array([1, 1])
y = 1

yhat = model.forward(X)

loss = model.loss(y, yhat)
model.backward(loss)