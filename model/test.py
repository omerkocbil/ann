import numpy as np

import os
import sys

sys.path.insert(0, os.path.join(os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))))
from layer.dense import Dense
from model.network import Network


model = Network()

model.add(Dense(4, weight_init='uniform', bias_init='zero', 
                summation='sum', activation='relu'))
model.add(Dense(8, summation='sum', activation='relu'))
model.add(Dense(1))

model.build()
print(model.network)
print(model.weights)
print(model.biases)

model.config(loss='mse', optimizer='sgd')

X = np.array([1, 2, -1, 0.5])
y = [7.5]

yhat = model.forward(X)
print(yhat)

loss = model.loss(y, yhat)
print(loss)

model.backward(y, yhat)
