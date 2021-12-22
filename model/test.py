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
model.add(Dense(2))

model.build()
print(model.network)
print(model.weights)
print(model.biases)

X = np.array([1, 1, 1, 1])
model.forward(X)
