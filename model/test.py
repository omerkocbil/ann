import numpy as np
import pdb

import os
import sys

sys.path.insert(0, os.path.join(os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))))
from layer.dense import Dense
from model.network import Network
import optimizer.sgd as optimizer


model = Network()

model.add(Dense(2, weight_init='uniform', bias_init='zero', 
                summation='sum', activation='relu'))
model.add(Dense(8, summation='sum', activation='relu'))
model.add(Dense(1))

model.build()
print(model.weights)

SGD = optimizer.SGD(learning_rate=0.01)
model.config(loss='mse', optimizer=SGD)

X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
y = [[1], [1], [0], [0]]

results = [-1, -1, -1, -1]
for epoch in range(5000):
    for i, x in enumerate(X):
        yhat = model.forward(x)
        results[i] = yhat

        loss = model.loss(y[i], yhat)
        #print('loss:', loss)

        dw, w = model.backward(y[i], yhat)
        #print('dweights:', dw)
        #print('weights:', w)
    
    print('results:', results)