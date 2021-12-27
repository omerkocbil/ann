import numpy as np

class MeanSquaredError():
    
    def __init__(self):
        return
    
    def calculate(self, y, yhat):
        loss = np.subtract(y, yhat)
        loss = np.square(loss)
        loss = loss.sum() / len(yhat)
        return loss
    
    def backward(self, y, yhat):
        return -2 * np.mean(np.subtract(y, yhat))