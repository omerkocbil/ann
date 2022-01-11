import numpy as np

class BinaryCrossEntropy():
    
    def __init__(self):
        return
    
    def calculate(self, y, yhat):
        return -(y*np.log(yhat) + ((1-y)*np.log(1-yhat)))
    
    def backward(self, y, yhat):
        return -2 * np.mean(np.subtract(y, yhat))


bce = BinaryCrossEntropy()
bce.calculate(np.array([0, 1]), np.array([0.8, 0.8]))