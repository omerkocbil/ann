import numpy as np

class BinaryCrossEntropy():
    
    def __init__(self):
        return
    
    def calculate(self, y, yhat):
        return -((y * np.log(yhat)) + ((1-y) * np.log(1-yhat)))
    
    def backward(self, y, yhat):
        return (-y/yhat) + ((1-y) / (1-yhat))