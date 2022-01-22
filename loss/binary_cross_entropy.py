import numpy as np

class BinaryCrossEntropy():
    
    def __init__(self):
        return
    
    def calculate(self, y, yhat):
        if not isinstance(y, np.ndarray): y = np.array(y)
        if not isinstance(yhat, np.ndarray): yhat = np.array(yhat)
        
        clipped_yhat = np.clip(yhat, 1e-7, 1 - 1e-7)
        
        return -((y * np.log(clipped_yhat)) + ((1-y) * np.log(1-clipped_yhat)))
    
    def backward(self, y, yhat):
        clipped_yhat = np.clip(yhat, 1e-7, 1 - 1e-7)
        
        return (-y/clipped_yhat) + ((1-y) / (1-clipped_yhat))