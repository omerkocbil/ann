import numpy as np

class MeanSquaredError():
    
    def __init__(self):
        return
    
    def calculate(self, y, yhat):
        if isinstance(yhat, float):
            return (y - yhat) ** 2
        elif isinstance(yhat, np.ndarray):
            loss = np.subtract(y, yhat)
            loss = np.square(loss)
            loss = loss.sum() / len(yhat)
            return loss