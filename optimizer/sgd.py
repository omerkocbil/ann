import numpy as np

class SGD():
    
    def __init__(self, learning_rate=None, momentum=None):
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    def update_params(self, weights, dweights):
        if not self.learning_rate and not self.momentum:
            if isinstance(weights, (float, int)):
                weights -= dweights
            elif isinstance(weights, np.ndarray):
                for layer in range(len(weights)):
                    weights[layer] -= dweights[layer]
        elif self.learning_rate and not self.momentum:
            if isinstance(weights, (float, int)):
                weights -=  self.learning_rate * dweights
            elif isinstance(weights, list):
                for layer in range(len(weights)):
                    weights[layer] -= self.learning_rate * dweights[layer]
        elif self.learning_rate and self.momentum:
            pass #TODO: Taha yazacak
        
        return weights