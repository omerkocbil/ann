import numpy as np

class Sigmoid():
    
    def __init__(self):
        return
    
    def forward(self, net):
        return (1 / (1 + np.exp(-net)))
    
    def backward(self, net):
        return self.forward(net) * (1-self.forward(net))