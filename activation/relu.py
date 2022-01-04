import numpy as np

class ReLu():
    
    def __init__(self):
        return
    
    def forward(self, output):
        if isinstance(output, float):
            return np.max(0, output)
        elif isinstance(output, np.ndarray):
            return np.maximum(np.zeros(output.size), output)
    
    def backward(self, net):
        if isinstance(net, float):
            net = 1 if net>0 else 0
        elif isinstance(net, np.ndarray):
            net[net>0], net[net<=0] = 1, 0
        
        dnet = net
        return dnet