import numpy as np

class ReLu():
    
    def __init__(self):
        return
    
    def forward(self, output):
        if isinstance(output, float):
            return np.max(0, output)
        elif isinstance(output, np.ndarray):
            return np.maximum(np.zeros(output.size), output)