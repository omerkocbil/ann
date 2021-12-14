import numpy as np

class ReLu():
    
    def __init__(self):
        return
    
    def build(self, output):
        return np.max(0, output)