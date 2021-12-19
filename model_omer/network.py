import os
import sys

sys.path.insert(0, os.path.join(os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))))
from layer_omer.dense import Dense

class Network():

    def __init__(self):
        self.network = []
        self.weights = []
        self.biases = []
    
    def add(self, layer):
        self.network.append(layer)
        
            