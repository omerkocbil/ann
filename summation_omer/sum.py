import numpy as np
import operator

class Sum():
    
    def __init__(self):
        return
    
    def build(self, wn_xn):
        return 0 if not wn_xn else wn_xn[0] + self.build(wn_xn[1:])