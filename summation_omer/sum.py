
class Sum():
    
    def __init__(self):
        return
    
    def forward(self, wn_xn):
        if wn_xn.ndim == 1:
            return 0 if not wn_xn else wn_xn[0] + self.build(wn_xn[1:])
        elif wn_xn.ndim == 2:
            return wn_xn.sum(axis=1)
