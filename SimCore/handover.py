import numpy as np

class Association:
    def __init__(self, n_cells):
        self.serving = None
    def step(self, action, rsrp):
        ho = 0
        best = np.argmax(rsrp)
        # Neu chua co serving cell
        if self.serving is None:
            self.serving = best
            return self.serving, 0
        
        # Neu yeu ca HO
        if action["ho"] == 1:
            if rsrp[best] - rsrp[self.serving] > 1.0:
          
                self.serving = best
                ho = 1
        return self.serving, ho
    
        