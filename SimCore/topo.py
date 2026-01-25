import numpy as np
class Topology:
    def __init__(self, n_cells=10, isd=200):
        self.n_cells = n_cells
        self.isd = isd 
        self.position = np.arange(n_cells)*isd
        self.min_x = self.position[0]
        self.max_x = self.position[-1]
    def wrap(self, position):
        L = self.max_x
        if position < self.min_x:
            return position + L
        if position > self.max_x:
            return position - L
        return position      
        
    def move(self, position):
        return self.wrap(position)
        
