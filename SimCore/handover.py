import numpy as np

class Association:
    def __init__(self, n_cells):
        self.serving = np.random.randint(n_cells)
    def step(self, action, rsrp):
        ho = 0
        if action == "HO":
            self.serving = np.argmax(rsrp)
            ho = 1
        return self.serving, ho
    
        