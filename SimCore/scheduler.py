import numpy as np
class Scheduler:
    def __init__(self):
        self.prb = 20
        self. max_prb = 100
    
    def step(self, action):
        if action == "PRB_UP":
            self.prb += 5
        elif action == "PRB_DOWN":
            self.prb -= 5
        self.prb = np.clip(self.prb, 5, self.max_prb)
        