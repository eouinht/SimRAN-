import numpy as np
class UE:
    def __init__(self, topology):
        self.topo = topology
        self.position = np.random.uniform(self.topo.min_x, self.topo.max_x)
        self.speed = np.random.uniform(-2, 2)
        self.demand = np.random.uniform(2, 8)
        
    def move(self):
        self.position += self.speed
        self.position = self.topo.wrap(self.position)
        
         