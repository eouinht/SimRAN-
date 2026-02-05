import numpy as np
import random
from config import SimConfig

class UE:
   
    def __init__(self, ue_id, pos, speed_min, speed_max, area):
        self.id = ue_id
        self.pos = pos
        self.speed = np.random.uniform(speed_min, speed_max)
        self.area = area
        self.traffic = np.random.uniform(0.05, 0.2)
        
        self.serving_cell = None
        self.sinr = 0
        self.rsrp = 0
        self.rsrq = 0
        self.throughput = 0
        
    def move(self):
        theta = np.random.uniform(0, 2*np.pi)
        dx = self.speed * np.cos(theta)
        dy = self.speed * np.sin(theta)

        self.pos += np.array([dx, dy])
        self.pos = np.clip(self.pos, 0, self.area)