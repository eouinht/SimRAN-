import numpy as np
import math

class Topology:
    def __init__(self, n_cells, area=1000):
        self.macro_pos = np.array([[0.0, 0.0]])

        self.small_pos = []
        for i in range(5):
            ang = i * 2*math.pi / 5
            self.small_pos.append([
                400*np.cos(ang),
                400*np.sin(ang)
            ])

        self.small_pos = np.array(self.small_pos)

        self.all_pos = np.vstack([self.macro_pos, self.small_pos])