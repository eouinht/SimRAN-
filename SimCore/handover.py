import numpy as np
from config import SimConfig

class Association:
    def __init__(self):
        self.serving = None

    def step(self, rsrp_vec):

        best = np.argmax(rsrp_vec)

        if self.serving is None:
            self.serving = best
            return best,0

        if rsrp_vec[best] - rsrp_vec[self.serving] > SimConfig.HYSTERESIS:
            self.serving = best
            return best,1

        return self.serving,0
