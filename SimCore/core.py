import numpy as np
from .config import SimConfig
from .channel import sinr
from .traffic import generate_traffic
from .mobility import move
from .scheduler import allocate_prb
from .handover import ho_pen

class SimCore:
    def __init__(self, config=SimConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.ue_pos = np.random.rand(self.config.N_UES, 2) * self.config.AREA
        self.ue_vel = np.random.randn(self.config.N_UES, 2)
        self.serving = np.random.randint(0, self.config.N_CELLS, self.config.N_UES)
        self.queue = np.zeros(self.config.N_UES)
        self.txPower = np.ones(self.config.N_CELLS)*self.config.MAX_TX
        self.cell_pos = np.random.rand(self.config.N_CELLS, 2)*self.config.AREA        
        
    def step(self, action):
        ho = action["handover"]
        prb_ratio = action["prb"]
        power_ratio = action["power"]
        
        new_serving = ho
        ho_cost = sum(new_serving != self.serving)
        self.serving = new_serving
        self.txPower = self.config.MIN_TX + power_ratio*(self.config.MAX_TX - self.config.MIN_TX)
        self.ue_pos = move(self.ue_pos, self.ue_vel, self.config.AREA)
        prb_allocated = allocate_prb(self.serving, prb_ratio, self.config.MAX_PRB, self.config.N_CELLS)
        rates = np.zeros(self.config.N_CELLS)
        
        for i in range(self.config.N_UES):
            s = sinr(i, self.serving[i], self.ue_pos, self.cell_pos, self.txPower, self.config.NOISE, self.config.PATHLOSS)
            rates[i] = prb_allocated[i]*np.log2(1+s)
            
        self.queue += generate_traffic(self.config.N_CELLS, self.config.LAMDA)
        self.queue -= rates
        self.queue = np.clip(self.queue, 0, None)
        
        delay_vio = np.sum(self.queue > self.config.DELAY_MAX)
        energy = np.sum(self.txPower)
        
        return{
            "throughput": np.sum(rates),
            "delay_violation": delay_vio,
            "handover": ho_cost,
            "energy": energy
            
        }
    def get_state(self):
        return{
            "serving": self.serving,
            "queue": self.queue,
            "txPower": self.txPower
        }