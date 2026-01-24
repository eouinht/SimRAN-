import numpy as np
from .config import SimConfig
from .channel import Channel
from .topo import Topology
from .scheduler import Scheduler
from .handover import Association
from .ue import UE
from .kpi import KPI

class SimCore:
    def __init__(self, n_cells=10):
        self.topo = Topology(n_cells)
        self.ue = UE(self.topo)
        self.channel = Channel()
        self.assoc = Association(n_cells)
        self.scheduler = Scheduler()
    
    def step(self, action):
        self.ue.move()
        
        d = np.abs(self.topo.position - self.ue.position)
        rsrp = self.channel.rsrp(d)
        
        serving, ho = self.assoc.step(action, rsrp)
        rsrp_serv = rsrp[serving]
        
        sinr = self.channel.sinr(rsrp_serv, np.delete(rsrp, serving))
        self.scheduler.step(action)
        sinr_eff = sinr + 0.05*self.scheduler.prb

        throughput = KPI.throughput(self.scheduler.prb, sinr_eff)
        drop = KPI.drop(sinr_eff)         
        
        return {
            "sinr":sinr,
            "prb": self.scheduler.prb,
            "throughput": throughput,
            "handover": ho,
            "serving": serving,
            "drop": drop
        }