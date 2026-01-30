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
        self.n_cells = n_cells
        self.serving = 0
        
    def step(self, action):
        
        # UE MOBILITY:
        self.ue.move()
        
        # CHANNEL:
        d = np.abs(self.topo.position - self.ue.position)
        # print (f"d = {d} \n")
        rsrp = self.channel.rsrp(d)
        
        # HANDOVER:
        self.serving, ho = self.assoc.step(action, rsrp)
        
        rsrp_serv = rsrp[self.serving]
        interf = np.delete(rsrp, self.serving)
        
        sinr = self.channel.sinr(rsrp_serv, interf)
        
        # SCHEDULER:
        self.scheduler.step(action)
        sinr_eff = sinr + 0.05*self.scheduler.prb

        # KPI:
        throughput = KPI.throughput(self.scheduler.prb, sinr_eff)
        drop = KPI.drop(sinr_eff)         
        
        info = {
            "sinr": sinr,
            "sinr_eff": sinr_eff,
            "prb": self.scheduler.prb,
            "throughput": throughput,
            "handover": ho,
            "serving": self.serving,
            "drop": drop
        }

        self.last_info = info
        return info
