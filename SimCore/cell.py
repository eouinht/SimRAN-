import numpy as np

class Cell:
    def __init__(self, cell_id, pos, tx_power, max_prb):
       
        self.id = cell_id
        self.pos = pos          # (x,y)
        self.tx_power = tx_power
        self.max_prb = max_prb  
        
        self.connected_ues = []
        
        self.load = 0.0 # The hien prb usage
        self.prb_usage = 0.0
        
        self.avg_sinr = 0.0
        self.avg_rsrp = 0.0
        self.avg_rsrq = 0.0
        
        self.total_traffic = 0.0
        
    def reset(self):
        self.connected_ues = []
        self.prb_usage = 0.0
        self.load = 0.0

        self.avg_sinr = 0.0
        self.avg_rsrp = 0.0
        self.avg_rsrq = 0.0

        self.total_traffic = 0.0

    def apply_power(self, delta):
        self.tx_power = np.clip(self.tx_power + delta, 10, 46)
        
    def update_prb_usage(self, ue_demands):
            """
            ue_demands: list nhu cầu PRB của từng UE (0→1)
            """

            total_demand = np.sum(ue_demands)

            self.prb_usage = np.clip(
                total_demand / self.max_prb,
                0.0,
                1.0
            )

            self.load = self.prb_usage