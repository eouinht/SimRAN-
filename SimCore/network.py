import numpy as np

class Network:
    def __init__(self, config):
        self.config = config
    
    def compute(self, ues, cells):
        metrics = {}
        metrics["totalTraffic"] = np.sum([u.traffic for u in ues])
        
        metrics["connectedUEs"] = len(
           [u for u in ues if u.serving_cell is not None]
        )
        
        sinrs = [u.sinr for u in ues]
        metrics["avgSINR"] = np.mean(sinrs)
        
        metrics["avgThroughput"] = np.mean(
            [u.throughput for u in ues]
        )
        metrics["maxPrbUsage"] = np.max(
            [c.prb_usage / c.max_prb for c in cells]
        )
        metrics["totalTxPower"] = np.sum(
            [c.tx_power for c in cells]
        )

        return metrics