import numpy as np
class KPI:
    @staticmethod
    def throughput (prb, sinr):
        return prb*np.log2(1 + max(0.1, sinr))
    
    @staticmethod
    def drop(sinr_eff):
        p = 1/(1 + np.exp(0.8 * sinr_eff))
        return np.random.rand() < p 