import numpy as np

# def pathloss(ue_pos, cell_pos, alpha):
#     d = np.linalg.norm(ue_pos - cell_pos) + 1
#     return 1.0/(d**alpha)

# def sinr(ue, serving, ue_pos, cell_pos, txPower, noise, alpha):
#     sig = txPower[serving] * pathloss(ue_pos[ue], cell_pos[serving], alpha)
#     interf = 0
#     for j in range(len(txPower)):
#         if j != serving:
#             interf += txPower[j] * pathloss(ue_pos[ue], cell_pos[j], alpha)
#     return sig/(interf + noise)
class Channel:
    def __init__(self):
        self.noise = np.random.normal(5, 1)
    
    def rsrp(self, d):
        return 50 - 0.5*d
    def sinr(self, rsrp_serv, rsrp_all):
        interf = np.max(rsrp_all)
        return rsrp_serv - (interf + self.noise)