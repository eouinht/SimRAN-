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
    # rsrp của cell đang phục vụ UE
    def rsrp(self, d):
        pathloss = 32.4 + 20*np.log10(d)
        
        return 50 - pathloss

    def sinr(self, rsrp_serv, rsrp_all):
        # rsrp_all là danh sách rsrp từ tất cả  cell tới UE
        interf_list = [p for p in rsrp_all if p != rsrp_serv]
        if len(interf_list) == 0:
            interf = -120
        else:
            interf = np.max(interf_list)
        sinr = rsrp_serv - (interf + self.noise)
        # print(f"rsrp_Serv: {rsrp_serv} -- interf: {interf} ---- noise {self.noise}\n ")
        return sinr 