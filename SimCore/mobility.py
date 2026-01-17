import numpy as np

def move(ue_pos, ue_vel, area):
    ue_pos += ue_vel
    ue_pos = np.clip(ue_pos, 0, area)
    return ue_pos