import random
from collections import deque
import numpy as np

class RelayBuffer:
    def __init__(self, size=100000):
        self.buffer = deque(maxlen=size)
        
    def push(self, s, a, r, s2, d):
        self.buffer.append((s, int(a), r, s2, d))
    
    def sample(self, batch):
        samples = random.sample(self.buffer, batch)
        S,A,R,S2,D = zip(*samples)
        return np.array(S), np.array(A), np.array(R), np.array(S2), np.array(D)
    
    def __len__(self):
        return len(self.buffer)