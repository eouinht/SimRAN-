import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SimRANEnv(gym.Env):
    # UE = RRC measurementr
    # RSRP = Pathloss
    # SINR = Interference + Noise
    # PRB = Scheduler
    # Drop = BLER
    # HO cost
    # Reward 
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low = np.array([0, -50, 0, 0]),
            high = np.array([30, 50, 1, 10]),
            dtype = np.float32
        )
        
        self.PRB_t = 100
        self.max_steps = 300    
        self.reset()
        
        
    def reset(self):
        self.ue_pos = np.random.uniform(0, 100)
        self.ue_speed = np.random.uniform(-2, 2)
        self.serving = 0 
        self.PRB_a = 10
        self.ue_demand = np.random.uniform(2, 8)
        self.t = 0
        return self.__getstate__(), {}
    
    def _rsrp(self, d):
        # Tạm thời set logic đơn giản (có thể không thực tế lắm)
        # Trong quá trình áp dụng vào bài toán thật thì sửa công thức đúng vào đây :>
        return 50 - 0.5*d 
    
    def _drop_prob(sefl, sinr_eff):
        return 1/(1 + np.exp(0.8 * sinr_eff))
    
    def step(self, action):
        self.t += 1
        
        # UE mobility
        self.ue_pos += self.ue_speed
        if self.ue_pos<0 or self.ue_pos>100:
            self.ue_speed *= -1
        self.ue_pos = np.clip(self.ue_pos, 0, 100)
        
        # distgance to gNB (0 and 100)
        d0 = abs(self.ue_pos - 0)
        d1 = abs(self.ue_pos - 100)
        
        # rsrp
        rsrp0 = self._rsrp(d0)        
        rsrp1 = self._rsrp(d1)
        
        # Handover
        HO = 0
        if action == 3:
            self.serving = 1 - self.serving
            HO = 1
    
        rsrp_serv = rsrp0 if self.serving == 0 else rsrp1
        rsrp_nei = rsrp1 if self.serving == 0 else rsrp0
        
        # Noise
        noise = np.random.normal(5,1)
        
        # SINR
        sinr = rsrp_serv - (rsrp_nei + noise)  
        sinr_eff = sinr + 0.1*self.PRB_a
        
        # PRB control
        if action == 1:
            self.PRB_a += 5
        if action == 2:
            self.PRB_a -= 5
        
        self.PRB_a = np.clip(self.PRB_a, 1, self.PRB_t)
        
        # Throughput
        sinr_l = 10**(sinr_eff/10)
        throughput = self.PRB_a*np.log2(1+max(0.1, sinr_l))

        # Drop
        p_drop = self._drop_prob(sinr_eff)
        drop = 1 if np.random.rand() < p_drop else 0
        
        # Load
        load = self.PRB_a/self.PRB_t
        
        # Reward
        reward = (throughput - 5*load - 10*drop -5*HO)
        done = self.t >= self.max_steps
        info = {
            "ue_pos": self.ue_pos,
            "serving": self.serving,
            "rsrp_serv": rsrp_serv,
            "rsrp_nei": rsrp_nei,
            "sinr": sinr,
            "prb": self.PRB_a,
            "drop_prob": p_drop,
            "reward": reward
        }
        return self.__getstate__(), reward, done, False, info
    
    
    def __getstate__(self):
        # Return state vector
        
        d0 = abs(self.ue_pos-0)
        d1 = abs(self.ue_pos-100)
        
        rsrp0 = self._rsrp(d0)
        rsrp1 = self._rsrp(d1)
        
        rsrp_serv = rsrp0 if self.serving == 0 else rsrp1
        rsrp_nei = rsrp1 if self.serving == 0 else rsrp0
        
        sinr = rsrp_serv - rsrp_nei
        return np.array([sinr, self.ue_pos, self.serving, self.PRB_a], dtype=np.float32)
        
        
                
        