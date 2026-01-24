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
        # TOPOLOGY:
        self.cell_position = np.array([0.0, 100.0])
        self.n_cells = len(self.cell_pos)
        self.tx_power = 46.0                        #(dBm)

        # RESOURCES:
        self.PRB_total = 100
        self.PRB_step = 5
        
        # RL SPACES: 
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low = np.array([0, -50, 0, 0]),
            high = np.array([30, 50, 1, 10]),
            dtype = np.float32
        )
        
        
        self.max_steps = 300    
        self.reset()
        
    # PHYSICAL:
    def pathloss(self, d):
        return 32.4 + 20*np.log10(max(d, 1.0))
    
    def rsrp(self, d):
        return self.tx_power - self.pathloss(d)
    
    def drop_prob(self, sinr_eff):
        return 1 / (1 + np.exp(0.8 * sinr_eff))
    
    # RESET
         
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ue_position = np.random.uniform(0, 100)
        self.ue_speed = np.random.uniform(-2, 2)
        self.serving = 0 
        self.PRB_allocated = 10
        self.ue_demand = np.random.uniform(2, 8)
        self.t = 0
        return self._get_state(), {}
        
    def step(self, action):
        self.t += 1
        
        # UE MOBILITY:
        self.ue_position += self.ue_speed
        if self.ue_position<0 or self.ue_position>100:
            self.ue_speed *= -1
        self.ue_position = np.clip(self.ue_position, 0, 100)
        
        # PRB CONTROL:
        if action == 1:
            self.PRB_allocated += self.PRB_step
        if action == 2:
            self.PRB_allocated -= self.PRB_step
        
        self.PRB_allocated = np.clip(self.PRB_allocated, 1, self.PRB_total)
        
        # HANDOVER:
        HO = 0
        if action == 3:
            self.serving = 1 - self.serving
            HO = 1
        
        # CHANNEL:
        d_serv = abs(self.ue_position - self.cell_position[self.serving])
        d_nei = abs(self.ue_position - self.cell_position[1-self.serving])
        
        # rsrp
        rsrp_serv = self.rsrp(d_serv)        
        rsrp_nei = self.rsrp(d_nei)
            
        # Noise
        noise = np.random.normal(-100,2) #dBm
        sinr = rsrp_serv - (rsrp_nei + noise)  
         
        # PRB lam tang SINR hieu dung (coding/ diversity)
        sinr_eff = sinr + 0.1*self.PRB_allocated 
        
        # Throughput
        sinr_linear = 10**(sinr_eff/10)
        throughput = self.PRB_allocated*np.log2(1+max(sinr_linear, 1e-6))

        # Drop
        p_drop = self.drop_prob(sinr_eff)
        drop =  np.random.rand() < p_drop
        
        # Load
        load = self.PRB_allocated/self.PRB_total
        
        # Reward
        reward = (throughput - 5*load - 10*drop -5*HO)
        
        done = self.t >= self.max_steps
        info = {
            
            "sinr": sinr,
            "sinr_eff": sinr_eff,
            "rsrp_serv": rsrp_serv,
            "rsrp_nei": rsrp_nei,
            "prb": self.PRB_allocated,
            "drop_prob": p_drop,
            "HO": HO
        }
        return self._get_state(), reward, done, False, info
    
    
    def _get_state(self):
        # Return state vector
        
        # CHANNEL:
        d_serv = abs(self.ue_position - self.cell_position[self.serving])
        d_nei = abs(self.ue_position - self.cell_position[1-self.serving])
        
        # rsrp
        rsrp_serv = self.rsrp(d_serv)        
        rsrp_nei = self.rsrp(d_nei)
            
        sinr = rsrp_serv - rsrp_nei
           
        return np.array([
            sinr, 
            self.ue_position, 
            self.serving, 
            self.PRB_allocated
            ], dtype=np.float32)
        
        
                
        