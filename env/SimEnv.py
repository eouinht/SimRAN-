import gymnasium as gym
from gymnasium import spaces
from SimCore.core import SimCore 
import numpy as np

class SimRANEnv(gym.Env):
    def __init__(self, n_cells=10, max_steps=300):
        
        super(SimRANEnv, self).__init__()
        self.sim = SimCore(n_cells)
        self.n_cells = n_cells
        self.max_steps = max_steps
        self.step_count = 0
        
        # Action Space (4 actions)
        self.action_space = spaces.Discrete(4)
        # State: [sinr, ue_positon, serving, prb_allocated]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
    def _decode_action(self, action):

        if action == 0:
            return {"ho": 0, "prb_unit": 0}

        if action == 1:
            return {"ho": 0, "prb_unit": 1}

        if action == 2:
            return {"ho": 0, "prb_unit": -1}

        if action == 3:
            return {"ho": 1, "prb_unit": 0}
        
    def step(self, action):

        self.step_count += 1

        act = self._decode_action(action)
        info = self.sim.step(act)
        
        throughput = info["throughput"]
        sinr = info["sinr"]
        ho = info["handover"]
        drop = info["drop"]
        
        throughput_norm = throughput/150.0
        sinr_norm = sinr / 60.0
        # print(f"Truoc chuan hoa: Throughput: {throughput}, sinr: {sinr}, Ho: {ho}, Drop: {drop}\n")
        reward = (
            1.2 * throughput_norm
            + 0.3 * sinr_norm
            - 2.0 * ho
            - 5.0 * drop
        )
        # print(f"Sau chuan hoa: Throughput: {throughput_norm}, sinr: {sinr_norm}, Ho: {ho}, Drop: {drop}\n")
        terminated = False
        truncated = self.step_count >= self.max_steps
        # print(f"Reward={reward}\n")
        return self._get_state(info), reward, terminated, truncated, info     
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim = SimCore(self.n_cells)
        self.step_count = 0

        action = self._decode_action(1)
        info = self.sim.step(action)

        return self._get_state(info), {}
        
    def _get_state(self, info):
        # Return state vector
        
        sinr = np.clip(info["sinr"] / 60.0, 0, 1)

        ue_position = self.sim.ue.position / self.sim.topo.max_x
        serving = info["serving"] / self.n_cells
        prb_allocated = info["prb"] / 100
           
        return np.array([
            sinr, 
            ue_position, 
            serving, 
            prb_allocated
            ], dtype=np.float32)
        
        
                
        