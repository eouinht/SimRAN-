import gymnasium as gym
from gymnasium import spaces
from SimCore.core import SimCore 
import numpy as np
from config import SimConfig

class RANEnv(gym.Env):
    def __init__(self, config = SimConfig):
        super().__init__()
        self.config = config
        self.sim = SimCore(config)
        
        self.action_space = spaces.MultiDiscrete(
            [3] * self.config.N_CELLS
        )
        
        ue_dim = 5
        cell_dim = 4
        
        obs_dim = (
            self.cfg.N_UES * ue_dim +
            self.cfg.N_CELLS * cell_dim
        )
        
        self.observation_space = spaces.Box(
            low=-1e9,
            high=1e9,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.steps = 0

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)

        self.sim = SimCore(self.cfg)
        self.steps = 0

        state, _, _, _ = self.sim.step(
            np.zeros(self.cfg.N_CELLS)
        )

        return state, {}

    def step(self, action):

        # Decode {0,1,2} → {-1,0,1}
        real_action = action - 1

        state, reward, done, info = self.sim.step(real_action)

        self.steps += 1
        if self.steps >= self.cfg.MAX_STEPS:
            done = True

        return state, reward, done, False, info