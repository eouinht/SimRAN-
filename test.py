from env.SimEnv import RANEnv
from config import SimConfig

env = RANEnv(SimConfig)
obs, _ = env.reset()

print(obs.shape)
print(env.action_space.sample())
