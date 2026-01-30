
from env.SimEnv import SimRANEnv
from agents.agent import DQNAgent
from agents.relay_buffer import ReplayBuffer
import torch
import numpy as np


EPISODES = 500  
BATCH_SIZE = 64
BUFFER_SIZE = 100000
TARGET_UPDATE = 100
MAX_STEPS = 500

device = "cpu"

env = SimRANEnv()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    device=device
)

buffer = ReplayBuffer(BUFFER_SIZE)
reward_log = []
    
for ep in range(EPISODES):
    state, _ = env.reset()
    ep_reward = 0
    for step in range(MAX_STEPS):
        action = agent.select_action(state)
        next_state, reward, done, _ , info = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        ep_reward += reward
        
        if len(buffer) > BATCH_SIZE:
            batch = buffer.sample(BATCH_SIZE)
            agent.train_step(batch)
        
        if done:
            break
    agent.decay_epsilon()
    
    if ep % TARGET_UPDATE == 0:
        agent.update_target()
    reward_log.append(ep_reward)
    if ep % 10 == 0:
        print(f"Episode {ep:4d} | Reward: {ep_reward:8.2f} | Epsilon: {agent.epsilon:.3f}")

torch.save(agent.qnet.state_dict(), "dqn_simran.pt")
np.save("train_rewards.npy", reward_log)
print("Training finished!")
