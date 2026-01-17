import gymnasium as gym
import matplotlib.pyplot as plt
from env.SimEnv import SimRANEnv
from agents.algo import DQN
import torch.optim as optim
from agents.relay_buffer import RelayBuffer
import torch
import numpy as np

env = SimRANEnv()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

qnet = DQN(state_dim,action_dim)
target = DQN(state_dim, action_dim)
target.load_state_dict(qnet.state_dict())

optimizer = optim.Adam(qnet.parameters(), lr=1e-3)
buffer = RelayBuffer()
gamma = 0.99
batch_size = 64
epsilon = 1.0
reward = []

for ep in range(400):
    s = env.reset()
    ep_reward = 0
    
    for t in range(300):
        if np.random.rand() < epsilon:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = torch.argmax(qnet(torch.tensor(s).float())).item()
        
        s2, r, done, _ = env.step(a)
        buffer.push(s, a, r, s2, done)
        s = s2
        ep_reward += r
        if len(buffer) > batch_size:
            S, A, R, S2, D = buffer.sample(batch_size)
            
            S = torch.tensor(S).float()
            A = torch.tensor(A).long()
            print(f"Hereeeeeeee{A.dtype}")
            R = torch.tensor(R).float()
            S2 = torch.tensor(S2).float()
            D = torch.tensor(D).float()
            q = qnet(S).gather(1, A.unsqueeze(1)).squeeze()
            with torch.no_grad():
                q2 = target(S2).max(1)[0]
                
            y = R + gamma*q2*(1-D)
            loss = ((q-y)**2).mean()
            optimizer.zero_grad()
            optimizer.step()
            
        if done:
            break
    
    epsilon = max(0.05, epsilon*0.995)
    reward.append(ep_reward)
    
    if ep%10 == 0:
        target.load_state_dict(qnet.state_dict())
        print(f"Episode {ep}, reward {ep_reward:.1f}")
    
    
