import torch 
import numpy as np
from .algo import DQN

class DQNAgent:
    def __init__(
        self,
        state_dim, 
        action_dim,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min = 0.05,
        epsilon_decay=0.995,
        device="cpu"
    ):
    
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = device
        
        self.qnet = DQN(state_dim,action_dim).to(device)
        self.target = DQN(state_dim, action_dim).to(device)
        self.target.load_state_dict(self.qnet.state_dict())
        
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=lr)

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
    
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            s = torch.tensor(state).float().unsqueeze(0).to(self.device)
            return torch.argmax(self.qnet(s)).item()
    
    def train_step(self, batch):
        state, action, reward, next_state, done = batch
        
        state = torch.tensor(state).float().to(self.device)
        action = torch.tensor(action).long().to(self.device)
        reward = torch.tensor(reward).float().to(self.device)
        next_state = torch.tensor(next_state).float().to(self.device)
        done = torch.tensor(done).float().to(self.device)
        
        q = self.qnet(state).gather(1, action.unsqueeze(1)).squeeze()
        with torch.no_grad():
            q_next = self.target(next_state).max(1)[0]
        
        y = reward + self.gamma * q_next *(1 - done)
        loss = ((q - y) ** 2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target(self):
        self.target.load_state_dict(self.qnet.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
        
        