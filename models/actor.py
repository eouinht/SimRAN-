import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """Gaussian Policy Network
    Input: state vector
    OutpuT: 
        mean: action mean
        log_std
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self._init_weights()
        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)
        
        nn.init.orthogonal_(self.mean_layer.weight, gain = 0.01)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = self.mean_layer(x)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std
    
    def get_dist(self, state):
        logits = self.forward(state)
        return torch.distributions.Categorical(logits=logits)