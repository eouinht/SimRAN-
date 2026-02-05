import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """State Function V(s)

    Args:
        nn (_type_): _description_
    """
    
    def __init__(self, state_dim, hidden_dim = 256):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, 1)
        
        self._init_weight()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)
        
        nn.init.orthogonal_(self.value_layer.weight, gain = 1.0) 
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        X = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.value_layer(x)
        return value
        