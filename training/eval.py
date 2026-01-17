import torch
import numpy as np
from env.SimEnv import SimRANEnv
from agents.algo import DQN

# ================================
# Load trained model
# ================================
env = SimRANEnv()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = DQN(state_dim, action_dim)
model.load_state_dict(torch.load("results/models/dqn_ran.pt"))
model.eval()

# ================================
# Evaluation
# ================================
EPISODES = 50
MAX_STEPS = 300

total_reward = 0
total_drop = 0
total_ho = 0
total_throughput = 0
total_prb = 0
total_steps = 0

for ep in range(EPISODES):
    s = env.reset()

    for t in range(MAX_STEPS):
        with torch.no_grad():
            a = torch.argmax(model(torch.tensor(s).float())).item()

        s2, r, done, info = env.step(a)

        total_reward += r
        total_steps += 1

        # Metrics from env (read from internal state)
        total_prb += env.PRB_a

        d0 = abs(env.ue_pos - 0)
        d1 = abs(env.ue_pos - 100)
        rsrp0 = env._rsrp(d0)
        rsrp1 = env._rsrp(d1)
        rsrp_serv = rsrp0 if env.serving == 0 else rsrp1
        rsrp_nei = rsrp1 if env.serving == 0 else rsrp0

        sinr = rsrp_serv - rsrp_nei
        sinr_lin = 10 ** (sinr / 10)
        throughput = env.PRB_a * np.log2(1 + sinr_lin)
        total_throughput += throughput

        if done:
            break

# ================================
# Print results
# ================================
print("=== 5G RAN DQN Evaluation ===")
print(f"Average reward: {total_reward / EPISODES:.2f}")
print(f"Avg throughput: {total_throughput / total_steps:.2f}")
print(f"Avg PRB usage: {total_prb / total_steps:.2f}")
