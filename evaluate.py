import numpy as np
import torch
from env.SimEnv import SimRANEnv
from agents.algo import DQN

MODEL_PATH = "dqn_simran.pt"
EPISODES = 20
MAX_STEPS = 300

device = "cpu"
env = SimRANEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

qnet = DQN(state_dim, action_dim).to(device)
qnet.load_state_dict(torch.load(MODEL_PATH))
qnet.eval()

episode_rewards = []

for ep in range(EPISODES):

    state, _ = env.reset()
    ep_reward = 0

    for t in range(MAX_STEPS):
        with torch.no_grad():
            s = torch.tensor(state).float().unsqueeze(0).to(device)
            action = torch.argmax(qnet(s)).item()

        state, reward, done, _, info = env.step(action)
        ep_reward += reward

        if done:
            break

    episode_rewards.append(ep_reward)
    print(f"[EVAL] Episode {ep:3d} | Reward: {ep_reward:8.2f}")

print("Average reward:", np.mean(episode_rewards))