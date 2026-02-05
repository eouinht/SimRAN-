import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env.SimEnv import RANEnv
from models.actor import Actor
from models.critic import Critic

# =====================
# HYPERPARAMETERS
# =====================

GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.2
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
EPOCHS = 10
BUFFER_SIZE = 2048
MAX_EPISODES = 2000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.values = []
        self.dones = []

    def clear(self):
        self.__init__()

class PPOAgent:
    def __init__(self, state_dim, action_dim):

        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic = Critic(state_dim).to(DEVICE)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.buffer = Buffer()

    def select_action(self, state):

        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            dist = self.actor.get_dist(state)
            action = dist.sample()
            logprob = dist.log_prob(action).sum(dim=-1)
            value = self.critic(state)

        return (
            action.cpu().numpy().flatten(),
            logprob.item(),
            value.item()
        )

    def compute_gae(self, rewards, values, dones):

        advantages = []
        gae = 0

        values = values + [0]

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + GAMMA * values[i+1] * (1-dones[i]) - values[i]
            gae = delta + GAMMA * LAMBDA * (1-dones[i]) * gae
            advantages.insert(0, gae)

        return advantages


    def update(self):

        states = torch.FloatTensor(self.buffer.states).to(DEVICE)
        actions = torch.FloatTensor(self.buffer.actions).to(DEVICE)
        old_logprobs = torch.FloatTensor(self.buffer.logprobs).to(DEVICE)

        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = self.buffer.values

        advantages = self.compute_gae(rewards, values, dones)
        returns = torch.FloatTensor(advantages).to(DEVICE) + \
                  torch.FloatTensor(values).to(DEVICE)

        advantages = torch.FloatTensor(advantages).to(DEVICE)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(EPOCHS):

            dist = self.actor.get_dist(states)
            new_logprobs = dist.log_prob(actions).sum(dim=-1)

            ratio = torch.exp(new_logprobs - old_logprobs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-CLIP, 1+CLIP) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(self.critic(states).squeeze(), returns)

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

        self.buffer.clear()


def train():

    env = RANEnv()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]   # Box

    agent = PPOAgent(state_dim, action_dim)

    timestep = 0

    for ep in range(MAX_EPISODES):

        state, _ = env.reset()
        ep_reward = 0

        while True:

            action, logprob, value = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.buffer.states.append(state)
            agent.buffer.actions.append(action)
            agent.buffer.logprobs.append(logprob)
            agent.buffer.values.append(value)
            agent.buffer.rewards.append(reward)
            agent.buffer.dones.append(done)

            state = next_state
            ep_reward += reward
            timestep += 1

            if timestep % BUFFER_SIZE == 0:
                agent.update()

            if done:
                break

        print(f"Episode {ep} | Reward: {ep_reward:.2f}")

    env.close()

if __name__ == "__main__":
    train()
