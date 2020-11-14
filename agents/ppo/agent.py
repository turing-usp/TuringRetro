import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from .experience_replay import ExperienceReplay
from .network import ActorCritic

class PPOAgent:
    def __init__(self, observation_space, action_space, lr=7e-4, gamma=0.99, lam=0.95, vf_coef=0.5, entropy_coef=0.005,clip_param =0.2, epochs =10, n_steps=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.lam = lam
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.clip_param = clip_param
        self.epochs = epochs

        self.n_steps = n_steps
        self.memory = ExperienceReplay(n_steps, observation_space.shape)

        self.actorcritic = ActorCritic(observation_space.shape, action_space.n).to(self.device)
        self.actorcritic_optimizer = optim.Adam(self.actorcritic.parameters(), lr=lr)

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        probs, v = self.actorcritic.forward(state)
        action = probs.sample()
        log_prob = probs.log_prob(action)
        return action.cpu().detach().item(), log_prob.detach().cpu().numpy()

    def remember(self, state, action, reward, next_state, done, logp):
        self.memory.update(state, action, reward, next_state, done, logp)

    def compute_gae(self, rewards, dones, v, v2):
        T = len(rewards)

        returns = torch.zeros_like(rewards)
        gaes = torch.zeros_like(rewards)
        
        future_gae = torch.tensor(0.0, dtype=rewards.dtype)
        next_return = torch.tensor(v2[-1], dtype=rewards.dtype)

        not_dones = 1 - dones
        deltas = rewards + not_dones * self.gamma * v2 - v

        for t in reversed(range(T)):
            returns[t] = next_return = rewards[t] + self.gamma * not_dones[t] * next_return
            gaes[t] = future_gae = deltas[t] + self.gamma * self.lam * not_dones[t] * future_gae

        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8) # Normalização

        return gaes, returns

    def train(self):
        if self.memory.length < self.n_steps:
            return

        (states, actions, rewards, next_states, dones, old_logp) = self.memory.sample()

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)
        old_logp = torch.FloatTensor(old_logp).to(self.device)
        
        with torch.no_grad():
            _, v = self.actorcritic.forward(states)
            _, v2 = self.actorcritic.forward(next_states)
        
        advantages, returns = self.compute_gae(rewards, dones, v, v2)
        
        for epoch in range(self.epochs):
            
            probs, v = self.actorcritic.forward(states)

            new_logp = probs.log_prob(actions)

            #Equações principais do algoritmo
            ratio = (new_logp.unsqueeze(-1) - old_logp.unsqueeze(-1)).exp() 
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages.detach()

            entropy = probs.entropy().mean()

            policy_loss =   - torch.min(surr1,surr2).mean()
            value_loss =    self.vf_coef * F.mse_loss(v, returns.detach())
            entropy_loss = -self.entropy_coef * entropy

            self.actorcritic_optimizer.zero_grad()
            (policy_loss + entropy_loss + value_loss).backward()
            self.actorcritic_optimizer.step()

        return policy_loss + entropy_loss + value_loss

    def save_model(self, path):
        torch.save(self.actorcritic.state_dict(), path)
    
    def load_model(self, path):
        self.actorcritic.load_state_dict(torch.load(path))