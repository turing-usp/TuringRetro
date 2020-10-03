import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from experience_replay import NStepBuffer
from per import PrioritizedReplayBuffer
from dqn import Network

class DQNAgent:
    """
    Uma classe que cria um agente DQN que utiliza NStepBuffer como memória
    """
    def __init__(self, 
                 observation_space, 
                 action_space,
                 alpha = 0.6,
                 beta = 0.4,
                 beta_decay = 2e-5, 
                 lr=7e-4, 
                 gamma=0.99,
                 max_memory=100000,
                 epsilon_init=0.5,
                 epsilon_decay=0.9995,
                 min_epsilon=0.01,
                 n_step=3):
        """
        Inicializa o agente com os parâmetros dados
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.alpha = alpha
        self.loss_param = beta
        self.loss_param_decay = beta_decay
        self.gamma = gamma
        self.n_step = n_step
        self.memory = PrioritizedReplayBuffer(max_memory, observation_space.shape, alpha)
        self.n_step_buffer = NStepBuffer(observation_space.shape, gamma, n_step)
        self.action_space = action_space

        self.epsilon = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.dqn = Network(observation_space.shape, action_space.n).to(self.device, non_blocking=True)

        self.optimizer  = optim.Adam(self.dqn.parameters(), lr=lr)

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_epsilon)

        if np.random.random() < self.epsilon:
            action = self.action_space.sample()
            return action

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device, non_blocking=True)
            action = self.dqn.forward(state).argmax(dim=-1)
            action = action.cpu().numpy()[0]

        return action

    def remember(self, state, action, reward, new_state, done):
        experience = self.n_step_buffer.update(state, action, reward, new_state, done)
        if(experience != None):
            self.memory.update(experience[0],experience[1],experience[2],experience[3],experience[4])
        

    def train(self, batch_size=32, epochs=1):
        # Se temos menos experiências que o batch size
        # não começamos o treinamento
        if batch_size > self.memory.size:
            return -float("inf")
        
        for epoch in range(epochs):
            # Pegamos uma amostra das nossas experiências para treinamento
            (states, actions, rewards, next_states, dones, priorities, indexes) = self.memory.sample(batch_size, self.loss_param)
            self.loss_param *= 1+self.loss_param_decay if self.loss_param < 1 else 1

            # Transformar nossas experiências em tensores
            states = torch.as_tensor(states).to(self.device, non_blocking=True)
            actions = torch.as_tensor(actions).to(self.device, non_blocking=True).unsqueeze(-1)
            rewards = torch.as_tensor(rewards).to(self.device, non_blocking=True).unsqueeze(-1)
            next_states = torch.as_tensor(next_states).to(self.device, non_blocking=True)
            dones = torch.as_tensor(dones).to(self.device, non_blocking=True).unsqueeze(-1)
            weights = torch.as_tensor(priorities).to(self.device, non_blocking=True).unsqueeze(-1)

            q = self.dqn.forward(states).gather(-1, actions.long())

            with torch.no_grad():
                q2 = self.dqn.forward(next_states).max(dim=-1, keepdim=True)[0]

                target = (rewards + (1 - dones) * (self.gamma ** self.n_step) * q2)

            loss = F.mse_loss(q, target, reduction="none")
            
            priorities = loss.detach().cpu().numpy() + 1e-6
            self.memory.update_priorities(indexes, priorities)

            weighted_loss = loss * weights
            final_loss = torch.mean(weighted_loss)

            self.optimizer.zero_grad()
            final_loss.backward()
            for param in self.dqn.parameters():
                param.grad.data.clamp_(-100,100)
            self.optimizer.step()

            return final_loss

    def save_model(self, path):
        torch.save(self.dqn.state_dict(), path)
    
    def load_model(self, path):
        self.dqn.load_state_dict(torch.load(path))