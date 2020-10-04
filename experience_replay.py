import numpy as np
import random
import torch

class ReplayBuffer:
    """Experience Replay Buffer para DQNs, com priority sampling."""
    def __init__(self, max_length, observation_space, alpha, offset):
        """Cria um Replay Buffer.

        Parâmetros
        ----------
        max_length: int
            Tamanho máximo do Replay Buffer.
        observation_space: int
            Tamanho do espaço de observação.
        alpha: float no intervalo (0,1)
            Quanto a priorização é relevante.
        offset: float < 1
            Garante que todas as experiências podem ser revisitadas.
        """
        self.index, self.size, self.max_length = 0, 0, max_length

        self.alpha, self.offset = alpha, offset

        self.states = np.zeros((max_length, *observation_space), dtype=np.float32)
        self.actions = np.zeros((max_length), dtype=np.int32)
        self.rewards = np.zeros((max_length), dtype=np.float32)
        self.next_states = np.zeros((max_length, *observation_space), dtype=np.float32)
        self.dones = np.zeros((max_length), dtype=np.float32)
        self.priority = np.zeros((max_length), dtype=np.float32)

    def __len__(self):
        """Retorna o tamanho do buffer."""
        return self.size


    def update(self, state, action, reward, next_state, done):
        """Adiciona uma experiência ao Replay Buffer.

        Parâmetros
        ----------
        state: np.array
            Estado da transição.
        action: int
            Ação tomada.
        reward: float
            Recompensa recebida.
        state: np.array
            Estado seguinte.
        done: int
            Flag indicando se o episódio acabou.
        """
        if len(self) > 0:
            mx = np.max(self.priority[:len(self)])
        else:
            mx = 1

        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done
        self.priority[self.index] = mx
        
        self.index = (self.index + 1) % self.max_length
        if self.size < self.max_length:
            self.size = self.index
    
    def update_priority(self, index, new_priority):
        q = (new_priority.type(torch.float64) + self.offset) ** self.alpha
        self.priority[index] = q.detach().cpu().numpy().flatten()
            
    def sample(self, batch_size):
        """Retorna um batch de experiências.
        
        Parâmetros
        ----------
        batch_size: int
            Tamanho do batch de experiências.

        Retorna
        -------
        states: np.array
            Batch de estados.
        actions: np.array
            Batch de ações.
        rewards: np.array
            Batch de recompensas.
        next_states: np.array
            Batch de estados seguintes.
        dones: np.array
            Batch de flags indicando se o episódio acabou.
        """
        # Escolhe índices aleatoriamente do Replay Buffer
        n = len(self)
        idxs = random.choices(range(n), weights=self.priority[:n], k=batch_size)

        return (self.states[idxs],
                self.actions[idxs],
                self.rewards[idxs],
                self.next_states[idxs],
                self.dones[idxs],
                self.priority[idxs]/sum(self.priority),
                idxs
                )

class NStepBuffer:
    """Experience Replay Buffer com n-step para DQNs."""
    def __init__(self, observation_space, gamma, n_step=3):

        self.gamma, self.n_step, self.episode_step = gamma, n_step, 0
        self.index, self.full = 0, False

        self.states = np.zeros((n_step, *observation_space), dtype=np.float32)
        self.actions = np.zeros((n_step), dtype=np.int32)
        self.n_step_rewards = np.zeros((n_step), dtype=np.float32)
        self.next_states = np.zeros((n_step, *observation_space), dtype=np.float32)
        self.dones = np.zeros((n_step), dtype=np.float32)

    def update(self, state, action, reward, next_state, done):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.n_step_rewards[self.index] = 0
        self.dones[self.index] = done

        # Soma o reward atual aos n instantes passados
        for n in range(self.n_step):
            if self.episode_step - n < 0:
                break

            self.n_step_rewards[(self.index - n) % self.n_step] += reward * self.gamma**n

        if done:
            self.episode_step = -1
            self.dones[:] = 1

        if self.episode_step - self.n_step + 1 >= 0:
            self.next_states[(self.index - self.n_step + 1) % self.n_step] = next_state

        self.episode_step += 1
        self.index = (self.index + 1) % self.n_step
        if self.index == 0:
            self.full = True

        if self.full == False:
            return None

        return (self.states[self.index], self.actions[self.index], self.n_step_rewards[self.index], self.next_states[self.index], self.dones[self.index])