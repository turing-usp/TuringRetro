import random
import numpy as np

class ReplayBuffer:
    """Experience Replay Buffer para DQNs."""
    def __init__(self, max_length, observation_space):
        """Cria um Replay Buffer.

        Parâmetros
        ----------
        max_length: int
            Tamanho máximo do Replay Buffer.
        observation_space: int
            Tamanho do espaço de observação.
        """
        self.index, self.size, self.max_length = 0, 0, max_length

        self.states = np.zeros((max_length, *observation_space), dtype=np.float32)
        self.actions = np.zeros((max_length), dtype=np.int32)
        self.rewards = np.zeros((max_length), dtype=np.float32)
        self.next_states = np.zeros((max_length, *observation_space), dtype=np.float32)
        self.dones = np.zeros((max_length), dtype=np.float32)

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
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done
        
        self.index = (self.index + 1) % self.max_length
        if self.size < self.max_length:
            self.size = self.index
            
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
        idxs = np.random.randint(0, self.size, size=batch_size)

        return (self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.dones[idxs])

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Stores the priorities and sums of priorities
        self.indices = np.zeros(capacity)  # Stores the indices of the experiences
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, index):
        idx = self.write + self.capacity - 1

        self.indices[self.write] = index
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        assert s <= self.total()
        idx = self._retrieve(0, s)
        indexIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.indices[indexIdx])

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, max_length, observation_space, alpha):
        super(PrioritizedReplayBuffer, self).__init__(max_length, observation_space)
        assert alpha >= 0
        self.alpha = alpha

        self.tree = SumTree(max_length)
        self._max_priority = 1.0

    def update(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self.index
        super().update(*args, **kwargs)
        self.tree.add(self._max_priority ** self.alpha, self.index)

    def sample(self, batch_size, beta):
        assert beta > 0

        priorities = np.zeros((batch_size), dtype=np.float32)
        batch_idxs = np.zeros(batch_size)
        tree_idxs = np.zeros(batch_size, dtype=np.int)

        for i in range(batch_size):
            s = random.uniform(0, self.tree.total())
            (tree_idx, p, idx) = self.tree.get(s)

            priorities[i] = p
            batch_idxs[i] = idx
            tree_idxs[i] = tree_idx

        batch_idxs = np.asarray(batch_idxs).astype(int)

        sampling_probabilities = priorities / self.tree.total()
        weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        weights /= weights.max()

        return (self.states[batch_idxs], self.actions[batch_idxs], self.rewards[batch_idxs], self.next_states[batch_idxs], self.dones[batch_idxs], weights, tree_idxs)

    def update_priorities(self, tree_idxs, errors):
        priorities = np.power(errors, self.alpha).squeeze()
        assert len(priorities) == tree_idxs.size
        for p, i in zip(priorities, tree_idxs):
            self.tree.update(i, p)