

import numpy as np

class ExperienceReplay:
    """Experience Replay Buffer para PPO."""
    def __init__(self, max_length, observation_space):
        """Cria um Replay Buffer.

        Parâmetros
        ----------
        max_length: int
            Tamanho máximo do Replay Buffer.
        observation_space: int
            Tamanho do espaço de observação.
        """
        self.length = 0
        self.max_length = max_length

        self.states = np.zeros((max_length, *observation_space), dtype=np.float32)
        self.actions = np.zeros((max_length), dtype=np.int32)
        self.rewards = np.zeros((max_length), dtype=np.float32)
        self.next_states = np.zeros((max_length, *observation_space), dtype=np.float32)
        self.dones = np.zeros((max_length), dtype=np.float32)
        self.logp = np.zeros((max_length), dtype=np.float32)

    def update(self, states, actions, rewards, next_states, dones, logp):
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
        logp: float
            Log da probabildiade de acordo com a política.
        """
        self.states[self.length] = states
        self.actions[self.length] = actions
        self.rewards[self.length] = rewards
        self.next_states[self.length] = next_states
        self.dones[self.length] = dones
        self.logp[self.length] = logp
        self.length += 1

    def sample(self):
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
        logp: np.array
            Batch do log da probabildiade de acordo com a política.
        """
        self.length = 0

        return (self.states, self.actions, self.rewards, self.next_states, self.dones, self.logp)

