import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super(ActorCritic, self).__init__()

        self.convs = nn.Sequential(
                    nn.Conv2d(4, 32, 8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, stride=1, padding=0),
                    nn.ReLU())

        self.fc = nn.Linear(7 * 7 * 64, 512)
        nn.init.orthogonal_(self.fc.weight, np.sqrt(2))

        self.policy = nn.Linear(512, action_shape)
        nn.init.orthogonal_(self.policy.weight, np.sqrt(0.01))

        self.value = nn.Linear(512, 1)
        nn.init.orthogonal_(self.value.weight, 1)

    def forward(self, state):
        conv = self.convs(state)
        flat = conv.reshape(-1, 7 * 7 * 64)

        h = F.relu(self.fc(flat))

        v = self.value(h)
        probs = F.softmax(self.policy(h),dim=-1)
        probs = Categorical(probs)

        return probs, v