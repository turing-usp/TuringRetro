import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Network, self).__init__()
        self.out_dim = out_dim

        self.convs = nn.Sequential(
                    nn.Conv2d(4, 32, 8, stride=4, padding=0), 
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, stride=1, padding=0), 
                    nn.ReLU())

        self.q1 = nn.Linear(7 * 7 * 64, 512)
        self.q2 = nn.Linear(512, out_dim)

    def forward(self, state):
        conv = self.convs(state)
        flat = conv.reshape(-1, 7 * 7 * 64)
        q1 = F.relu(self.q1(flat))
        q = self.q2(q1)
        return q