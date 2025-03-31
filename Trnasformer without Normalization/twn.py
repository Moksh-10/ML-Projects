import torch
from torch import nn

class dyt(nn.Module):
    def __init__(self, dim: int, alpha):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha)
        self.beta = nn.Parameter(torch.ones(dim))
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.gamma * nn.functional.tanh(self.alpha * x) + self.beta
