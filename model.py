import torch
from torch import nn


# def my persional Network
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.Layer1 = nn.Linear(784, 256)
        self.Layer2 = nn.Linear(256, 128)
        self.Layer3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.Layer1(x)
        x = torch.relu(x)
        x = self.Layer2(x)
        x = torch.relu(x)
        return self.Layer3(x)
