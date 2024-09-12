import torch
from torch import nn
from torch.nn import functional as F

class PINN(nn.Module):
    def __init__(self, n_layers, n_neurons):
        super(PINN, self).__init__()

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(2, n_neurons))

        for _ in range(n_layers - 2):
            self.layers.append(nn.Linear(n_neurons, n_neurons))

        self.layers.append(nn.Linear(n_neurons, 1))

    def forward(self, x, y):
        inputs = torch.cat([x, y], dim=-1)

        for layer in self.layers[:-1]:
            inputs = F.tanh(layer(inputs))

        u = self.layers[-1](inputs)

        return u
