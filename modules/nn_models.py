import torch
from torch import nn

import torch
from torch import nn

class LinearJarolim(nn.Module):
    def __init__(self, in_shape: int, out_shape: int, hidden_units: int, n_layers: int):
        super().__init__()
        self.name = "SimpleLinear"
        self.n_layers = n_layers
        self.input = nn.Linear(in_features=in_shape, out_features=hidden_units)
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.hidden_layers.append(nn.Linear(in_features=hidden_units, out_features=hidden_units))
            self.hidden_layers.append(nn.BatchNorm1d(hidden_units))
            self.hidden_layers.append(nn.Dropout(p=0.5))

        self.output = nn.Linear(in_features=hidden_units, out_features=out_shape)
        
    def forward(self, x):
        x = self.input(x)
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                x = layer(x)
                x = torch.sin(x)
            else:
                x = layer(x)
        x = self.output(x)
        return x