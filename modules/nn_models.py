import torch
from torch import nn

class LinearJarolim(nn.Module):
    def __init__(self, in_shape: int, out_shape: int, hidden_units: int, n_layers: int):
        super().__init__()
        self.name = "SimpleLinear"
        self.n_layers = n_layers
        self.input = nn.Linear(in_features = in_shape, out_features = hidden_units)
        
        self.hidden_linear = nn.Linear(in_features = hidden_units, out_features = hidden_units)

        self.output = nn.Linear(in_features = hidden_units, out_features = out_shape)
        
    def forward(self, x):
        x = self.input(x)
        for i in range(self.n_layers):
            x = self.hidden_linear(x)
            x = torch.sin(x)
        x = self.output(x)
        return x