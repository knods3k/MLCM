import torch.nn as nn
from tools.hyperparameters import Hyperparameters

HYPERPARAMS = Hyperparameters()


class MLP(nn.Module):
    def __init__(self, hyperparams=HYPERPARAMS) -> None:
        super(MLP, self).__init__()
        self.hidden_dim = hyperparams.hidden_dim
        self.input_dim = hyperparams.input_dim
        self.output_dim = hyperparams.output_dim
        self.n_layers = hyperparams.n_layers

        self.layers = nn.ModuleList()
        self.activation = nn.ReLU()

        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        self.layers.append(self.activation)
        for _ in range(self.n_layers):
            linear = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.layers.append(linear)
            self.layers.append(self.activation)
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data