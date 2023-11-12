import torch
import torch.nn as nn
from tools.hyperparameters import Hyperparameters
from tools.training import start_training, start_evaluation
from tools.settings import DEVICE


HYPERPARAMS = Hyperparameters()

class Model(nn.Module):
    """
    Model is the base class for all models. It contains the start_training, start_evaluation and save methods.
    :param hyperparams: The hyperparameters to use for training the network. Default values are set in tools/hyperparameters.py.
    """
    def __init__(self, *args, hyperparams=HYPERPARAMS, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hyperparams = hyperparams

    def start_training(self, *args, **kwargs):
        return start_training(self, *args, **kwargs)
    
    def start_evaluation(self, *args, **kwargs):
        return start_evaluation(self, *args, **kwargs)

    def sum_weights(self):
        return sum(torch.linalg.norm(p) for p in self.parameters())
    
    def save(self, model_file):
        torch.save(self, model_file)

class MLP(Model):
    """
    MLP is a class that represents a multilayer perceptron. It inherits from Model base class.
    """
    def __init__(self, *args, **kwargs) -> None:
        super(MLP, self).__init__(*args, **kwargs)
        self.hidden_dim = self.hyperparams.hidden_dim
        self.input_dim = self.hyperparams.input_dim
        self.output_dim = self.hyperparams.output_dim
        self.n_layers = self.hyperparams.n_layers

        self.layers = nn.ModuleList()
        self.activation = self.hyperparams.activation

        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        self.layers.append(self.activation)
        for _ in range(self.n_layers):
            linear = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.layers.append(linear)
            self.layers.append(self.activation)
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, input_data):
        """
        Forward pass of the network.
        :param input_data:
        :return: input_data passed through the network.
        """
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data
