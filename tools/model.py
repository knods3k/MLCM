#%%
from tools.data import MaterialDataHandler

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
        self.activation = self.hyperparams.activation

        self.build()

    def build(self):
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        self.layers.append(self.activation)
        for _ in range(self.n_layers):
            linear = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.layers.append(linear)
            self.layers.append(self.activation)
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        return self

    def forward(self, input_data):
        """
        Forward pass of the network.
        :param input_data:
        :return: input_data passed through the network.
        """
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data


class CANN(Model):
    def __init__(self, *args, input_dim=3, polynomial_degree=3, alpha=0, **kwargs):
        super(CANN, self).__init__(*args, **kwargs)
        self.hyperparams.input_dim = input_dim
        self.exponents = torch.arange(int(polynomial_degree))

        self.activation_functions = self.BuildActivationFunctions()

        self.layers = []

        self.layers.append(self.FirstLayer)
        self.layers.append(self.SecondLayer)
        self.layers.append(nn.Linear(int(
            self.hyperparams.input_dim * polynomial_degree * len(self.activation_functions)),
            1))

        self.alpha = alpha
    
    @staticmethod
    def linear_activation(x):
        return torch.pow(x, 1)
    
    @staticmethod
    def quadratic_activation(x):
        return torch.pow(x, 2)
    
    def linear_exponential(self, x):
        return torch.exp(self.alpha * x) - 1
    
    def quadratic_exponential(self, x):
        return torch.exp(self.alpha*(x**2)) - 1
    
    def FirstLayer(self, input_data):
        return torch.pow(input_data.unsqueeze(-1), self.exponents)
    
    def BuildActivationFunctions(self):
        functions = []
        functions.append(self.linear_activation)
        functions.append(self.quadratic_activation)
        functions.append(self.linear_exponential)
        functions.append(self.quadratic_exponential)
        return functions
    
    def SecondLayer(self, input_data):
        activations = [a(input_data).flatten(-2) for a in self.activation_functions]
        return torch.cat(activations, axis=-1)
    

    def forward(self, input_data):
        """
        Forward pass of the network.
        :param input_data:
        :return: input_data passed through the network.
        """
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data

        
        
if __name__ == "__main__":
    mdata = MaterialDataHandler()
    x, y = mdata.get_training_data()
    model = CANN()
    print(model(x))


        
        
# %%
