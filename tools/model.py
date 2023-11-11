import torch
import torch.nn as nn
from tools.hyperparameters import Hyperparameters
from matplotlib import pyplot as plt
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

    def start_training(self, data_handler, model_file=None, verbosity=2):
        data_handler.batch_size = self.hyperparams.batch_size
        train_loader = data_handler.get()
        self.to(DEVICE)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hyperparams.learning_rate)
        avg_losses = torch.zeros(self.hyperparams.epochs)

        for epoch in range(self.hyperparams.epochs):
            self.train()
            avg_loss = 0.

            for x, y in train_loader:
                self.zero_grad()

                out = self(x.to(DEVICE))
                loss = self.hyperparams.criterion(out, y.to(DEVICE))

                loss.backward()
                optimizer.step()

                avg_loss += loss.item()

                if epoch % 20 == 0 and verbosity >= 1:
                    # print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {} = {} + {}".format(epoch, epoch,
                    #                                                                                 len(train_loader),
                    #                                                                                 avg_loss / (epoch+1), loss.item(), 0),\
                    #                                                                                     end='\r', flush=True)
                    print("Epoch {}/{} Done, Total Loss: {}".format(epoch, self.hyperparams.epochs, avg_loss / len(train_loader)),\
                                                                                                        end='\r', flush=True)
            avg_losses[epoch] = avg_loss / len(train_loader) 

        if verbosity >= 2:
            plt.figure(figsize=(12, 8))
            plt.plot(avg_losses, "x-")
            plt.title("Train loss (MSE, reduction=mean, averaged over epoch)")
            plt.xlabel("Epoch")
            plt.ylabel("loss")
            plt.grid(visible=True, which='both', axis='both')
            plt.show()

        if model_file is not None:
            torch.save(self, model_file)
        return self
    
    def start_evaluation(self, data_handler):
        test_x, test_y = data_handler.get_test_data()
        with torch.no_grad():
            self.eval() 
            outputs = [] 
            targets = []
            testlosses = []

            out = self(test_x.to(DEVICE))

            outputs.append(out.cpu().detach().numpy())
            targets.append(test_y.cpu().detach().numpy())
            testlosses.append(self.hyperparams.criterion(out, test_y.to(DEVICE)).item())
        return outputs, targets, testlosses

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
