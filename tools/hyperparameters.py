import torch.nn as nn

HIDDEN_DIM = 640
INPUT_DIM = 2
OUTPUT_DIM = 1
N_LAYERS = 3
ACTIVATION = nn.ReLU()

EPOCHS = 1500
LEARNING_RATE = .001
CRITERION = nn.MSELoss(reduction='mean')

class Hyperparameters():
    def __init__(self, hidden_dim=HIDDEN_DIM, input_dim=INPUT_DIM,\
                 output_dim=OUTPUT_DIM, n_layers=N_LAYERS, activation=ACTIVATION,\
                      epochs=EPOCHS,\
                    learning_rate=LEARNING_RATE, criterion=CRITERION) -> None:
        super(Hyperparameters, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.activation = activation

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.criterion = criterion