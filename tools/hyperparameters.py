import torch.nn as nn

HIDDEN_DIM = 640
INPUT_DIM = 2
OUTPUT_DIM = 1
N_LAYERS = 3

EPOCHS = 1500
LEARNING_RATE = .001

class Hyperparameters():
    def __init__(self, hidden_dim=HIDDEN_DIM, input_dim=INPUT_DIM,\
                 output_dim=OUTPUT_DIM, n_layers=N_LAYERS) -> None:
        super(Hyperparameters, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers