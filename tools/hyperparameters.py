import torch.nn as nn

HIDDEN_DIM = 640
INPUT_DIM = 2
OUTPUT_DIM = 1
N_LAYERS = 5
ACTIVATION = nn.ReLU()

EPOCHS = 1500
LEARNING_RATE = .001
BATCH_SIZE = 160
CRITERION = nn.MSELoss(reduction='mean')

class Hyperparameters():
    """
    Hyperparameters is a class that contains all the hyperparameters for the network. The parameters are set as
    global variables.
    :param hidden_dim: The number of neurons in each hidden layer.
    :param input_dim: The number of input neurons.
    :param output_dim: The number of output neurons.
    :param n_layers: The number of hidden layers.
    :param activation: The activation function to use.
    :param epochs: The number of epochs to train the network.
    :param learning_rate: The learning rate to use for the optimizer.
    :param criterion: The loss function to use.
    """
    
    def __init__(self, hidden_dim=HIDDEN_DIM, input_dim=INPUT_DIM,
                 output_dim=OUTPUT_DIM, n_layers=N_LAYERS,
                 activation=ACTIVATION, epochs=EPOCHS,
                 learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
                 criterion=CRITERION) -> None:
      
        super(Hyperparameters, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.activation = activation

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.criterion = criterion