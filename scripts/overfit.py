#%%
from tools.model import MLP
from tools.hyperparameters import Hyperparameters
from tools.data import DataHandler
from tools.plotting import eval_and_plot

EPOCHS = 2000
LEARNING_RATE = .0005
SNR = 1.
DATA_HANDLER = DataHandler(snr=SNR)

def overfit(data_handler=DATA_HANDLER, epochs=EPOCHS, learning_rate=LEARNING_RATE):
    params = Hyperparameters(epochs=EPOCHS, learning_rate=LEARNING_RATE)
    model = MLP(hyperparams=params) 
    model.start_training(data_handler)
    return model
