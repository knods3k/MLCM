#%%
from tools.hyperparameters import Hyperparameters
from tools.model import MLP
from tools.data import DataHandler

hyperparams = Hyperparameters(hidden_dim=3)
data_handler = DataHandler()
model = MLP(hyperparams=hyperparams)
model.start_training(data_handler)
# %%
