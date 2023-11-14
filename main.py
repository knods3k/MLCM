#%%
from tools.hyperparameters import Hyperparameters
from tools.model import MLP
from tools.data import DataHandler
from tools.plotting import eval_and_plot

hyperparams = Hyperparameters(hidden_dim=3)
data_handler = DataHandler()
model = MLP(hyperparams=hyperparams)
model.start_training(data_handler)
eval_and_plot(model, data_handler)
# %%
