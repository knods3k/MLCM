#%%
from tools.model import MLP
from tools.hyperparameters import Hyperparameters
from tools.data import DataHandler
from tools.plotting import eval_and_plot

data_handler = DataHandler(snr=1.)
params = Hyperparameters(epochs=2000, learning_rate=.0005)
model = MLP(hyperparams=params) 
model.start_training(data_handler)

eval_and_plot(model, data_handler)

# %%
eval_metrics_overfit = eval_and_plot(model, data_handler)
