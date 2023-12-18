#%%
import pathsettings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tools.hyperparameters import Hyperparameters
from tools.data import MaterialDataHandler
from tools.model import MLP
from scripts.fit_material import (MATERIAL, DEFAULT_DEFORMATION_FUNCTION,INPUT_DIM,
                                  PATIENCE, HIDDEN_DIMS, N_LAYERS)
from scripts.param_search import parameter_search

import torch

EPOCHS=2

torch.random.manual_seed(1)
data_handler = MaterialDataHandler(material=MATERIAL,
                                    deformation_function=DEFAULT_DEFORMATION_FUNCTION,
                                    max_body_scale=1, batchsize=16, samples=1,
                                    body_resolution=10)

hyperparams = Hyperparameters(input_dim=INPUT_DIM, n_layers=N_LAYERS)
hyperparams = Hyperparameters(input_dim=INPUT_DIM, n_layers=N_LAYERS, epochs=EPOCHS, hidden_dim=55)
model = MLP(hyperparams=hyperparams)

# model.start_training(data_handler, verbosity=2)


#%%
model, error_heat_map = parameter_search(model, data_handler, epochs=EPOCHS, patience=PATIENCE,
                            hidden_dimensions=HIDDEN_DIMS, verbosity=1)

# %%
#df = pd.DataFrame(error_heat_map)
#sns.heatmap(df, annot=True, fmt='.3g', cbar_kws={'label': 'MSE'}, cmap='Oranges')
#plt.xlabel('Hidden Dimension')
#plt.ylabel('Learning Rate')
#plt.savefig('ignore/heatmap')

# %%
def plot_error_heat_map(error_dict, plot_title, xlabel='Hidden Dimension', ylabel='Learning Rate'):
    df = pd.DataFrame(error_dict)
    sns.heatmap(df, annot=True, fmt='.3g', cbar_kws={'label': 'MSE'}, cmap='Oranges')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_title)
    #plt.savefig('ignore/heatmap')

# %%
plot_error_heat_map(error_heat_map, 'Material Model Parameter Search', xlabel='Hidden Dimension', ylabel='Learning Rate')
# %%