#%%
import pathsettings
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from tools.hyperparameters import Hyperparameters
from tools.model import MLP
from scripts.fit_material import INPUT_DIM,PATIENCE, HIDDEN_DIMS, DATA_HANDLER
from scripts.param_search import parameter_search

import torch

EPOCHS=25

def plot_error_heat_map(error_dict, plot_title, xlabel='Hidden Dimension', ylabel='Learning Rate'):
    df = pd.DataFrame(error_dict)
    sns.heatmap(df, annot=True, fmt='.3g', cbar_kws={'label': 'MSE'}, cmap='Oranges', norm=LogNorm())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_title)

if __name__ == '__main__':
    torch.random.manual_seed(1)

    hyperparams = Hyperparameters(input_dim=INPUT_DIM)
    hyperparams = Hyperparameters(input_dim=INPUT_DIM, epochs=EPOCHS, hidden_dim=55)
    model = MLP(hyperparams=hyperparams)

    model, error_heat_map = parameter_search(model, data_handler=DATA_HANDLER, epochs=EPOCHS, patience=PATIENCE,
                                hidden_dimensions=HIDDEN_DIMS, verbosity=1)

    plot_error_heat_map(error_heat_map, 'Material Model Parameter Search', xlabel='Hidden Dimension', ylabel='Learning Rate')
# %%