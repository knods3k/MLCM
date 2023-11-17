#%%
from tools.material import HyperelasticMaterial
from tools.model import MLP
from tools.data import DataHandler

import torch

m = HyperelasticMaterial()

def target_function(X):
    u = X**3
    return m.get_helmholtz_free_energy(X,u)

if __name__ == "__main__":
    data_handler = DataHandler(function=target_function)
    model = MLP()
    model.start_training(data_handler)

# %%
