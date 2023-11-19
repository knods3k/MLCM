#%%
from tools.material import HyperelasticMaterial
from tools.model import MLP
from tools.data import MaterialDataHandler

import torch

m = HyperelasticMaterial()

def deformation_function(X):
    return X**2 + torch.sin(X)

if __name__ == "__main__":
    '''
    This trains an MLP to predict the Helmholtz Free Energy.
    '''
    data_handler = MaterialDataHandler(material=m, deformation_function=deformation_function)
    model = MLP()
    model.hyperparams.epochs=10
    model.input_dim = 4
    model.build()
    model.start_training(data_handler)

# %%
