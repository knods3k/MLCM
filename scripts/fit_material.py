#%%
from tools.material import HyperelasticMaterial
from tools.model import MLP
from tools.data import DataHandler
from tools.settings import DEVICE

import torch
import matplotlib.pyplot as plt

m = HyperelasticMaterial()

def target_function(X):
    u = X**3 + X**2 + X + 1
    return m.get_helmholtz_free_energy(X,u)

if __name__ == "__main__":
    '''
    This trains an MLP to predict the Helmholtz Free Energy
    for one point in 2D and a corresponding displacement.
    For a surface in 2D this will predict local energies over
    the surface. Summing the result will yield the total energy. 
    '''
    data_handler = DataHandler(function=target_function)
    model = MLP()
    model.start_training(data_handler)

    X, _ = data_handler.get_test_data()
    X_mesh, _ = data_handler.get_mesh()
    local_energies = model(X.to(DEVICE)).reshape(X_mesh.shape).detach().cpu().numpy()
    total_energy = local_energies.sum()
    
    plt.imshow(local_energies, cmap='hot')


# %%
