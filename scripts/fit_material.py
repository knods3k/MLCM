#%%
import sys
sys.path.append('')
from tools.material import HyperelasticMaterial
from tools.model import MLP
from tools.data import DataHandler
from tools.settings import DEVICE
from tools.plotting import eval_and_plot

import torch

m = HyperelasticMaterial()

def target_function(X1, X2):
    X = torch.stack((X1,X2), dim=-1).squeeze()
    X.requires_grad = True
    u = X**3 + X**2 + X + 1
    return m.get_helmholtz_free_energy(X,u) * torch.ones_like(X1)

if __name__ == "__main__":
    '''
    This trains an MLP to predict the Helmholtz Free Energy.
    '''
    data_handler = DataHandler(target_function=target_function)
    model = MLP()
    model.start_training(data_handler)

    X, _ = data_handler.get_test_data()
    X_mesh, _ = data_handler.get_mesh()
    local_energies = model(X.to(DEVICE)).reshape(X_mesh.shape).detach().cpu().numpy()
    total_energy = local_energies.mean()
    eval_and_plot(model, data_handler)

# %%
