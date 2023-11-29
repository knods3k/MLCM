#%%
from tools.settings import DEVICE
from tools.material import HyperelasticMaterial
from tools.model import MLP
from tools.data import MaterialDataHandler
from tools.hyperparameters import Hyperparameters

from scripts.param_search import parameter_search

import torch
import matplotlib.pyplot as plt

MATERIAL = HyperelasticMaterial()
HIDDEN_DIM = 120
N_LAYERS = 5
EPOCHS = 1000
PATIENCE = 100


def deformation_function(X):
    return X**3 + X**2 + torch.sin(X) + torch.log(X)


def fit_material_model(material=MATERIAL, n_layers=N_LAYERS,
                       epochs=EPOCHS, patience=PATIENCE,
                       deformation_function=deformation_function):
    '''
    Train an MLP to predict the Helmholtz Free Energy.
    '''
    torch.random.manual_seed(1)
    data_handler = MaterialDataHandler(material=material,
                                       deformation_function=deformation_function)

    hyperparams = Hyperparameters(input_dim=4, n_layers=n_layers)
    model = parameter_search(data_handler, hyperparams=hyperparams,
                     epochs=epochs, patience=patience)


    return model
    

def test_material_model(model, material=MATERIAL):
    d_functions = {
        '1.1x': lambda x: 1.1*x,
        '100x': lambda x: 100.*x,
        'Sqaure': torch.square,
        'Root': torch.sqrt,
        'Exp.': torch.exp,
        'Log.': torch.log
    }
    print('\n\n\n')
    for func_name, function in d_functions.items():
        torch.random.manual_seed(1)
        test_body_size = 1000
        n_test_bodies = 10
        scale_test_bodies = 1.
        X = torch.rand((n_test_bodies,test_body_size,2), requires_grad=True)*scale_test_bodies
        material.set_body_configuration(X)
        u = material.deform(function)
        x = material.x

        energy_true = material.get_helmholtz_free_energy()
        deformation_tensor = material.C.flatten(start_dim=-2)
        energy_predicted = model((deformation_tensor).to(DEVICE)).cpu().squeeze()

        eps = torch.finfo(float).eps
        energy_predicted += eps
        energy_true += eps
        relative_error = torch.exp(torch.log(energy_predicted/energy_true).abs().mean()) -1
        relative_error = relative_error.mean().detach().numpy()
        relative_error *= 100

        body = X[0].detach().cpu().numpy()
        deformed = x[0].detach().cpu().numpy()
        plt.xlabel(f"\nDeformation Function: {func_name} \n Mean Relative Error: {relative_error:2.5}%")
        plt.title(f'(Predicted) {energy_predicted.mean():.3} : {energy_true.mean():.3} (True)')
        plt.scatter(body[:,0], body[:,1], c='w', label='Original')
        plt.scatter(deformed[:,0], deformed[:,1], c='r', label='Deformed')
        plt.legend()
        plt.show()

        # print(f"\nDeformation Function: {func_name} \t Mean Relative Error: {relative_error:.3}% \t \t")
    return model

if __name__ == "__main__":
    model = fit_material_model()
    test_material_model(model)

    # %%
