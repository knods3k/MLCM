#%%
import scripts.scriptsettings
from tools.settings import DEVICE, MACHINE_EPSILON
from tools.material import HyperelasticMaterial
from tools.model import MLP
from tools.data import MaterialDataHandler
from tools.hyperparameters import Hyperparameters

from scripts.param_search import parameter_search

import torch
import matplotlib.pyplot as plt

def emale(true, pred):
    return torch.exp(
        torch.log(true/pred).abs().mean()
        ).detach().cpu()

MATERIAL = HyperelasticMaterial()
HIDDEN_DIMS = [10, 20, 30, 40]
INPUT_DIM = 3
N_LAYERS = 5
EPOCHS = 250
PATIENCE = 10

DEFAULT_DEFORMATION_FUNCTION = 'incompressible'


def fit_material_model(material=MATERIAL, n_layers=N_LAYERS,
                       epochs=EPOCHS, patience=PATIENCE,
                       deformation_function=DEFAULT_DEFORMATION_FUNCTION):
    '''
    Train an MLP to predict the Helmholtz Free Energy.
    '''
    torch.random.manual_seed(1)
    data_handler = MaterialDataHandler(material=material,
                                       deformation_function=deformation_function,
                                       max_body_scale=1, batchsize=16, samples=1, body_resolution=10)

    hyperparams = Hyperparameters(input_dim=INPUT_DIM, n_layers=n_layers)
    model = parameter_search(data_handler, hyperparams=hyperparams,
                     epochs=epochs, patience=patience, hidden_dimensions=HIDDEN_DIMS)


    return model, data_handler
    

def test_material_model(model, data_handler, material=MATERIAL):
    incompressible = lambda x, factor: torch.stack([factor*x[:,:,0], 1/factor*x[:,:,1]], axis=-1)
    d_functions = {
        '1x': lambda x: incompressible(x, 1),
        '2x': lambda x: incompressible(x, 2),
        '10x': lambda x: incompressible(x, 10),
        'A1x': data_handler.random_incompressible_deformation,
        'A2x': data_handler.random_incompressible_deformation,
        'A3x': data_handler.random_incompressible_deformation,
        # '1.1x': lambda x: 1.1*x,
        # '2.0x': lambda x: 2.*x,
        # '100x': lambda x: 100.*x,
        # 'Sqaure': torch.square,
        # 'Root': torch.sqrt,
        # 'Exp.': torch.exp,
        # 'Log.': torch.log
    }
    print('\n\n\n')
    torch.random.manual_seed(1)
    for func_name, function in d_functions.items():
        test_body_size = 1000
        n_test_bodies = 10
        scale_test_bodies = 100.
        X = torch.rand((n_test_bodies,test_body_size,2), requires_grad=True)*scale_test_bodies
        material.set_body_configuration(X)
        u = material.deform(function)
        x = material.x

        energy_true = material.get_helmholtz_free_energy()
        input_data = material.get_invariant_deviations()
        input_data /= data_handler.normalizing_constant_in
        energy_predicted = model((input_data).to(DEVICE)).cpu().squeeze()

        energy_predicted *= data_handler.normalizing_constant_out
        energy_predicted += MACHINE_EPSILON
        energy_true += MACHINE_EPSILON
        error = emale(energy_true, energy_predicted.detach().cpu().numpy())

        body = X[0].detach().cpu().numpy()
        deformed = x[0].detach().cpu().numpy()
        plt.xlabel(f"\nDeformation Function: {func_name} \n eMALE: {error:2.5}")
        plt.title(f'(Predicted) {energy_predicted.mean():.3} : {energy_true.mean():.3} (True)')
        plt.scatter(body[:,0], body[:,1], c='w', label='Original')
        plt.scatter(deformed[:,0], deformed[:,1], c='r', label='Deformed')
        plt.legend()
        plt.show()
    return model

if __name__ == "__main__":
    model, data_handler = fit_material_model()
    test_material_model(model, data_handler)

# %%
