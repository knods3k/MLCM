#%%
import pathsettings
from tools.settings import DEVICE, MACHINE_EPSILON
from tools.material import HyperelasticMaterial
from tools.model import MLP
from tools.data import MaterialDataHandler
from tools.hyperparameters import Hyperparameters

from scripts.param_search import parameter_search

import torch
import matplotlib.pyplot as plt

def exp_mean_log_error(true, pred):
    '''
    Compute the exponential mean logarithmic error (eMLE).

    The eMLE is 1 iff true and predicted values are exactly equal.
    Large underestimations tend to 0.
    Large overestimations tend to infinity.
    '''
    return torch.exp(
        torch.log(pred/true).mean()
        ).detach().cpu()

MATERIAL = HyperelasticMaterial()
DATA_HANDLER = MaterialDataHandler(material=MATERIAL,
                                       max_body_scale=1, batchsize=160, samples=1,
                                       body_resolution=10)
HIDDEN_DIMS = [35, 40, 45, 50, 55, 60]
INPUT_DIM = 3
EPOCHS = 2000
PATIENCE = 10


def fit_material_model(data_handler=DATA_HANDLER, epochs=EPOCHS,
                       patience=PATIENCE):
    '''
    Train an MLP to predict the Helmholtz Free Energy.
    '''
    data_handler.reset()

    hyperparams = Hyperparameters(input_dim=INPUT_DIM)
    model = MLP(hyperparams=hyperparams)
    model, error_heat_map = parameter_search(model, data_handler, epochs=epochs, patience=patience,
                             hidden_dimensions=HIDDEN_DIMS)


    return model, data_handler
    

def test_material_model(model, data_handler, material=MATERIAL):
    incompressible_deformation = lambda x, factor: x @ torch.tensor([[factor, 0], [0, 1/factor]])
    d_functions = {
        '1x': lambda x: incompressible_deformation(x, 1),
        '2x': lambda x: incompressible_deformation(x, 2),
        '10x': lambda x: incompressible_deformation(x, 10),
        '100x': lambda x: incompressible_deformation(x, 100),
        '1000x': lambda x: incompressible_deformation(x, 1000),
        'A1x': data_handler.random_incompressible_deformation,
        'A2x': data_handler.random_incompressible_deformation,
        'A3x': data_handler.random_incompressible_deformation,
    }
    print('\n\n\n')
    torch.random.manual_seed(1)
    for func_name, deformation_function in d_functions.items():
        test_body_size = 1000
        n_test_bodies = 10
        scale_test_bodies = 100.
        X = torch.rand((n_test_bodies,test_body_size,2), requires_grad=True)*scale_test_bodies
        material.set_body_configuration(X)
        u = material.deform(deformation_function)
        x = material.x

        energy_true = material.get_helmholtz_free_energy()
        input_data = material.get_invariant_deviations()
        input_data /= data_handler.normalizing_constant_in
        energy_predicted = model((input_data).to(DEVICE)).cpu().squeeze()

        energy_predicted *= data_handler.normalizing_constant_out
        energy_predicted += MACHINE_EPSILON
        energy_true += MACHINE_EPSILON
        error = exp_mean_log_error(energy_true, energy_predicted)

        body = X[0].detach().cpu().numpy()
        deformed = x[0].detach().cpu().numpy()
        plt.xlabel(f"\nDeformation Function: {func_name} \n eMALE: {error:2.5}")
        plt.title(f'(Predicted) {energy_predicted.mean():.3} : {energy_true.mean():.3} (True)')
        plt.scatter(body[:,0], body[:,1], s=1, c='w', label='Original')
        plt.scatter(deformed[:,0], deformed[:,1], s=1, c='r', label='Deformed')
        plt.legend()
        plt.show()
    
    error_per_deformation_amount = []
    # deformation_amounts = torch.linspace(2,1000,999)
    deformation_amounts = torch.logspace(0, 3, 99) + .5
    for deformation_amount in deformation_amounts:
        deformation_function = lambda x: incompressible_deformation(x, deformation_amount)
        test_body_size = 100
        n_test_bodies = 1
        scale_test_bodies = 100.

        X = torch.rand((n_test_bodies,test_body_size,2), requires_grad=True)*scale_test_bodies
        material.set_body_configuration(X)
        u = material.deform(deformation_function)
        x = material.x

        energy_true = material.get_helmholtz_free_energy()
        input_data = material.get_invariant_deviations()
        input_data /= data_handler.normalizing_constant_in
        energy_predicted = model((input_data).to(DEVICE)).cpu().squeeze()

        energy_predicted *= data_handler.normalizing_constant_out
        energy_predicted += MACHINE_EPSILON
        energy_true += MACHINE_EPSILON
        error = exp_mean_log_error(energy_true, energy_predicted)
        error_per_deformation_amount.append(error.detach().cpu().numpy())

    plt.plot(deformation_amounts, error_per_deformation_amount,
             label="Model Performance")
    plt.plot(deformation_amounts, torch.ones_like(deformation_amounts),
             'r--',alpha=.5, label="Optimum")
    plt.ylabel('eMALE')
    plt.xscale('log')
    plt.xlabel('Deformation Amount')
    plt.legend()
    plt.show()

    return model

if __name__ == "__main__":
    model, data_handler = fit_material_model()
    test_material_model(model, data_handler)

# %%
