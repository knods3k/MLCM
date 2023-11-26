#%%
from tools.settings import DEVICE
from tools.material import HyperelasticMaterial
from tools.model import MLP
from tools.data import MaterialDataHandler

import torch

MATERIAL = HyperelasticMaterial()
HIDDEN_DIM = 120
N_LAYERS = 5
EPOCHS = 20000


def deformation_function(X):
    return X**3 + X**2 + torch.sin(X) + torch.log(X)


def fit_material_model(material=MATERIAL, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
                       epochs=EPOCHS, deformation_function=deformation_function):
    '''
    Train an MLP to predict the Helmholtz Free Energy.
    '''
    torch.random.manual_seed(1)
    data_handler = MaterialDataHandler(material=material,
                                       deformation_function=deformation_function)
    model = MLP()
    model.input_dim = 4
    model.hidden_dim = hidden_dim
    model.n_layers = n_layers
    model.build()
    model.hyperparams.epochs = epochs
    for exponent in torch.linspace(1,4, 11):
        lr = .1**exponent
        print(f'\n \n Learning Rate: {lr:1.1e}')
        model.hyperparams.learning_rate=lr
        model.start_training(data_handler, patience=50, verbosity=1)

    return model
    

def test_material_model(model, material=MATERIAL):
    d_functions = {
        'Linear': lambda x: 2.*x,
        'Sqaure': torch.square,
        'Root': torch.sqrt,
        'Exp.': torch.exp,
        'Sinus': torch.sin,
        'Log.': torch.log
    }
    print('\n\n\n')
    for func_name, function in d_functions.items():
        test_body_size = 1000
        n_test_bodies = 10
        scale_test_bodies = .1
        X = torch.rand((n_test_bodies,test_body_size,2), requires_grad=True)*scale_test_bodies
        material.set_body_configuration(X)
        u = material.deform(function)

        energy_true = material.get_helmholtz_free_energy(X,u)
        deformation_tensor = material.C.flatten(start_dim=1)
        energy_predicted = model((deformation_tensor).to(DEVICE)).cpu().squeeze()

        relative_error = torch.abs( (energy_predicted - energy_true) / energy_true)
        relative_error = relative_error.mean().detach().numpy()
        relative_error *= 100

        print(f"\nDeformation Function: {func_name} \t Mean Relative Error: {relative_error:2.3}% \t \t")
    return model

if __name__ == "__main__":
    model = fit_material_model()
    test_material_model(model)

    # %%
