#%%
from tools.data import DataHandler
from tools.hyperparameters import Hyperparameters
from tools.model import MLP

MODELFILE = "mymodel.torch"
LEARNING_RATES = [.1, .01, .001, .0001, .00001, .000001]
HIDDEN_DIMENSIONS = [80, 160, 320, 640]

data_handler = DataHandler()

for hidden_dim in HIDDEN_DIMENSIONS:
    print(f'Hidden Dimension: {hidden_dim}')
    for lr in LEARNING_RATES:
        params = Hyperparameters(learning_rate=lr, epochs=1000)
        model = MLP(hyperparams=params)
        model.start_training(data_handler, verbosity=0)
        _,_, testlosses = model.start_evaluation(data_handler)
        print(f'\t Learning Rate: {lr} \t \t Loss: {round(min(testlosses), ndigits=4)}')


# %%