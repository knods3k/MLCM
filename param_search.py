#%%
from numpy import Inf
from tools.data import DataHandler
from tools.hyperparameters import Hyperparameters
from tools.model import MLP
import torch.nn as nn

MODELFILE = "mymodel.torch"
LEARNING_RATES = [.1, .01, .001, .0001, .00001]
HIDDEN_DIMENSIONS = [80, 160, 320, 640, 1280]
EPOCHS = 250

ADJUST_LR = [.1, .25, .5, .75 , 1., 1.5, 2., 4.]

data_handler = DataHandler()
params = Hyperparameters()

best_lr = None
best_dim = None
min_error = Inf
for hidden_dim in HIDDEN_DIMENSIONS:
    print(f'Hidden Dimension: {hidden_dim}')
    for lr in LEARNING_RATES:
        params.hidden_dim=hidden_dim
        params.learning_rate = lr
        params.epochs = EPOCHS
        model = MLP(hyperparams=params)
        model.start_training(data_handler, verbosity=0)
        _,_, testlosses = model.start_evaluation(data_handler)
        error = min(testlosses)
        if error < min_error:
            min_error = error
            best_lr = lr
            best_dim = hidden_dim
        print(f'\t Learning Rate: {lr} \t \t \t Loss: {round(error, ndigits=4)}')

print(f'Best Hidden Dimension: {best_dim} \t \t \t Best Learning Rate: {best_lr}')

very_best_lr = best_lr
min_error = Inf
for adjust_lr in ADJUST_LR:
    lr = best_lr * adjust_lr
    params.hidden_dim = best_dim
    params.learning_rate = lr
    params.epochs = 2*EPOCHS
    model = MLP(hyperparams=params)
    model.start_training(data_handler, verbosity=0)
    _,_, testlosses = model.start_evaluation(data_handler)
    error = min(testlosses)
    if error < min_error:
        min_error = error
        very_best_lr = lr
    print(f'\t Learning Rate: {round(lr, ndigits=10)} \t \t \t Loss: {round(error, ndigits=4)}')

print(f'Very Best Learning Rate: {very_best_lr}')
params.hidden_dim = best_dim
params.learning_rate = very_best_lr
params.epochs = 4*EPOCHS
model = MLP(hyperparams=params)
model.start_training(data_handler, verbosity=2)
model.save(MODELFILE)



# %%