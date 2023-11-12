#%%
from tools.data import DataHandler
from tools.hyperparameters import Hyperparameters
from tools.model import MLP
import torch.nn as nn

MODELFILE = "mymodel.torch"
LEARNING_RATES = [.1, .01, .001, .0001, .00001]
HIDDEN_DIMENSIONS = [80, 160, 320, 640, 1280]
EPOCHS = 500
PATIENCE = 50

ADJUST_LR = [.1, .3, .5, .7 , 1., 2., 3., 4.]

SNR = 1.

data_handler = DataHandler(snr=SNR)
params = Hyperparameters()

best_lr = None
best_dim = None
min_error = float('inf')
for hidden_dim in HIDDEN_DIMENSIONS:
    print(f'Hidden Dimension: {hidden_dim}')
    for lr in LEARNING_RATES:
        data_handler.reset()
        params.hidden_dim=hidden_dim
        params.learning_rate = lr
        params.epochs = EPOCHS
        model = MLP(hyperparams=params)
        model.start_training(data_handler, verbosity=0, patience=PATIENCE)
        _,_, testlosses = model.start_evaluation(data_handler)
        error = min(testlosses)
        if error < min_error:
            min_error = error
            best_lr = lr
            best_dim = hidden_dim
        print(f'\t Learning Rate: {lr} \t \t \t Evaluation Error: {round(error, ndigits=4)}')

print(f'Best Hidden Dimension: {best_dim} \t \t \t Best Learning Rate: {best_lr}')

very_best_lr = None
min_error = float('inf')
for adjust_lr in ADJUST_LR:
    data_handler.reset()
    lr = best_lr * adjust_lr
    params.hidden_dim = best_dim
    params.learning_rate = lr
    params.epochs = 2*EPOCHS
    model = MLP(hyperparams=params)
    model.start_training(data_handler, verbosity=0, patience=PATIENCE)
    _,_, testlosses = model.start_evaluation(data_handler)
    error = min(testlosses)
    if error < min_error:
        min_error = error
        very_best_lr = lr
    print(f'\t Learning Rate: {round(lr, ndigits=10)} \t \t \t Evaluation Error: {round(error, ndigits=4)}')

print(f'Very Best Learning Rate: {very_best_lr}')
params.hidden_dim = best_dim
params.learning_rate = very_best_lr
params.epochs = 4*EPOCHS
model = MLP(hyperparams=params)
model.start_training(data_handler, verbosity=2, patience=PATIENCE)
model.save(MODELFILE)



# %%