#%%
from tools.plotting import  eval_and_plot
from tools.data import DataHandler
import torch
from param_search import data_handler

# SNR = 1.
# data_handler = DataHandler(snr=SNR)


MODELFILE = "mymodel.torch"
LAMBDAS = [9., 5., 1., .5, .1, 0.]
LEARNING_RATES = [.1, .01, .001, .0001, .00001]
EPOCHS = 500
PATIENCE = 50

ADJUST_LR = [.1, .3, .5, .7 , 1., 2., 3., 4.]

best_lr = None
best_dim = None
min_error = float('inf')
for lam in LAMBDAS:

    print(f'Regularization Paramter: {lam}')
    for lr in LEARNING_RATES:
        data_handler.reset()
        model = torch.load(MODELFILE)
        criterion = model.hyperparams.criterion
        new_criterion = lambda x,y: criterion(x,y) + lam * model.sum_weights()
        model.hyperparams.criterion = new_criterion
        model.hyperparams.learning_rate = lr
        model.hyperparams.epochs = EPOCHS
        model.start_training(data_handler, verbosity=0, patience=PATIENCE)

        model.hyperparams.criterion = criterion
        _,_, testlosses = model.start_evaluation(data_handler)

        error = min(testlosses)
        if error < min_error:
            min_error = error
            best_lr = lr
            best_lam = lam

        print(f'\t Learning Rate: {lr} \t \t \t Evaluation Error: {round(error, ndigits=4)}')

print(f'Best Regularization Paramter: {best_lam} \t \t Best Learning Rate: {best_lr}')

very_best_lr = None
min_error = float('inf')
for adjust_lr in ADJUST_LR:
    data_handler.reset()
    model = torch.load(MODELFILE)
    lr = best_lr * adjust_lr
    new_criterion = lambda x,y: criterion(x,y) + best_lam * model.sum_weights()
    model.hyperparams.criterion = new_criterion
    model.hyperparams.learning_rate = lr
    model.hyperparams.epochs = 2*EPOCHS
    model.start_training(data_handler, verbosity=0, patience=PATIENCE)

    model.hyperparams.criterion = criterion
    _,_, testlosses = model.start_evaluation(data_handler)
    error = min(testlosses)
    if error < min_error:
        min_error = error
        very_best_lr = lr
    print(f'\t Learning Rate: {round(lr, ndigits=10)} \t \t \t Evaluation Error: {round(error, ndigits=4)}')

print(f'Very Best Learning Rate: {very_best_lr}')
data_handler.reset()
model = torch.load(MODELFILE)
new_criterion = lambda x,y: criterion(x,y) + best_lam * model.sum_weights()
model.hyperparams.criterion = new_criterion
model.hyperparams.learning_rate = very_best_lr
model.hyperparams.epochs = 2*EPOCHS
model.start_training(data_handler, verbosity=2, patience=PATIENCE)

model.hyperparams.criterion = criterion
_,_, testlosses = model.start_evaluation(data_handler)
model.save('regularized_'+MODELFILE)

eval_and_plot(model, data_handler)



# %%
