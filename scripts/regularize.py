#%%
import pathsettings
from tools.data import DataHandler
from tools.model import MLP
import torch

SNR = 1.
DATA_HANDLER = DataHandler(snr=SNR)


MODELFILE = "mymodel.torch"
INITIAL_MODEL = MLP()
LAMBDAS = [9., 5., 1., .5, .1, 0.]
LEARNING_RATES = [.1, .01, .001, .0001, .00001]
EPOCHS = 500
PATIENCE = 50

ADJUST_LEARNING_RATES = [.1, .3, .5, .7 , 1., 2., 3., 4.]

def regularize(initial_model=INITIAL_MODEL, data_handler=DATA_HANDLER,
               lambdas=LAMBDAS, learning_rates=LEARNING_RATES,
               adjust_learning_rates=ADJUST_LEARNING_RATES, epochs=EPOCHS,
               patience=PATIENCE, verbosity=0):
    best_lr = None
    best_lam = None
    min_error = float('inf')
    error_heat_map = {}
    for lam in lambdas:
        errors_per_lam = {}
        print(f'\n Regularization Paramter: {lam}')
        for lr in learning_rates:
            data_handler.reset()
            model = initial_model
            criterion = model.hyperparams.criterion
            new_criterion = lambda x,y: criterion(x,y) + lam * model.sum_weights()
            model.hyperparams.criterion = new_criterion
            model.hyperparams.learning_rate = lr
            model.hyperparams.epochs = epochs
            model.start_training(data_handler, verbosity=0, patience=patience)

            model.hyperparams.criterion = criterion
            error = model.start_evaluation(data_handler)
            errors_per_lam.update({str(lr): error})
            if error < min_error:
                min_error = error
                best_lr = lr
                best_lam = lam
            print(f'\t Learning Rate: {lr} \t \t \t Evaluation Error: {round(error, ndigits=4)}')
        error_heat_map.update({str(lam): errors_per_lam})


    print(f'Best Regularization Paramter: {best_lam} \t \t Best Learning Rate: {best_lr}')

    very_best_lr = None
    min_error = float('inf')
    for adjust_lr in adjust_learning_rates:
        data_handler.reset()
        model = initial_model
        lr = best_lr * adjust_lr
        new_criterion = lambda x,y: criterion(x,y) + best_lam * model.sum_weights()
        model.hyperparams.criterion = new_criterion
        model.hyperparams.learning_rate = lr
        model.hyperparams.epochs = 2*epochs
        model.start_training(data_handler, verbosity=0, patience=patience)

        model.hyperparams.criterion = criterion
        error = model.start_evaluation(data_handler)
        if error < min_error:
            min_error = error
            very_best_lr = lr
        print(f'\t Learning Rate: {round(lr, ndigits=10)} \t \t \t Evaluation Error: {round(error, ndigits=4)}')

    print(f'Very Best Learning Rate: {very_best_lr}')
    data_handler.reset()
    model = initial_model
    new_criterion = lambda x,y: criterion(x,y) + best_lam * model.sum_weights()
    model.hyperparams.criterion = new_criterion
    model.hyperparams.learning_rate = very_best_lr
    model.hyperparams.epochs = 2*epochs
    model.start_training(data_handler, verbosity=2, patience=patience)

    model.hyperparams.criterion = criterion
    error = model.start_evaluation(data_handler)
    print(f'Final Evaluation Error: {round(error, ndigits=4)}')

    if verbosity >= 1:
        return model, error_heat_map
    return model


if __name__ == '__main__':
    regularize()

# %%
