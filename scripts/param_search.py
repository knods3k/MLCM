#%%
import pathsettings
from tools.data import DataHandler
from tools.hyperparameters import Hyperparameters
from tools.model import MLP


MODELFILE = "mymodel.torch"

LEARNING_RATES = [.1, .01, .001, .0001, .00001]
ADJUST_LEARNING_RATES = [.1, .3, .5, .7 , 1., 2., 3., 4.]
HIDDEN_DIMENSIONS = [8, 16, 32, 64, 128]

EPOCHS = 500
PATIENCE = 50

SNR = 1.

DATA_HANDLER = DataHandler(snr=SNR)
HYPERPARAMS = Hyperparameters()
MODEL = MLP(hyperparams=HYPERPARAMS)


def parameter_search(initial_model = MODEL, data_handler=DATA_HANDLER, learning_rates=LEARNING_RATES,
                     adjust_learning_rates=ADJUST_LEARNING_RATES,
                     hidden_dimensions=HIDDEN_DIMENSIONS, epochs=EPOCHS, patience=PATIENCE,
                     verbosity=0):


    best_lr = None
    best_dim = None
    min_error = float('inf')
    error_heat_map = {}
    for hidden_dim in hidden_dimensions:
        print(f'\n Hidden Dimension: {hidden_dim}')
        errors_per_learning_rate = {}
        for lr in learning_rates:
            data_handler.reset()
            model = initial_model
            model.hyperparams.hidden_dim=hidden_dim
            model.hyperparams.learning_rate = lr
            model.hyperparams.epochs = epochs
            model.build()
            model.start_training(data_handler, verbosity=0, patience=patience)
            data_handler.reset()
            error = model.start_evaluation(data_handler)
            errors_per_learning_rate.update({str(lr): error})
            if error < min_error:
                min_error = error
                best_lr = lr
                best_dim = hidden_dim
            print(f'\t Learning Rate: {lr} \t \t Test Error: {error:.2e}')
        error_heat_map.update({str(hidden_dim): errors_per_learning_rate})

    print(f'Best Hidden Dimension: {best_dim} \t \t Best Learning Rate: {best_lr}')

    very_best_lr = None
    min_error = float('inf')
    for adjust_lr in adjust_learning_rates:
        data_handler.reset()
        lr = best_lr * adjust_lr
        model = initial_model
        model.hyperparams.hidden_dim=best_dim
        model.hyperparams.learning_rate = lr
        model.hyperparams.epochs = epochs*2
        model.build()
        model.start_training(data_handler, verbosity=0, patience=patience)
        data_handler.reset()
        error = model.start_evaluation(data_handler)
        if error < min_error:
            min_error = error
            very_best_lr = lr
        print(f'\t Learning Rate: {round(lr, ndigits=10)} \t \t Test Error: {error:.3e}')

    print(f'Very Best Learning Rate: {very_best_lr}')
    data_handler.reset()
    model = initial_model
    model.hyperparams.hidden_dim=best_dim
    model.hyperparams.learning_rate = very_best_lr
    model.hyperparams.epochs = epochs*4
    model.build()
    model.start_training(data_handler, verbosity=0, patience=patience)
    data_handler.reset()
    error = model.start_evaluation(data_handler)
    print(f'\t Learning Rate: {round(very_best_lr, ndigits=10)} \t \t Test Error: {error:.3e}')
    
    if verbosity >= 1:
        return model, error_heat_map
    return model

if __name__ == '__main__':
    parameter_search()


# %%