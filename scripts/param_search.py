#%%
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


def parameter_search(data_handler=DATA_HANDLER, learning_rates=LEARNING_RATES,
                     adjust_learning_rates=ADJUST_LEARNING_RATES,
                     hidden_dimensions=HIDDEN_DIMENSIONS, epochs=EPOCHS, patience=PATIENCE,
                     hyperparams=HYPERPARAMS, modelfile=MODELFILE):


    best_lr = None
    best_dim = None
    min_error = float('inf')
    for hidden_dim in hidden_dimensions:
        print(f'Hidden Dimension: {hidden_dim}')
        for lr in learning_rates:
            data_handler.reset()
            hyperparams.hidden_dim=hidden_dim
            hyperparams.learning_rate = lr
            hyperparams.epochs = epochs
            model = MLP(hyperparams=hyperparams)
            model.start_training(data_handler, verbosity=0, patience=patience)
            _,_, testlosses = model.start_evaluation(data_handler)
            error = min(testlosses)
            if error < min_error:
                min_error = error
                best_lr = lr
                best_dim = hidden_dim
            print(f'\t Learning Rate: {lr} \t \t Test Error: {error:.2e}')

    print(f'Best Hidden Dimension: {best_dim} \t \t Best Learning Rate: {best_lr}')

    very_best_lr = None
    min_error = float('inf')
    for adjust_lr in adjust_learning_rates:
        data_handler.reset()
        lr = best_lr * adjust_lr
        hyperparams.hidden_dim = best_dim
        hyperparams.learning_rate = lr
        hyperparams.epochs = 2*epochs
        model = MLP(hyperparams=hyperparams)
        model.start_training(data_handler, verbosity=0, patience=patience)
        _,_, testlosses = model.start_evaluation(data_handler)
        error = min(testlosses)
        if error < min_error:
            min_error = error
            very_best_lr = lr
        print(f'\t Learning Rate: {round(lr, ndigits=10)} \t \t Test Error: {error:.3e}')

    print(f'Very Best Learning Rate: {very_best_lr}')
    hyperparams.hidden_dim = best_dim
    hyperparams.learning_rate = very_best_lr
    hyperparams.epochs = 4*epochs
    model = MLP(hyperparams=hyperparams)
    model.start_training(data_handler, verbosity=2, patience=patience)
    model.save(modelfile)
    return model

if __name__ == '__main__':
    parameter_search()


# %%