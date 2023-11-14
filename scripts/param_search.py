#%%
from tools.data import DataHandler
from tools.hyperparameters import Hyperparameters
from tools.model import MLP


MODELFILE = "mymodel.torch"

LEARNING_RATES = [.1, .01, .001, .0001, .00001]
ADJUST_LEARNING_RATES = [.1, .3, .5, .7 , 1., 2., 3., 4.]
HIDDEN_DIMENSIONS = [80, 160, 320, 640, 1280]

EPOCHS = 500
PATIENCE = 50

SNR = 1.

DATA_HANDLER = DataHandler(snr=SNR)


def parameter_search(data_handler=DATA_HANDLER, learning_rates=LEARNING_RATES,
                     adjust_learning_rates=ADJUST_LEARNING_RATES,
                     hidden_dimensions=HIDDEN_DIMENSIONS, epochs=EPOCHS, patience=PATIENCE,
                     modelfile=MODELFILE):
    params = Hyperparameters()

    best_lr = None
    best_dim = None
    min_error = float('inf')
    for hidden_dim in hidden_dimensions:
        print(f'Hidden Dimension: {hidden_dim}')
        for lr in learning_rates:
            data_handler.reset()
            params.hidden_dim=hidden_dim
            params.learning_rate = lr
            params.epochs = epochs
            model = MLP(hyperparams=params)
            model.start_training(data_handler, verbosity=0, patience=patience)
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
    for adjust_lr in adjust_learning_rates:
        data_handler.reset()
        lr = best_lr * adjust_lr
        params.hidden_dim = best_dim
        params.learning_rate = lr
        params.epochs = 2*epochs
        model = MLP(hyperparams=params)
        model.start_training(data_handler, verbosity=0, patience=patience)
        _,_, testlosses = model.start_evaluation(data_handler)
        error = min(testlosses)
        if error < min_error:
            min_error = error
            very_best_lr = lr
        print(f'\t Learning Rate: {round(lr, ndigits=10)} \t \t \t Evaluation Error: {round(error, ndigits=4)}')

    print(f'Very Best Learning Rate: {very_best_lr}')
    params.hidden_dim = best_dim
    params.learning_rate = very_best_lr
    params.epochs = 4*epochs
    model = MLP(hyperparams=params)
    model.start_training(data_handler, verbosity=2, patience=patience)
    model.save(modelfile)
    return model

if __name__ == '__main__':
    parameter_search()


# %%