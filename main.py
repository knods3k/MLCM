#%%
from tools.data import DataHandler
from tools.plotting import eval_and_plot
from scripts.overfit import overfit
from scripts.param_search import parameter_search
from scripts.regularize import regularize

import torch

MODELFILE = "mymodel.torch"

LEARNING_RATES = [.1, .01, .001, .0001, .00001]
ADJUST_LEARNING_RATES = [.1, .3, .5, .7 , 1., 2., 3., 4.]
HIDDEN_DIMENSIONS = [80, 160, 320, 640, 1280]

EPOCHS = 2000
PATIENCE = 50

SNR = 1.

DATA_HANDLER = DataHandler(snr=SNR)

INITIAL_MODEL = torch.load(MODELFILE)
LAMBDAS = [9., 5., 1., .5, .1, 0.]

if __name__ == '__main__':
    overfitted_model = overfit(data_handler=DATA_HANDLER, epochs=EPOCHS)
    eval_and_plot(overfitted_model, DATA_HANDLER, plot_title='Overfitting')

    initial_model = parameter_search(data_handler=DATA_HANDLER, epochs=EPOCHS,
                                     learning_rates=LEARNING_RATES,
                                     adjust_learning_rates=ADJUST_LEARNING_RATES,
                                     hidden_dimensions=HIDDEN_DIMENSIONS, patience=PATIENCE)
    eval_and_plot(initial_model, DATA_HANDLER, plot_title='Initial Model')

    regularized_model = regularize(initial_model=INITIAL_MODEL,
                                   data_handler=DATA_HANDLER, epochs=EPOCHS,
                                     learning_rates=LEARNING_RATES,
                                     adjust_learning_rates=ADJUST_LEARNING_RATES,
                                     lambdas=LAMBDAS, patience=PATIENCE)
    eval_and_plot(regularized_model, DATA_HANDLER, plot_title='Regularized Model')



# %%
