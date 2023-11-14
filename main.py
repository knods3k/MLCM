#%%
from tools.data import DataHandler
from tools.plotting import eval_and_plot
from scripts.overfit import overfit
from scripts.param_search import parameter_search
from scripts.regularize import regularize

SNR=1.
data_handler = DataHandler(snr=SNR)

if __name__ == '__main__':
    overfitted_model = overfit(data_handler=data_handler)
    eval_and_plot(overfitted_model, data_handler, plot_title='Overfitting')

    initial_model = parameter_search(data_handler=data_handler)
    eval_and_plot(initial_model, data_handler, plot_title='Initial Model')

    regularized_model = regularize(data_handler=data_handler)
    eval_and_plot(regularized_model, data_handler, plot_title='Regularized Model')



# %%
