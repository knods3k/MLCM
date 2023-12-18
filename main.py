#%%
import pathsettings
from tools.data import DataHandler, MaterialDataHandler
from tools.material import HyperelasticMaterial
from tools.model import MLP
from tools.plotting import eval_and_plot
from scripts.overfit import overfit
from scripts.param_search import parameter_search
from scripts.regularize import regularize
from scripts.fit_material import fit_material_model, test_material_model
from scripts.error_heat_map import plot_error_heat_map

MODELFILE = "mymodel.torch"

LEARNING_RATES = [.1, .01, .001, .0001, .00001]
ADJUST_LEARNING_RATES = [.1, .3, .5, .7 , 1., 2., 3., 4.]
#HIDDEN_DIMENSIONS = [80, 160, 320, 640, 1280]
HIDDEN_DIMENSIONS = [8, 16, 32, 64, 128, 256]

EPOCHS = 2000
PATIENCE = 200

SNR = 1.

DATA_HANDLER = DataHandler(snr=SNR)

LAMBDAS = [9., 5., 1., .5, .1, 0.]

MODEL = MLP()

MATERIAL = HyperelasticMaterial()
MATERIAL_DATA_HANDLER = MaterialDataHandler(material=MATERIAL,
                                       deformation_function='incompressible',
                                       max_body_scale=1, batchsize=16, samples=1,
                                       body_resolution=10)

if __name__ == '__main__':
    overfitted_model = overfit(data_handler=DATA_HANDLER, epochs=EPOCHS)
    eval_and_plot(overfitted_model, DATA_HANDLER, plot_title='Overfitted Model')

    regularized_model, param_search = parameter_search(initial_model=MODEL, data_handler=DATA_HANDLER, epochs=EPOCHS,
                                     learning_rates=LEARNING_RATES,
                                     adjust_learning_rates=ADJUST_LEARNING_RATES,
                                     hidden_dimensions=HIDDEN_DIMENSIONS, patience=PATIENCE, verbosity=1)    
    eval_and_plot(regularized_model, DATA_HANDLER, plot_title='Regularized Model')
    regularized_model.save(MODELFILE)

    #material_model, data_handler = fit_material_model(epochs=250, patience=10)
    #test_material_model(material_model, data_handler)
    #material_model.save('material_'+MODELFILE)


# %%
from scripts.error_heat_map import plot_error_heat_map
if __name__ == '__main__':
    metrics_overfitted = eval_and_plot(overfitted_model, DATA_HANDLER, plot_title='Overfitted Model')

    metrics_regularized = eval_and_plot(regularized_model, DATA_HANDLER, plot_title='Regularized Model')
    plot_error_heat_map(param_search, 
                        plot_title='Parameter Search Regularized Model',
                        xlabel='Hidden Dimension',
                        ylabel='Learning Rate')    

# %%
# compare the metrics_overfitted vs metrics_initial in a bar plot
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111)
ax.bar(metrics_overfitted.keys(), metrics_overfitted.values(), label='Overfitted Model')
ax.bar(metrics_regularized.keys(), metrics_regularized.values(), label='Regularized Model')
ax.grid()
ax.legend()
ax.set_title('Overfitted vs Regularized Model')
# %%
