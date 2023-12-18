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


MODELFILE = "mymodel.torch"

LEARNING_RATES = [.1, .01, .001, .0001, .00001]
ADJUST_LEARNING_RATES = [.1, .3, .5, .7 , 1., 2., 3., 4.]
HIDDEN_DIMENSIONS = [80, 160, 320, 640, 1280]

EPOCHS = 2000
PATIENCE = 50

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
    eval_and_plot(overfitted_model, DATA_HANDLER, plot_title='Overfitting')

    initial_model = parameter_search(initial_model=MODEL, data_handler=DATA_HANDLER, epochs=EPOCHS,
                                     learning_rates=LEARNING_RATES,
                                     adjust_learning_rates=ADJUST_LEARNING_RATES,
                                     hidden_dimensions=HIDDEN_DIMENSIONS, patience=PATIENCE)
    
    eval_and_plot(initial_model, DATA_HANDLER, plot_title='Initial Model')
    initial_model.save(MODELFILE)

    regularized_model = regularize(initial_model=initial_model,
                                   data_handler=DATA_HANDLER, epochs=EPOCHS,
                                     learning_rates=LEARNING_RATES,
                                     adjust_learning_rates=ADJUST_LEARNING_RATES,
                                     lambdas=LAMBDAS, patience=PATIENCE)
    eval_and_plot(regularized_model, DATA_HANDLER, plot_title='Regularized Model')
    regularized_model.save('regularized_'+MODELFILE)

    material_model, data_handler = fit_material_model(epochs=250, patience=10)
    test_material_model(material_model, data_handler)
    material_model.save('material_'+MODELFILE)


# %%
