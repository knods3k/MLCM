#%%
from matplotlib import pyplot as plt
import numpy as np
import torch
from tools.data import DataHandler
from tools.hyperparameters import Hyperparameters
from tools.model import MLP

MODELFILE = "mymodel.torch"
params = Hyperparameters()
model = MLP(hyperparams=params)

#%%
data_handler = DataHandler()
test_x_mesh, test_y_mesh = data_handler.get_mesh()

#%%
model.start_training(data_handler, MODELFILE)
model.start_evaluation(data_handler)
#%%
model = torch.load(MODELFILE)


#%%
def eval_and_plot(model):
        plt.figure(figsize=(16,9))
        ax = plt.subplot(111, projection='3d')

        net_outputs_test, targets_test, testlosses = evaluate(model, test_x, test_y)
        ax.plot_surface(test_x1_mesh, test_x2_mesh, sin(test_x1_mesh,test_x2_mesh), color='w', label="Target", alpha=.2, lw=.5, edgecolor='b')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()

        ## Call the network on the training data
        net_outputs_train, targets_train, testlosses = evaluate(model, train_x, train_y)
        targets_train = np.array(targets_train)
        ## First, plot the targets in red, i.e. plot the training data set
        ax.scatter(train_x[:,0], train_x[:,1], targets_train[0], "^", color="r", label="Target", s=100)

        ## Now, plot the output of the NN on the whole test interval in green
        ## This allows us to see how the NN performs for interpolation as well as for extrapolation
        net_outputs_test = np.array(net_outputs_test)
        ax.plot_surface(test_x1_mesh, test_x2_mesh, net_outputs_test.reshape(samples,samples), edgecolor="w", color="g", alpha=.3, label="Learned", lw=.1)

        ## All plotting is done, open the plot window
        plt.show()

eval_and_plot(model)

# %%