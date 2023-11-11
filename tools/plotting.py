import numpy as np
import matplotlib.pyplot as plt
from tools.settings import DEVICE

def eval_and_plot(model, data_handler):
        test_x1_mesh, test_x2_mesh = data_handler.get_mesh()
        train_x, train_y = data_handler.get_training_data()
        test_x, test_y = data_handler.get_test_data()
        net_outputs_test = model(test_x.to(DEVICE)).cpu().detach()
        # train_x = train_x.cpu().detach().numpy()
        # train_y = train_y.cpu().detach().numpy()

        plt.figure(figsize=(16,9))
        ax = plt.subplot(111, projection='3d')

        ax.plot_surface(test_x1_mesh, test_x2_mesh,
                        test_y.reshape(data_handler.samples,data_handler.samples),
                        color='w', label="Target", alpha=.2, lw=.5, edgecolor='b')
        
        ax.scatter(train_x[:,0], train_x[:,1], train_y, marker="^", color="r", label="Target", s=100)
        
        ax.plot_surface(
                test_x1_mesh, test_x2_mesh,
                net_outputs_test.reshape(data_handler.samples,data_handler.samples),
                edgecolor="w", color="g", alpha=.3, label="Learned", lw=.1
                )
        ## All plotting is done, open the plot window
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()