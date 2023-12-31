import numpy as np
import matplotlib.pyplot as plt
from tools.settings import DEVICE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

def eval_and_plot(model, data_handler, plot_title=""):
        """
        Evaluates the model and plots the results. Evaluation metrics are RMSE, MSE, MAE, R2, MAPE.
        :param model:
        :param data_handler:
        :param plot_title:
        :return: evaluation metrics
        """
        test_x1_mesh, test_x2_mesh = data_handler.get_mesh()
        data_handler.reset()
        train_x, train_y = next(iter(data_handler.get()))
        test_x, test_y = data_handler.get_test_data()
        net_outputs_test = model(test_x.to(DEVICE)).cpu().detach()
        # train_x = train_x.cpu().detach().numpy()
        # train_y = train_y.cpu().detach().numpy()

        # Calculate the standard metrics RMSE, MSE, MAE, R2, MAPE
        rmse = np.sqrt(mean_squared_error(test_y, net_outputs_test))
        print(f"RMSE: {rmse}")
        mse = mean_squared_error(test_y, net_outputs_test)
        print(f"MSE: {mse}")
        mae = mean_absolute_error(test_y, net_outputs_test)
        print(f"MAE: {mae}")
        r2 = r2_score(test_y, net_outputs_test)
        print(f"R2: {r2}")


        eval_metrics = {"RMSE": rmse, "MSE": mse, "MAE": mae, "R2": r2}

        plt.figure(figsize=(16,9))
        ax = plt.subplot(111, projection='3d')

        ax.plot_surface(test_x1_mesh, test_x2_mesh,
                        test_y.reshape(data_handler.test_samples,data_handler.test_samples),
                        color='w', label="Target", alpha=.3, lw=.5, edgecolor='b')
        
        #ax.scatter(train_x[:,0], train_x[:,1], train_y, marker="^", color="r", label="Target", s=100)
        
        ax.plot_surface(
                test_x1_mesh, test_x2_mesh,
                net_outputs_test.reshape(data_handler.test_samples,data_handler.test_samples),
                edgecolor="w", color="g", alpha=.3, label="Learned", lw=.1
                )
        ## All plotting is done, open the plot window
        plt.xlabel("x")
        plt.ylabel("y")
        ax.set_zlabel("Function value")
        plt.title(plot_title)
        plt.legend()
        plt.tight_layout()
        plt.show()

        return eval_metrics