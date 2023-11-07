#%%
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tools.settings import DEVICE as device
from tools.data import DataHandler
from tools.hyperparameters import Hyperparameters
from tools.model import MLP


hidden_dim = 640
input_dim = 2
output_dim = 1

params = Hyperparameters()
model = MLP(params)

epochs = 1500
lr = .001
criterion = nn.MSELoss(reduction='mean')


#%%
data_handler = DataHandler()
training_loader = data_handler.get()
test_x_mesh, test_y_mesh = data_handler.get_mesh()

#%% 
def evaluate(model, test_x, test_y):
    with torch.no_grad():
        model.eval() 
        outputs = [] 
        targets = []
        testlosses = []

        out = model(test_x.to(device))

        outputs.append(out.cpu().detach().numpy())
        targets.append(test_y.cpu().detach().numpy())
        testlosses.append(criterion(out, test_y.to(device)).item())
    return outputs, targets, testlosses


def train(train_loader, learn_rate, epochs):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    avg_losses = torch.zeros(epochs)

    for epoch in range(epochs):
        model.train()
        avg_loss = 0.
        counter = 0

        for x, y in train_loader:
            counter += 1
            model.zero_grad()

            out = model(x.to(device))
            loss = criterion(out, y.to(device))

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            if counter % 20 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {} = {} + {}".format(epoch, counter,
                                                                                                len(train_loader),
                                                                                                avg_loss / counter, loss.item(), 0))
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, epochs, avg_loss / len(train_loader)))
        avg_losses[epoch] = avg_loss / len(train_loader)   

    plt.figure(figsize=(12, 8))
    plt.plot(avg_losses, "x-")
    plt.title("Train loss (MSE, reduction=mean, averaged over epoch)")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.grid(visible=True, which='both', axis='both')
    plt.show()

    torch.save(model, model_file)

    return model   
model = train(training_loader, lr, epochs)
model = torch.load(model_file)


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