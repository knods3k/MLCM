from tools.settings import DEVICE
import torch
import matplotlib.pyplot as plt
from os import remove

class EarlyStopping():
    def __init__(self, patience=10, delta_min=0) -> None:
        self.patience = patience
        self.delta_min = delta_min
        self.counter = 0
        self.min_validation_loss = float('inf')

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.delta_min):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

class RestoreBest():
    def __init__(self, delta_min=0) -> None:
        self.best_model = None
        self.delta_min = delta_min
        self.min_validation_loss = float('inf')

    def __call__(self, model, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            torch.save(model.state_dict(), 'tmp.torch')
    


def start_training(model, data_handler, model_file=None, verbosity=2, patience=float('inf'), restore=True):
    early_stopping = EarlyStopping(patience=patience)
    if restore:
        restore_best = RestoreBest()
    else:
        restore_best = lambda _,__: None
    data_handler.batch_size = model.hyperparams.batch_size
    test_x, test_y = data_handler.get_training_data()
    test_x = test_x.to(DEVICE)
    test_y = test_y.to(DEVICE)
    train_loader = data_handler.get()
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=model.hyperparams.learning_rate)
    min_train_losses = torch.zeros(model.hyperparams.epochs)
    test_losses = torch.zeros(model.hyperparams.epochs)

    for epoch in range(model.hyperparams.epochs):
        model.train()
        min_train_loss = []

        for x, y in train_loader:
            model.zero_grad()

            out = model(x.to(DEVICE))
            loss = model.hyperparams.criterion(out, y.to(DEVICE))

            loss.backward()
            optimizer.step()

            min_train_loss.append(loss.item())

        min_train_losses[epoch] = min(min_train_loss)

        with torch.no_grad():
            test_loss = model.hyperparams.criterion(model(test_x), test_y).item()
            test_losses[epoch] = test_loss

        restore_best(model, test_loss)
        if early_stopping(test_loss):
            break
        if epoch % 20 == 0 and verbosity >= 2:
            print(f"Epoch {epoch:6d}/{model.hyperparams.epochs} \t\t Test Loss: \t {test_loss:.2e} \t",
                                                            end='\r', flush=True)
    
    if restore:
        model.load_state_dict(torch.load('tmp.torch'))
        remove('tmp.torch')

    test_loss = model.hyperparams.criterion(model(test_x), test_y)

    print(f"\n \t\t \t\t Final Loss: \t {test_loss:.2e} \t",
                                                                end='\r', flush=True)



    if verbosity >= 1:
        plt.figure()
        plt.plot(min_train_losses, "-", label='Training Loss')
        plt.plot(test_losses, ".", label='Test Loss')
        minimum = (torch.ones_like(test_losses)*test_losses.min()).detach().cpu()
        plt.plot(minimum, 'r--', label='Best Test Loss')
        plt.title("Training History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.grid(visible=True, which='both', axis='both', alpha=.66, linestyle='--')
        plt.yscale('log')
        plt.legend()
        plt.show()

    if model_file is not None:
        torch.save(model, model_file)
    return model


def start_evaluation(model, data_handler):
    test_x, test_y = data_handler.get_training_data()
    with torch.no_grad():
        model.eval() 
        out = model(test_x.to(DEVICE))
    return (model.hyperparams.criterion(out, test_y.to(DEVICE)).item()) 