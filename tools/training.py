from tools.settings import DEVICE
from tools.data import DataHandler
import torch
import matplotlib.pyplot as plt


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
            self.best_model = model
    


def start_training(model, data_handler, model_file=None, verbosity=2, patience=float('inf')):
    early_stopping = EarlyStopping(patience=patience)
    restore_best = RestoreBest()
    data_handler.batch_size = model.hyperparams.batch_size
    test_x, test_y = data_handler.get_test_data()
    test_x = test_x.to(DEVICE)
    test_y = test_y.to(DEVICE)
    train_loader = data_handler.get()
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=model.hyperparams.learning_rate)
    avg_losses = torch.zeros(model.hyperparams.epochs)

    for epoch in range(model.hyperparams.epochs):
        model.train()
        avg_loss = 0.

        for x, y in train_loader:
            model.zero_grad()

            out = model(x.to(DEVICE))
            loss = model.hyperparams.criterion(out, y.to(DEVICE))

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

        avg_losses[epoch] = avg_loss / len(train_loader) 
        validation_loss = model.hyperparams.criterion(model(test_x), test_y)
        restore_best(model, validation_loss)
        if early_stopping(validation_loss):
            break
        if epoch % 20 == 0 and verbosity >= 1:
            print(f"Epoch {epoch:6d}/{model.hyperparams.epochs} \t\t Test Loss: \t {validation_loss:.2e} \t",
                                                            end='\r', flush=True)
    
    model = restore_best.best_model

    validation_loss = model.hyperparams.criterion(model(test_x), test_y)

    print(f"\n \t\t \t\t Final Loss: \t {(validation_loss):.2e} \t",
                                                                end='\r', flush=True)



    if verbosity >= 2:
        plt.figure(figsize=(12, 8))
        plt.plot(avg_losses, "-")
        plt.title("Train loss (MSE, reduction=mean, averaged over epoch)")
        plt.xlabel("Epoch")
        plt.ylabel("loss")
        plt.grid(visible=True, which='both', axis='both')
        plt.show()

    if model_file is not None:
        torch.save(model, model_file)
    return model


def start_evaluation(model, data_handler):
    test_x, test_y = data_handler.get_test_data()
    with torch.no_grad():
        model.eval() 
        outputs = [] 
        targets = []
        testlosses = []

        out = model(test_x.to(DEVICE))

        outputs.append(out.cpu().detach().numpy())
        targets.append(test_y.cpu().detach().numpy())
        testlosses.append(model.hyperparams.criterion(out, test_y.to(DEVICE)).item())
    return outputs, targets, testlosses