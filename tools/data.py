import torch
from torch.utils.data import TensorDataset, DataLoader

SAMPLES = 160
SAMPLE_MIN = -5
SAMPLE_MAX = 5

BATCH_SIZE = 160

def sin(x, y):
    return torch.square(x) + torch.square(y)

class DataHandler():
    """
    DataHandler is a class that handles the data for the network. It can generate training and test data, and it can
    return a DataLoader object that can be used for training the network.
    :param samples: The number of samples to generate in each dimension. The total number of samples will be samples^2.
    :param sample_min: The lower bound of the sample interval.
    :param sample_max: The upper bound of the sample interval.
    :param function: The function to use for generating the data. The function must take two arguments, x and y.
    :param snr: The signal-to-noise ratio of the data. The noise is generated as a normal distribution with mean 0 and
    standard deviation equal to snr*max(function(x,y)).
    :param batchsize: The batch size to use for training the network.
    """
    def __init__(self, samples=SAMPLES, sample_min=SAMPLE_MIN, sample_max=SAMPLE_MAX,\
                function=sin, snr=0., batchsize=BATCH_SIZE) -> None:
        super(DataHandler, self).__init__()
        self.samples = samples
        self.sample_min = sample_min
        self.sample_max = sample_max
        self.function = function
        self.snr = snr
        self.batchsize = batchsize


    def noise(self, input_tensor):
        """
        Adds noise to the input tensor.
        :param input_tensor:
        :return: input_tensor + noise (n)
        """
        n = torch.normal(torch.zeros_like(input_tensor),
                         torch.zeros_like(input_tensor)+(self.snr*torch.max(input_tensor)))
        return input_tensor + n


    def get_mesh(self):
        """
        Generates a meshgrid for the test data.
        :return: meshgrid for the test data.
        """
        test_x1 = torch.linspace(self.sample_min*1.1, self.sample_max*1.1, self.samples)
        test_x2 = torch.linspace(self.sample_min*1.1, self.sample_max*1.1, self.samples)
        return torch.meshgrid(test_x1, test_x2, indexing='ij') 


    def get_training_data(self):
        """
        Generates training data. Independent variables x and dependent variables y are generated.
        :return: train_x and train_y.
        """
        sample_span = SAMPLE_MAX - SAMPLE_MIN
        train_x1 = (sample_span * torch.rand(self.samples) + self.sample_min * torch.ones(self.samples)).unsqueeze(1)
        train_x2 = (sample_span * torch.rand(self.samples) + self.sample_min * torch.ones(self.samples)).unsqueeze(1)
        train_x = torch.concat((train_x1, train_x2), dim=1)
        train_y = sin(train_x1, train_x2)
        train_y = self.noise(train_y)

        return train_x, train_y


    def get_test_data(self):
        """
        Generates test data. Independent variables x and dependent variables y are generated.
        :return: test_x and test_y.
        """
        test_x1 = torch.linspace(self.sample_min*1.1, self.sample_max*1.1, self.samples)
        test_x2 = torch.linspace(self.sample_min*1.1, self.sample_max*1.1, self.samples)
        test_x1_mesh, test_x2_mesh = torch.meshgrid(test_x1, test_x2, indexing='ij')
        test_x1 = test_x1_mesh.flatten().unsqueeze(1)
        test_x2 = test_x2_mesh.flatten().unsqueeze(1)
        test_x = torch.cat((test_x1, test_x2), dim=1)
        test_y = sin(test_x1, test_x2)

        return test_x, test_y


    def get_tensor_dataset(self):
        """
        Generates a TensorDataset object for the training data.
        :return: TensorDataset object for the training data.
        """
        train_x, train_y = self.get_training_data()
        return TensorDataset(train_x, train_y)


    def get_data_loader(self):
        """
        Generates a DataLoader object for the training data.
        :return: DataLoader object for the training data.
        """
        tensor_dataset = self.get_tensor_dataset()
        return DataLoader(tensor_dataset, shuffle=True, batch_size=self.batchsize, drop_last=False)


    def get(self):
        """
        Generates a DataLoader object for the training data.
        :return: self.get_data_loader()
        """
        return self.get_data_loader()