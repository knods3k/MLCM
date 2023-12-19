#%%
from tools.material import HyperelasticMaterial

import torch
from torch.utils.data import TensorDataset, DataLoader
from tools.settings import MACHINE_EPSILON

SAMPLES = 160
SAMPLE_MIN = -5
SAMPLE_MAX = 5

BATCH_SIZE = 160

MATERIAL = HyperelasticMaterial()
BODY_RESOLUTION = 200
MAX_BODY_SCALE = 3
DEFORMATION_SCALE = 10

def target_function(x, y):
    return x * y

def default_deformation_function(x):
    return x**2


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
                target_function=target_function, snr=0., batchsize=BATCH_SIZE) -> None:
        super(DataHandler, self).__init__()
        self.samples = samples
        self.test_samples = samples//10
        self.sample_min = sample_min
        self.sample_max = sample_max
        self.target_function = target_function
        self.snr = snr
        self.batchsize = batchsize

        self.reset()

    @staticmethod
    def reset():
        torch.random.manual_seed(1)
    
    def __len__(self):
        return self.batchsize
    
    def __getitem__(self, idx):
        return self.get_training_data()


    def noise(self, input_tensor):
        """
        Adds noise to the input tensor.
        :param input_tensor:
        :return: input_tensor + noise (n)
        """
        n = torch.normal(torch.zeros_like(input_tensor),
                         torch.zeros_like(input_tensor)+(
                            (self.snr*(input_tensor.abs().mean())).sqrt() + MACHINE_EPSILON))
        return input_tensor + n


    def get_mesh(self):
        """
        Generates a meshgrid for the test data.
        :return: meshgrid for the test data.
        """
        test_x1 = torch.linspace(self.sample_min*1.1, self.sample_max*1.1, self.test_samples)
        test_x2 = torch.linspace(self.sample_min*1.1, self.sample_max*1.1, self.test_samples)
        return torch.meshgrid(test_x1, test_x2, indexing='ij') 

    def get_training_data(self):
        """
        Generates training data. Independent variables x
        and dependent variables y are generated.
        :return: train_x and train_y.
        """
        sample_span = self.sample_max - self.sample_min
        train_x1 = (sample_span * torch.rand(self.samples)
                    + self.sample_min * torch.ones(self.samples)).unsqueeze(1)
        train_x2 = (sample_span * torch.rand(self.samples)
                    + self.sample_min * torch.ones(self.samples)).unsqueeze(1)
        
        train_x = torch.concat((train_x1, train_x2), dim=1)
        train_y = self.target_function(train_x1, train_x2)
        train_y = self.noise(train_y)

        return train_x, train_y


    def get_test_data(self):
        """
        Generates test data. Independent variables x
        and dependent variables y are generated.
        :return: test_x and test_y.
        """
        test_x1 = torch.linspace(self.sample_min * 1.1, self.sample_max * 1.1, self.test_samples)
        test_x2 = torch.linspace(self.sample_min * 1.1, self.sample_max * 1.1, self.test_samples)
        test_x1_mesh, test_x2_mesh = torch.meshgrid(test_x1, test_x2, indexing='ij')
        test_x1 = test_x1_mesh.flatten().unsqueeze(1)
        test_x2 = test_x2_mesh.flatten().unsqueeze(1)
        test_x = torch.cat((test_x1, test_x2), dim=1)
        test_y = self.target_function(test_x1, test_x2)

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
        return DataLoader(tensor_dataset, shuffle=True,
                          batch_size=self.batchsize, drop_last=False)
    
    def get_data_generator(self):
        yield self.get_training_data()



    def get(self):
        """
        Generates a DataLoader object for the training data.
        :return: self.get_data_loader()
        """
        return self.get_data_loader()
    

class MaterialDataHandler(DataHandler):
    def __init__(self, material=MATERIAL, samples=SAMPLES, body_resolution=BODY_RESOLUTION,
                 max_body_scale=MAX_BODY_SCALE, snr=0, batchsize=BATCH_SIZE):
        super().__init__(samples, snr, batchsize)

        self.material = material
        self.body_resolution = body_resolution
        self.max_body_scale = max_body_scale

        self.normalize()
    
    def normalize(self):
        '''
        Normalizes input and output data.
        
        Stores the normalizing constant for later use.
        '''
        self.normalizing_constant_out = 1.
        self.normalizing_constant_in = 1.
        x, y = self.get_test_data()
        self.normalizing_constant_in = x.max()
        self.normalizing_constant_out = y.max()

    @staticmethod
    def random_incompressible_deformation(x):
        '''
        Generate a random incompressible deformation.
        
        This is done by computing a randomized matrix with determinant, which ensures
        volume remains unchanged, and then multiplying the input by that matrix.
        '''
        dimension = x.shape[-1]
        A = torch.normal(0,DEFORMATION_SCALE,(dimension,dimension))
        A = torch.abs(A)
        A = torch.triu(A)
        A[0,0] /= torch.prod(torch.diag(A))
        return x @ A

    def get_test_data(self):
        """
        Generates training data. Independent variables x
        and dependent variables y are generated.
        :return: train_x and train_y.
        """
        body_scale = int(torch.randint(self.max_body_scale, ())) + 1 

        X = torch.rand((self.samples, self.body_resolution, 2)) * body_scale + MACHINE_EPSILON
        self.material.set_body_configuration(X)
        self.material.deform(self.random_incompressible_deformation)
        train_x = self.material.get_invariant_deviations()
        train_y = self.material.get_helmholtz_free_energy().unsqueeze(-1)

        train_x /= self.normalizing_constant_in
        train_y /= self.normalizing_constant_out

        return train_x, train_y


    def get_training_data(self):
        """
        Generates training data. Independent variables x
        and dependent variables yare generated.
        :return: train_x and train_y.
        """
        train_x, train_y = self.get_test_data()
        train_y = self.noise(train_y)

        return train_x, train_y
    
    def get(self):
        '''
        Generate data by invoking the __getitem__ method.

        Implements infinte data generation.
        '''
        return DataLoader(self, batch_size=self.batchsize)


if __name__ == "__main__":
    mdata = MaterialDataHandler()
    x, y = mdata.get_training_data()
# %%
