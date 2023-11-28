#%%

test_x = torch.linspace(self.sample_min*1.1, self.sample_max*1.1, self.sample_dim)
test_x = torch.cartesian_prod(test_x, test_x)
test_y = self.target_function(test_x[:,0], test_x[:,1])