#%%
import torch

LAM = 576.923
MU = 384.614


class HyperelasticMaterial():
    ''' 
    Implements a class for a hyperelastic material with
    Lame parameters 
        lambda = E*v/((1+v)*(1-2*v))
        mu     = E/(2*(1+v))
    '''

    def __init__(self, lam=LAM, mu=MU) -> None:
        self.lam = lam
        self.mu = mu

    def set_body_configuration(self, X):
        self.X = X
        if not self.X.requires_grad:
            self.X.requires_grad = True
        

    def deform(self, deformation_function):
        self.u = deformation_function(self.X) - self.X
        self.set_stresses()
        return self.u

    def set_stresses(self):
        self.x = self.X + self.u
        batch_size, body_size, _ = self.x.size()
        duxdxy = torch.autograd.grad(self.x[:,:,0], self.X,\
                                     grad_outputs=torch.ones((batch_size,body_size)),
                                     retain_graph=True)[0]
        duydxy = torch.autograd.grad(self.x[:,:,1], self.X,\
                                     grad_outputs=torch.ones((batch_size,body_size)),
                                       retain_graph=True)[0]
        self.F = torch.zeros(batch_size, body_size, 2, 2)
        self.F[:, :, 0, 0] = duxdxy[:, :, 0]
        self.F[:, :, 0, 1] = duxdxy[:, :, 1]
        self.F[:, :, 1, 0] = duydxy[:, :, 0]
        self.F[:, :, 1, 1] = duydxy[:, :, 1]
        self.C = torch.einsum('ijmn, ijmo->ijno', self.F, self.F)
        self.I1 = torch.einsum('ijkl,ijkl -> ij', self.F,self.F)
        self.I2 = .5 * ((self.I1**2) - (torch.einsum('ijkl,ijkl -> ij', self.C, self.C)))
        self.I3 = torch.linalg.det(self.C)
        self.J = torch.sqrt(self.I3)
        return self
        

    def get_helmholtz_free_energy(self):
        '''
        :param X: Reference locations (N_samplepoints x N_dimensions)
        :param u: Displacements (N_samplepoints x N_dimensions)

        :returns: Helmholtz free energy
        '''
        self.set_body_configuration(self.X)
        self.set_stresses()
        
        psi = (self.mu/2) * (self.I1 - 2) \
                - (self.mu * (torch.log(self.J))) \
                + (self.lam/2)*((torch.log(self.J))**2)
        

        # assert torch.all(psi != torch.inf) and torch.all(psi >= 0) and torch.all(psi != torch.nan)
        return psi
    
    def get_invariant_deviations(self):
        self.I1_deviation = self.I1 - 2
        self.I2_deviation = self.I2 - 2
        self.I3_deviation = self.I3 - 1
        return torch.stack((self.I1_deviation,
                               self.I2_deviation,
                               self.I3_deviation), axis=-1)


if __name__ == "__main__":
    torch.random.manual_seed(1)
    m = HyperelasticMaterial()
    X = torch.rand((100,1000,2), requires_grad=True)
    m.set_body_configuration(X)
    u = m.deform(lambda x: x)
    energy = m.get_helmholtz_free_energy()
    print(f'# Negative Energy: {len(energy[energy<0].numpy())}')
    print(f"Mean Energy: {energy.mean().numpy()}")
# %%
