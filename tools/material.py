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
        self.I3 = torch.linalg.det(self.C)
        self.J = torch.sqrt(self.I3)
        return self
        

    def get_helmholtz_free_energy(self, X=None, u=None):
        '''
        :param X: Reference locations (N_samplepoints x N_dimensions)
        :param u: Displacements (N_samplepoints x N_dimensions)

        :returns: Helmholtz free energy
        '''
        if X is None:
            X = self.X
        if u is None:
            u = self.u
        if not X.requires_grad:
            raise TypeError(f"X.requires_grad={X.requires_grad}, needs to be True")
        self.set_body_configuration(X)
        self.u = u
        self.set_stresses()
        
        # psi = .25*self.lam*(torch.log((self.J)**2) -1 -(2*torch.log(self.J))) \
        #     + .5 *self.mu *(torch.einsum('ijmm -> ij', self.C) - 2 -(2*torch.log(self.J)))

        psi = (self.mu/2) * (self.I1 - 2) \
                - (self.mu * (torch.log(self.J))) \
                + (self.lam/2)*((torch.log(self.J))**2)
        

        assert torch.all(psi != torch.inf)
        return psi


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
