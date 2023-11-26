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
        x = self.X + self.u
        self.F = torch.autograd.grad(x, self.X, grad_outputs=torch.ones_like(x),
                                     retain_graph=True)[0]
        self.C = torch.transpose(self.F, -2, -1).bmm(self.F)
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
        
        psi = (.25 * self.lam) * (
                torch.log(self.J**2) - 1 - (2*torch.log(self.J)) + (
                    (.5 * self.mu) * (torch.einsum('jii',self.C) - 2 - (2*torch.log(self.J)))))
        return psi


if __name__ == "__main__":
    torch.random.manual_seed(1)
    m = HyperelasticMaterial()
    X = torch.rand((10,10,2), requires_grad=True)
    m.set_body_configuration(X)
    u = m.deform(torch.square)
    energy = m.get_helmholtz_free_energy()
    print(energy)
# %%
