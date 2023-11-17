#%%
import torch

LAM = 6
MU = 4


class HyperelasticMaterial():
    ''' 
    Implements a class for a hyperelastic material with
    Lame parameters 
        lambda = E*v/((1+v)*(1-2*v))
        mü     = E/(2*(1+v))
    '''

    def __init__(self, lam=LAM, mu=MU) -> None:
        self.lam = lam
        self.mu = mu

    def get_helmholtz_free_energy(self, X, u):
        '''
        :param X: Reference locations (N_samplepoints x N_dimensions)
        :param u: Displacements (N_samplepoints x N_dimensions)

        :returns: Helmholtz free energy
        '''
        if not X.requires_grad:
            raise TypeError(f"X.requires_grad={X.requires_grad}, needs to be True")
        x = X + u
        F = torch.autograd.grad(x, X, grad_outputs=torch.ones_like(u), retain_graph=True)[0]
        C = F.T.mm(F)
        I3 = torch.linalg.det(C)
        J = torch.sqrt(I3)
        psi = self.lam*(torch.log(J**2)-1-2*torch.log(J))/4+self.mu*(torch.trace(C)-2-2*torch.log(J))/2
        
        return psi

    def displacement(X):
        return X

if __name__ == "__main__":
    torch.random.manual_seed(1)
    m = HyperelasticMaterial()
    X = torch.rand((10,2), requires_grad=True)
    u = X**3
    energy = m.get_helmholtz_free_energy(X, u)
    print(energy)
# %%
