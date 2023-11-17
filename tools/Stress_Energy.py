import torch
import numpy as np

''' Lame parameters 
    lambda = E*v/((1+v)*(1-2*v))
    mü     = E/(2*(1+v))'''
lmbda = 6
mü = 4

def helmholtz_free_energy(X,u):
    '''x: position in deformed configuration 
       J: Jacobi-Determinant of deformation map
       J = det(F)
       C: Right Cauchy Green deformation tensor 
       C = F.T*F  
       I3 = det(C) = det(u.T*u) 
       psi: Helmholtz free energy
       
       '''
    x = X + u
    F = np.gradient(x)
    C = F.T*F
    I3 = np.linalg.det(C)
    J = I3**(1/3)
    psi = lmbda*(log(J**2)-1-2*log(J))/4+mü*(np.trace(C)-2-2*log(J))/2
       
    return psi

'''

def displacement(X):
    
    return 

''''''