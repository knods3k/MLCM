#%%
# Import matplotlib for plotting purposes
import matplotlib
matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt

# Import PyTorch
import torch

# Float (32 bit) or Double (64 bit) precision? Choose!
torch.set_default_dtype(torch.float32)#64)
torch.set_num_threads(4) # Use _maximally_ 4 CPU cores


device = torch.device("cpu")
# Choose a device for major calculations (if there is a special GPU card, you usually want that).
# My GPU is not very performant for double precision, so I stay on CPU
#device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
device = torch.device(device)


X = torch.rand(4000, 2)
X.requires_grad_(True)

def helmholtz_free_energy(X, mu=384.61, lam=576.92, deformation_function = None, plot = True):
    """
    This function computes the Helmholtz free energy Psi for a given displacement field u.  The displacement field u is
    either given by a deformation function or allready known. If the deformation function is None, the displacement
    field u is computed from the reference configuration X. The displacement field u is then used to compute the
    deformation gradient tensor F, the right Cauchy-Green tensor C, the third invariant of the right Cauchy-Green
    tensor I3 (describes the volume change of the body),the Jacobian of the deformation J
    and the Helmholtz free energy Psi.
    :param X: Reference configuration
    :param mu: Shear modulus of the material in Pa
    :param lam: Lame's first parameter in Pa
    :param deformation_function: Function that computes the displacement field u from the reference configuration X
    :param plot: If True, the deformation intensity over the domain is plotted
    :return: The deformation gradient tensor F, the right Cauchy-Green tensor C, the invariants of the right
    Cauchy-Green tensor I3, the Jacobian of the deformation J and the Helmholtz free energy Psi
    """

    if deformation_function is None:
        u = torch.zeros_like(X)
        u[:, 0] = (0.3*X[:, 0] + 0.4*X[:, 1])
        u[:, 1] = (0.5*X[:, 1] + 0.6*X[:, 0])
        x = X + u
    else:
        u = deformation_function(X)
        x = X + u

    # Compute the partial derivatives of uy w.r.t. x and y
    duxdxy = torch.autograd.grad(u[:, 0].unsqueeze(1), X, torch.ones(x.size()[0], 1, device=device), create_graph=True,
                                retain_graph=True)[0]
    # Compute the partial derivatives of uy w.r.t. x and y
    duydxy = torch.autograd.grad(u[:, 1].unsqueeze(1), X, torch.ones(x.size()[0], 1, device=device), create_graph=True,
                                retain_graph=True)[0]
    # Initialize the deformation gradient tensor
    F = torch.zeros(x.size()[0], 2, 2)

    # Fill the deformation gradient tensor with the partial derivatives of the displacement field
    # duxdxy[:, 0] is the partial derivative of ux w.r.t. x
    F[:, 0, 0] = duxdxy[:, 0]
    # duxdxy[:, 1] is the partial derivative of ux w.r.t. y
    F[:, 0, 1] = duxdxy[:, 1]
    # duydxy[:, 0] is the partial derivative of uy w.r.t. x
    F[:, 1, 0] = duydxy[:, 0]
    # duydxy[:, 1] is the partial derivative of uy w.r.t. y
    F[:, 1, 1] = duydxy[:, 1]

    print("deformation gradient tensor F:")
    print(f"Shape of deformation gradient tensor F: {F.shape}")
    print(f"First 5 entries of deformation gradient tensor F:\n {F[:5, :, :]}")

    # Compute the right Cauchy-Green tensor
    C = torch.matmul(F, torch.transpose(F, 1, 2))

    print("Right Cauchy-Green tensor C:")
    print(f"Shape of right Cauchy-Green tensor C: {C.shape}")
    print(f"First 5 entries of right Cauchy-Green tensor C:\n {C[:5, :, :]}")

    # Compute the invariants of the right Cauchy-Green tensor
    I3 = torch.det(C.view(-1, 2, 2)) # describes the volume change of the body

    print("Invariants of the right Cauchy-Green tensor I3:")
    print(f"Shape of invariants of the right Cauchy-Green tensor I3: {I3.shape}")
    print(f"First 5 entries of invariants of the right Cauchy-Green tensor I3:\n {I3[:5]}")

    # Compute the Jacobian of the deformation
    J = torch.sqrt(I3)

    print("Jacobian of the deformation J:")
    print(f"Shape of Jacobian of the deformation J: {J.shape}")
    print(f"First 5 entries of Jacobian of the deformation J: {J[:5]}")

    # Debugging print statements
    #print("Debugging:")
    #print(f"torch.log(J**2): {torch.log(J ** 2)}")
    #print(f"2 * torch.log(J): {2 * torch.log(J)}")
    #print(f"torch.trace(C): {torch.einsum('bii->b', C)}")

    Psi = lam/4*(torch.log(J**2)-1-2*torch.log(J)) + mu/2*(torch.einsum("bii->b", C)-2-2*torch.log(J))

    print("Helmholtz free energy Psi:")
    print(f"Shape of Helmholtz free energy Psi: {Psi.shape}")
    print(f"First 5 entries of Helmholtz free energy Psi: {Psi[:5]}")
    print(f"Mean of Helmholtz free energy Psi: {torch.mean(Psi)}")

    if plot:
        plt.figure(figsize=(5, 10))
        plt.subplot(3,1,1)
        plt.scatter(X[:,0].detach().numpy(), X[:, 1].detach().numpy(), color='k', label = "Reference configuration X")
        plt.scatter(x[:, 0].detach().numpy(), x[:, 1].detach().numpy(), color='c', alpha = 0.1, label = "Deformed configuration x")
        plt.xlim(0, 2.5)
        plt.ylim(0, 2.5)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.gca().set_aspect('equal', adjustable='box')
        # add a legend, the box needs to be outside of the plot
        plt.legend(loc = "upper left")
        # title the first subplot
        plt.title("Reference and deformed configuration with the displacement field")
        plt.tight_layout()

        # F_plot defines the color of the scatter plot
        F_plot = F / torch.max(torch.abs(F))

        # Plot the deformation intensity over the domain
        plt.subplot(3,2,3)
        plt.scatter(x[:, 0].detach().numpy(), x[:, 1].detach().numpy(), c = F_plot[:, 0, 0], label = "duxdx")
        plt.xlim(0, 2.5)
        plt.ylim(0, 2.5)
        plt.ylabel("y")
        # make x axis invisible
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.legend(loc = "upper left")
        plt.gca().set_aspect('equal', adjustable='box')
        # ajust spacing between subplots

        plt.subplot(3,2,4)
        plt.scatter(x[:, 0].detach().numpy(), x[:, 1].detach().numpy(), c = F_plot[:, 1, 1], label = "duydy")
        plt.xlim(0, 2.5)
        plt.ylim(0, 2.5)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.legend(loc = "upper left")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().axes.get_yaxis().set_visible(False)

        plt.subplot(3,2,5)
        plt.scatter(x[:, 0].detach().numpy(), x[:, 1].detach().numpy(), c = F_plot[:, 0, 1], label = "duxdy")
        plt.xlim(0, 2.5)
        plt.ylim(0, 2.5)
        plt.ylabel("y")
        plt.xlabel("x")
        plt.legend(loc = "upper left")
        plt.gca().set_aspect('equal', adjustable='box')

        plt.subplot(3,2,6)
        plt.scatter(x[:, 0].detach().numpy(), x[:, 1].detach().numpy(), c = F_plot[:, 1, 0], label = "duydx")
        plt.xlim(0, 2.5)
        plt.ylim(0, 2.5)
        plt.xlabel("x")
        plt.legend(loc = "upper left")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().axes.get_yaxis().set_visible(False)

        #plt.tight_layout(pad=0.3, w_pad=0.3, h_pad=0.2)
        plt.subplots_adjust(wspace=0.05, hspace=0.2, top=0.95, bottom=0.2, left=0.1, right=0.95)
        plt.show()

    return F, C, I3, J, Psi

# %%
# test the function
_, _, _, _, Psi = helmholtz_free_energy(X, plot = True)

# %%

