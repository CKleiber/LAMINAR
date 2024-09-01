import torch
from scipy.special import gamma, gammainc, gammaincinv

'''
Helper functions for converting points from a gaussian distribution to a uniform distribution
on the unit sphere and vice versa.
'''

def gaussian_to_sphere(X: torch.Tensor) -> torch.Tensor:
    # Convert a multivariate gaussian of any dimension d to a d-dimensional sphere
    d = X.shape[1]
    # Compute the norm of each row
    norm = torch.norm(X, dim=1, keepdim=True)

    # compute cdf of each point
    cdf = gammainc(d/2, norm**2/2) ** (1/d)

    # calculate the new point with the adjusted radius
    X_sphere = X / norm * cdf
    return X_sphere

def sphere_to_gaussian(X: torch.Tensor) -> torch.Tensor:
    # Convert a d-dimensional sphere to a multivariate gaussian of any dimension d
    d = X.shape[1]
    # Compute the norm of each row
    norm = torch.norm(X, dim=1, keepdim=True)

    # check if any norm is 1 and set to 0.9999 to avoid infinities
    if torch.any(norm >= 1):
        norm_new = torch.where(norm >= 1, torch.tensor([0.9999]), norm)

        X = X / norm * norm_new
        norm = norm_new

    # compute cdf of each point
    inv_cdf = (gammaincinv(d/2, (norm ** d))*2)**0.5

    # calculate the new point with the adjusted radius
    X_gaussian = X / norm * inv_cdf
    return X_gaussian


def jacobian_gaussian_to_sphere(X: torch.Tensor) -> torch.Tensor: #at point x
    # Compute the jacobian of the transformation from a multivariate gaussian to a d-dimensional sphere
    d = X.shape[1]
    norm = torch.norm(X, dim=1, keepdim=True)[0]

    gammainc_d2 = torch.tensor(gammainc(d/2, norm.cpu().numpy()**2/2))
    gamma_d2 = torch.tensor(gamma(d/2))

    J = torch.zeros((d, d))

    for i in range(d):
        for j in range(d):
            J[i,j] = X[0, i] * X[0, j] * norm**(d-3) * torch.exp(-norm**2/2)/gamma_d2 * gammainc_d2**(1/d - 1) * 1/d - X[0, i] * X[0, j] * norm**(-3) * gammainc_d2**(1/d)

            if i == j:
                J[i,j] = J[i,j] + norm**-1 * gammainc_d2**(1/d)

    return J