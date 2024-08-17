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
    if torch.any(norm == 1):
        norm_new = torch.where(norm == 1, torch.tensor([0.9999]), norm)

        X = X / norm * norm_new
        norm = norm_new

    # compute cdf of each point
    inv_cdf = (gammaincinv(d/2, (norm ** d))*2)**0.5

    # calculate the new point with the adjusted radius
    X_gaussian = X / norm * inv_cdf
    return X_gaussian
