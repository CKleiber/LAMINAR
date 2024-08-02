import torch
from torch.special import erfinv

'''
Helper functions for converting points from a gaussian distribution to a uniform distribution
on the unit sphere and vice versa.
'''

def gaussian_to_sphere(X):
    # calculate norm of each datapoint
    norms = torch.norm(X, dim=1)

    # calculate the cdf of the norm
    cdf = ((0.5 * (1 + torch.erf(norms / torch.sqrt(torch.tensor(2.0)))) - 0.5) * 2)
    # this cdf is the new radius of the point, the new point is within the unit sphere
    # NOTE: radius can only be between 0 and 1, so the cdf has to be adjusted to be within this range

    # calculate the new point
    X_sphere = X / norms[:, None] * cdf[:, None]

    return X_sphere


def sphere_to_gaussian(X_sphere):
    # invert the function above to give the original gaussian point
    # calculate the norm of each point
    norms = torch.norm(X_sphere, dim=1)

    # calculate the inverse cdf of the norm
    inv_cdf = erfinv((norms/2 + 0.5)*2 - 1) * torch.sqrt(torch.tensor(2.0))
    # this inv_cdf is the original radius of the point

    # calculate the original point
    X = X_sphere / norms[:, None] * inv_cdf[:, None]

    return X
