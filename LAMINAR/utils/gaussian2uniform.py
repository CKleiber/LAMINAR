import torch
from scipy.special import gamma, gammainc, gammaincinv

'''
Helper functions for converting points from a gaussian distribution to a uniform distribution
on the unit sphere and vice versa.
'''

# vectorized implementation of the regularized lower incomplete gamma function
def vec_gammainc(a, x, num_steps=100):
    """
    Differentiable implementation of the regularized lower incomplete gamma function P(a, x).
    Supports vectorized inputs for `a` and `x`.

    Arguments:
        a: Shape parameter (tensor), must be positive.
        x: Upper limit of integration (tensor), must be non-negative.
        num_steps: Number of steps for numerical integration.
    Returns:
        Regularized lower incomplete gamma function P(a, x).
    """
    device = x.device

    #is a not tensor, make it
    if not torch.is_tensor(a):
        a = torch.tensor(a, dtype=torch.float32, device=device) 

    #assert torch.all(a > 0), "Parameter 'a' must be positive."
    #assert torch.all(x >= 0), "Parameter 'x' must be non-negative."

    # Prepare integration steps
    t = torch.linspace(0, 1, num_steps, device=device, dtype=x.dtype)  # Shape: (num_steps,)
    dt = x.unsqueeze(-1) * t  # Scale t to [0, x], Shape: (..., num_steps)

    # Compute integrand t^(a-1) * exp(-t)
    a_expanded = a.unsqueeze(-1)  # Shape: (..., 1)
    integrand = (dt ** (a_expanded - 1)) * torch.exp(-dt)  # Shape: (..., num_steps)
    #integrand[torch.isnan(integrand)] = 0  # Handle NaNs for edge cases (e.g., 0^(-1)).
    integrand = torch.where(torch.isnan(integrand), torch.zeros_like(integrand), integrand)  # Handle NaNs for edge cases (e.g., 0^(-1)).


    # Numerical integration using the trapezoidal rule
    trapezoid = (integrand[..., 1:] + integrand[..., :-1])  # Shape: (..., num_steps - 1)
    integral = trapezoid.sum(dim=-1) * (x / num_steps)  # Shape: (...)

    # Regularization with the gamma function
    gamma_a = torch.exp(torch.special.gammaln(a))  # Compute Gamma(a), Shape: (...)
    return integral / gamma_a  # Shape: (...)


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

# vecotrized implementation of the jacobian of the transformation from a multivariate gaussian to a d-dimensional sphere
def jacobian_gaussian_to_sphere(X: torch.Tensor) -> torch.Tensor: #at point x
    # Compute the jacobian of the transformation from a multivariate gaussian to a d-dimensional sphere
    n = X.shape[0]
    d = X.shape[1]
    norm = torch.norm(X, dim=1, keepdim=True).to(X.device)

    gammainc_d2 = vec_gammainc(d/2, norm**2/2).to(X.device)
    gamma_d2 = torch.tensor(gamma(d/2), dtype=torch.float32, device=X.device)

    constants = 2**(2-d) / gamma_d2 * 1/d

    nex_constants_1 = norm**(d-3) * torch.exp(-norm**2/2) * gammainc_d2**(1/d - 1)
    nex_constants_2 = norm**(-3) * gammainc_d2**(1/d)

    nex_unity = norm**-1 * gammainc_d2**(1/d)

    base_matrix = torch.einsum('bi,bj->bij', X, X)
    
    term1 = torch.einsum('bi,bij->bij', constants * nex_constants_1, base_matrix)
    term2 = torch.einsum('bi,bij->bij', -nex_constants_2, base_matrix)
    term3 = torch.einsum('bi,bij->bij', nex_unity, torch.eye(d, device=X.device).repeat(n, 1, 1))

    J = term1 + term2 + term3

    return J