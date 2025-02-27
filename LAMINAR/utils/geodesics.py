import torch
from torch.func import vmap


def christoffel_symbol(x, metric_func, eps=1e-6, numeric_diff = True):
    '''
    Calculate the christoffel symbols at the locations x (shape: n, dim) for the metric given by metric_func
    '''

    device = x.device
    dim = x.shape[1] # dimension of the space
    n = x.shape[0] # number of points

    # make input (shape: n, dim+1, dim) for metric call, which calls for every x and a small deviation in every direction for every x
    if numeric_diff:
        x_in = x.unsqueeze(1).repeat(1, 2*dim+1, 1)
        # add and subtract small deviation in every direction
        x_in[:, 1:dim+1] += torch.eye(dim, device=device).unsqueeze(0) * eps
        x_in[:, dim+1:] -= torch.eye(dim, device=device).unsqueeze(0) * eps

        # calculate metric
        #print(x_in.shape)
        g = metric_func(x_in)

        # get gradients of g wrt x (shape: n, dim, dim, dim)
        g_grad1 = (g[:, 1:dim+1] - g[:, 0].repeat(1, dim, 1, 1).reshape(n, dim, dim, dim)) / eps
        g_grad2 = (g[:, dim+1:] - g[:, 0].repeat(1, dim, 1, 1).reshape(n, dim, dim, dim)) / -eps

        g_grad = (g_grad1 + g_grad2) / 2

        g = g[:, 0] # only keep the metric at the point

    else:
        x_in = x
        g = metric_func(x_in)

        jacobian_func = torch.func.jacrev(metric_func, argnums=0)
        batched_jacobian_func = vmap(jacobian_func, in_dims=0)
        g_grad = batched_jacobian_func(x).permute(0, 3, 1, 2)

        #print(g_grad.shape)
        
    inv_g = torch.inverse(g)

    # calculate christoffel symbols

    term1 = g_grad.permute(0, 3, 1, 2)
    term2 = g_grad.permute(0, 3, 2, 1)
    term3 = g_grad.permute(0, 1, 2, 3)

    sum_term = (term1 + term2 - term3) / 2

    christoffel = torch.einsum('nij,njkl->nikl', inv_g, sum_term)

    return christoffel

def geodesic_equation(path, metric_func, eps=1e-6):

    # velocity at each point
    v = (path[2:] - path[:-2])/2 # shape: n-2, dim

    # acceleration at each point
    a = path[2:] - 2*path[1:-1] + path[:-2] # shape: n-2, dim

    # calculate christoffel symbols at each point
    christoffel = christoffel_symbol(path[1:-1], metric_func, eps=eps, numeric_diff=True) # shape: n-2, dim, dim, dim

    # calculate the geodesic equation
    delta_mu = torch.einsum('ndij,ni,nj->nd', christoffel, v, v) # shape: n-2, dim
    delta_mu = a + delta_mu

    tot_deviation = torch.sum(torch.sqrt(torch.sum(delta_mu**2, dim=1)))

    return tot_deviation


def geodesic_length(points, start, end, metric_func):
    # points is a tesor of nex-by-steps-by-d
    # allow calculations of multiple paths with points as intermediate points, while start and end are fixed
    n_paths = points.shape[0]

    # concat start, points, end
    paths = torch.cat([start.repeat(n_paths, 1, 1), points, end.repeat(n_paths, 1, 1)], dim=1)

    # calculate delta of points
    delta_x = paths[:, 1:] - paths[:, :-1]

    g = metric_func((paths[:, 1:]+paths[:, :-1])/2)

    ds_squared = torch.einsum('abi,abij,abj->ab', delta_x, g, delta_x)

    total_length = torch.sqrt(ds_squared).sum(dim=1)

    variance = torch.var(torch.sqrt(ds_squared))
    return total_length, variance


def geodesic_straight_line(starts, ends, metric_func, inbetween = 10):
    # starts shape (n, d)
    # ends shape (n, d)

    n = starts.shape[0]
    d = starts.shape[1]

    # for every instance n, make a straight line between start and end
    points = torch.linspace(0, 1, inbetween+2).view(-1, 1).repeat(n, 1, 1)
    points = torch.einsum('ij,ikj->ikj', (ends - starts), points.repeat(1, 1, d).reshape(n, inbetween+2, 2)) + starts.unsqueeze(1)

    delta_x = points[:, 1:] - points[:, :-1]
    g = metric_func((points[:, 1:] + points[:, :-1]) / 2)

    ds_squared = torch.einsum('abi,abij,abj->ab', delta_x, g, delta_x)
    total_length = torch.sqrt(ds_squared).sum(dim=1)

    return total_length


def geodesic_path(start, end, metric_func, inbetween = 8, lr = 1e-2, initial_guess = None, max_iter = 1000):
    device = start.device

    if initial_guess is None:
        points = torch.linspace(0, 1, inbetween+2).to(device).view(-1, 1).to(device) * (end - start) + start
        points = points[1:-1]
    
    else:
        points = initial_guess[1:-1]
        points = points.to(device)

    points.requires_grad = True

    opt = torch.optim.Adam([points], lr=lr)

    loss_hist = []

    best_loss = 9999999
    lr_drop_count = 0
    best_points = points

    for _ in range(max_iter):
        opt.zero_grad()

        path = torch.concatenate([start.reshape(1, 2), points, end.reshape(1, 2)], dim=0)
        loss = geodesic_equation(path, metric_func)

        loss.backward()       

        opt.step()

        loss_hist.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_points = points.detach()
            lr_drop_count = 0

        else:
            lr_drop_count += 1

        if lr_drop_count > 50:
            # drop learning rate and reset points to best points
            lr_drop_count = 0
            opt = torch.optim.Adam([points], lr=lr/2)
            points = best_points
            points.requires_grad = True
            
        if _ % 100 == 0:
            print(f"Iteration {_} | Loss: {loss.item()}")
        
    print('Max iterations reached')
    print(f"Final loss: {best_loss}")

    points = torch.concatenate([start.reshape(1, 2), best_points, end.reshape(1, 2)], dim=0)

    return points, loss_hist


def geodesic_path2(start, end, metric_func, inbetween = 8, lr = 1e-2, initial_guess = None, max_iter = 1000):
    device = start.device

    if initial_guess is None:
        points = torch.linspace(0, 1, inbetween+2).to(device).view(-1, 1).to(device) * (end - start) + start
        points = points[1:-1]
    
    else:
        points = initial_guess[1:-1]
        points = points.to(device)

    points.requires_grad = True

    opt = torch.optim.Adam([points], lr=lr)

    loss_hist = []

    best_loss = 9999999
    lr_drop_count = 0
    best_points = points

    for _ in range(max_iter):
        opt.zero_grad()

        path = torch.concatenate([start.reshape(1, 2), points, end.reshape(1, 2)], dim=0)
        loss = geodesic_equation(path, metric_func)

        loss.backward()       

        opt.step()

        loss_hist.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_points = points.detach()
            lr_drop_count = 0

        else:
            lr_drop_count += 1

        if lr_drop_count > 50:
            # drop learning rate and reset points to best points
            lr_drop_count = 0
            opt = torch.optim.Adam([points], lr=lr/2)
            #points = best_points
            #points.requires_grad = True
            
        if _ % 100 == 0:
            print(f"Iteration {_} | Loss: {loss.item()}")
        
    print('Max iterations reached')
    print(f"Final loss: {best_loss}")

    points = torch.concatenate([start.reshape(1, 2), best_points, end.reshape(1, 2)], dim=0)

    return points, loss_hist