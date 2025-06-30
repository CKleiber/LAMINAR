import torch
from torch.func import vmap
import os


# christoffel symbols are used to calculate the geodesic equation, which isn't used in the current implementation
def christoffel_symbol(x, metric_func, eps=1e-6, numeric_diff=True):
    '''
    Calculate the christoffel symbols at the locations x (shape: n, dim) for the metric given by metric_func
    '''

    device = x.device
    dim = x.shape[1] # dimension of the space
    n = x.shape[0] # number of points

    # make input (shape: n, dim+1, dim) for metric call, which calls for every x and a small deviation in every direction for every x
    if numeric_diff:

        # numeric differentiation will become very imprecise if eps is too large or too small. Very small changes in the metric function are not captured well.

        x_in = x.unsqueeze(1).repeat(1, 2*dim+1, 1)

        # add and subtract small deviation in every direction
        x_in[:, 1:dim+1] += torch.eye(dim, device=device).unsqueeze(0) * eps
        x_in[:, dim+1:] -= torch.eye(dim, device=device).unsqueeze(0) * eps

        # calculate metric
        g = metric_func(x_in)

        # get gradients of g wrt x (shape: n, dim, dim, dim)
        g_grad1 = (g[:, 1:dim+1] - g[:, 0].repeat(1, dim, 1, 1).reshape(n, dim, dim, dim)) / eps
        g_grad2 = (g[:, dim+1:] - g[:, 0].repeat(1, dim, 1, 1).reshape(n, dim, dim, dim)) / -eps

        g_grad = (g_grad1 + g_grad2) / 2

        g = g[:, 0] # only keep the metric at the point

    else:

        # using torch.func.jacrev to calculate the jacobian of the metric function works better in many cases, but for some functions which are not supported by jacrev, this will fail.

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


# calculate the total deviation from the geodesic equation. This is not used in the current implementation. Will maybe be imprecise due to the christoffel symbol calculations above.
def geodesic_equation(path, metric_func, eps=1e-6, numeric_diff=True):

    # velocity at each point
    v = (path[2:] - path[:-2])/2 # shape: n-2, dim

    # acceleration at each point
    a = path[2:] - 2*path[1:-1] + path[:-2] # shape: n-2, dim

    # calculate christoffel symbols at each point
    christoffel = christoffel_symbol(path[1:-1], metric_func, eps=eps, numeric_diff=numeric_diff) # shape: n-2, dim, dim, dim

    # calculate the geodesic equation
    delta_mu = torch.einsum('ndij,ni,nj->nd', christoffel, v, v) # shape: n-2, dim
    delta_mu = a + delta_mu

    tot_deviation = torch.sum(torch.sqrt(torch.sum(delta_mu**2, dim=1)))

    return tot_deviation

# Given a start and end point, this function calculates the length fow a batch of paths defined by the points in between.
def geodesic_length(points, start, end, metric_func):
    # points is a tensor of nex-by-steps-by-d
    # allow calculations of multiple paths with points as intermediate points, while start and end are fixed
    n_paths = points.shape[0]

    dim = points.shape[2]

    # concat start, points, end
    paths = torch.cat([start.repeat(n_paths, 1, 1), points, end.repeat(n_paths, 1, 1)], dim=1)

    # calculate delta of points
    delta_x = paths[:, 1:] - paths[:, :-1]

    g = metric_func((paths[:, 1:]+paths[:, :-1])/2)

    ds_squared = torch.einsum('abi,abij,abj->ab', delta_x, g, delta_x)

    #g_det = torch.linalg.det(g)
    #ds_squared = ds_squared * g_det**(-1/dim)

    total_length = torch.sqrt(ds_squared).sum(dim=1)

    variance = torch.var(torch.sqrt(ds_squared))
    return total_length, variance

# this does the same as the function above, just it assumes a euclidean straight line as a path between start and end
def geodesic_straight_line(starts, ends, metric_func, inbetween = 10):
    # starts shape (n, d)
    # ends shape (n, d)

    n = starts.shape[0]
    d = starts.shape[1]
    device = starts.device

    # for every instance n, make a straight line between start and end
    points = torch.linspace(0, 1, inbetween+2).view(-1, 1).repeat(n, 1, 1).to(device)
    points = torch.einsum('ij,ikj->ikj', (ends - starts), points.repeat(1, 1, d).reshape(n, inbetween+2, d)) + starts.unsqueeze(1)

    delta_x = points[:, 1:] - points[:, :-1]
    g = metric_func((points[:, 1:] + points[:, :-1]) / 2)

    ds_squared = torch.einsum('abi,abij,abj->ab', delta_x, g, delta_x)

    #g_det = torch.linalg.det(g)
    #ds_squared = ds_squared * g_det**(-1/d)

    total_length = torch.sqrt(ds_squared).sum(dim=1)

    return total_length


# the action is the quantity used to smoothen the geodesics. 
def action(path, metric_function):
    v = (path[2:] - path[:-2])/2
    v_start = (path[1] - path[0])
    v_end = (path[-1] - path[-2]) 

    v = torch.cat((v_start[None], v, v_end[None]), dim=0)

    g = metric_function(path)

    inner_prod = torch.einsum('ni,nij,nj->n', v, g, v)

    s = torch.sum(inner_prod, dim=0) * 0.5
    s *= path.shape[0]
    return s   


# returns a parameterized function taking stat point, end point and time, and returns the point a a time
# gamma_eta(x, y, t) = (1-t)*x + t*y + t * (1-t) * phi_eta(x, y, t) (compare Gruffaz et. al. 2025)
# phi is a neural network

# NOTE: This can definitely be improved. This is rather a proof of concept than a final implementation.

class geodesic_regression_function(torch.nn.Module):
    def __init__(self, dim, num_hidden=10, num_layers=2):
        super().__init__()

        self.phi = torch.nn.Sequential(
            torch.nn.Linear(2*dim + 1, num_hidden),
            torch.nn.ReLU(),
            *[
                torch.nn.Sequential(
                    torch.nn.Linear(num_hidden, num_hidden),
                    torch.nn.ReLU()
                )
                for _ in range(num_layers-1)
            ],
            torch.nn.Linear(num_hidden, dim)
        )

    def forward(self, x, y, t):
        # x is shape 1, dim
        # y is shape 1, dim
        # t is shape n_steps
        x = x.repeat(t.shape[0], 1)
        y = y.repeat(t.shape[0], 1)
        t = t.reshape(len(t), 1) # shape n_steps, 1

        phi = self.phi(torch.cat([x, y, t], dim=1))
        return ((1-t)*x + t*y + t*(1-t)*phi)
    
    def initial_fit(self, points):
        # get an initial fit to the approximation by using a reconstruction loss
        
        self.x = x = points[0]
        self.y = y = points[-1]

        t = torch.linspace(0, 1, len(points)).reshape(-1, 1)[1:-1]  # shape n_steps, 1

        func_target = points[1:-1]

        x.requires_grad = True
        y.requires_grad = True
        t.requires_grad = True

        optim = torch.optim.Adam(self.parameters(), lr=1e-4)
        loss_fn = torch.nn.MSELoss()

        self.phi.train()

        best_loss = 1e10
        counter = 0

        for i in range(25000): # 25k is random, could be optimised
            optim.zero_grad()
            pred = self.forward(x, y, t)
            loss = loss_fn(pred, func_target) # per point
            loss.backward()
            optim.step()

            current_loss = loss.item()

            if current_loss < best_loss:
                best_loss = current_loss

            elif current_loss > best_loss:
                counter += 1
                if counter > 100:
                    # reduce learning rate to 10% 
                    
                    for param_group in optim.param_groups:
                        if param_group['lr'] > 1e-8:
                            param_group['lr'] *= 0.1     # 10% is random too, could also be optimised
                            print(f'Learning rate reduced to {param_group["lr"]}')
                            counter = 0

                if counter > 100:
                    break

            # if approximation is good enough, break
            if loss.item() < 1.5e-3:
                break

        print(f'Final loss: {current_loss}')
        self.phi.eval()

    def fit_to_geodesic(self, metric_func):
        # use action as loss function to fit the geodesic regression function to the geodesic equation

        t = torch.linspace(0, 1, 100).reshape(-1, 1)
        optim = torch.optim.Adam(self.parameters(), lr=1e-5)

        best_loss = 1e10
        best_phi = None

        self.phi.train()
        
        l_list = []

        n_t = 100
        t = torch.linspace(0, 1, n_t).reshape(-1, 1)

        counter = 0

        for i in range(1000):
            
            optim.zero_grad()
            pred = self.forward(self.x, self.y, t)
    
            loss = action(pred, metric_func) 
            
            current_loss = loss.item()
            
            l_list.append(current_loss)

            if current_loss < best_loss:
                best_loss = current_loss
                torch.save(self.phi.state_dict(), 'best.pt')
            
            loss.backward()
            optim.step()

            if current_loss > best_loss:
                counter += 1
                if counter > 25:
                    # reduce learning rate to 10%
                    
                    for param_group in optim.param_groups:
                        if param_group['lr'] > 1e-8:
                            param_group['lr'] *= 0.1
                            print(f'Learning rate reduced to {param_group["lr"]}')
                            counter = 0
                
                if counter > 25:
                    break
                    
        print(f'Best loss: {best_loss}')

        best_phi = torch.load('best.pt')
        self.phi.load_state_dict(best_phi)
        # delete best.pt file
        os.remove('best.pt')
        self.phi.eval()

        with torch.no_grad():  # Ensure no gradients are computed
            pred = self.forward(self.x, self.y, t)
            final_loss = action(pred, metric_func)
            print(f'Final loss: {final_loss.item()}')

        return l_list