import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
import math

from scipy.spatial import KDTree
from scipy.sparse.csgraph import dijkstra


from LAMINAR.utils.gaussian2uniform import gaussian_to_sphere, jacobian_gaussian_to_sphere

from torch.func import vmap


# define activation functions
def antiderivTanh(x):
    return torch.abs(x) + torch.log(1 + torch.exp(-2.0 * torch.abs(x)))

def derivTanh(x):
    return 1 - torch.pow(torch.tanh(x), 2)


# define a ResNet module

class ResNet(nn.Module):
    def __init__(self, d: int, m: int, nTh: int = 2):
        """
            ResNet N portion of Phi
        :param d:   int, dimension of space input (expect inputs to be d+1 for space-time)
        :param m:   int, hidden dimension
        :param nTh: int, number of resNet layers , (number of theta layers)
        """
        super().__init__()
        # catch excepions
        if nTh < 2:
            print("nTh must be an integer >= 2")
            exit(1)

        self.d = d
        self.m = m
        self.nTh = nTh

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(d + 1, m, bias=True)) # opening layer
        self.layers.append(nn.Linear(m,m, bias=True)) # resnet layers
        for i in range(nTh-2):
            self.layers.append(copy.deepcopy(self.layers[1]))
        self.act = antiderivTanh
        self.h = 1.0 / (self.nTh-1) # step size for the ResNet


    def forward(self, x):
        """
            N(s;theta). the forward propogation of the ResNet
        :param x: tensor nex-by-d+1, inputs (+1 is time?)
        :return:  tensor nex-by-m,   outputs
        """

        x = self.act(self.layers[0].forward(x))

        for i in range(1, self.nTh):
            x = x + self.h * self.act(self.layers[i].forward(x))    # this is the resnet structure

        return x
    

def odefun(x, t, net, alph=[1.0,1.0,1.0]):
    """
    neural ODE combining the characteristics and log-determinant (see Eq. (2)), the transport costs (see Eq. (5)), and
    the HJB regularizer (see Eq. (7)).

    d_t  [x ; l ; v ; r] = odefun( [x ; l ; v ; r] , t )

    x - particle position
    l - log determinant
    v - accumulated transport costs (Lagrangian)
    r - accumulates violation of HJB condition along trajectory
    """
    nex, d_extra = x.shape
    d = d_extra - 3

    z = nn.functional.pad(x[:, :d], (0, 1, 0, 0), value=t) # concatenate with the time t

    gradPhi, trH = net.trHess(z)

    dx = -(1.0/alph[0]) * gradPhi[:,0:d]
    dl = -(1.0/alph[0]) * trH.unsqueeze(1)
    dv = 0.5 * torch.sum(torch.pow(dx, 2) , 1 ,keepdims=True)
    dr = torch.abs(-gradPhi[:,-1].unsqueeze(1) + alph[0] * dv) 
    
    return torch.cat((dx,dl,dv,dr), 1).to(x.device)


def stepRK4(odefun, z, Phi, alph, t0, t1):
    """
        Runge-Kutta 4 integration scheme
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param Phi:    Module, the Phi potential function
    :param alph:   list, the 3 alpha values for the OT-Flow Problem
    :param t0:     float, starting time
    :param t1:     float, end time
    :return: tensor nex-by-d+4, features at time t1
    """

    h = t1 - t0 # step size
    z0 = z

    K = h * odefun(z0, t0, Phi, alph=alph)
    z = z0 + (1.0/6.0) * K
    
    K = h * odefun( z0 + 0.5*K , t0+(h/2) , Phi, alph=alph)
    #z += (2.0/6.0) * K
    z = z + (2.0/6.0) * K

    K = h * odefun( z0 + 0.5*K , t0+(h/2) , Phi, alph=alph)
    #z += (2.0/6.0) * K
    z = z + (2.0/6.0) * K

    K = h * odefun( z0 + K , t0+h , Phi, alph=alph)
    #z += (1.0/6.0) * K
    z = z + (1.0/6.0) * K

    return z

def stepRK1(odefun, z, Phi, alph, t0, t1):
    """
        Runge-Kutta 1 / Forward Euler integration scheme.  Added for comparison, but we recommend stepRK4.
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param Phi:    Module, the Phi potential function
    :param alph:   list, the 3 alpha values for the mean field game problem
    :param t0:     float, starting time
    :param t1:     float, end time
    :return: tensor nex-by-d+4, features at time t1
    """
    #z += (t1 - t0) * odefun(z, t0, Phi, alph=alph)
    z = z + (t1 - t0) * odefun(z, t0, Phi, alph=alph)
    return z


def integrate(x, net, tspan , nt, stepper="rk4", alph =[1.0,1.0,1.0], intermediates=False):
    """
        perform the time integration in the d-dimensional space
    :param x:       input data tensor nex-by-d
    :param net:     neural network Phi
    :param tspan:   time range to integrate over, ex. [0.0 , 1.0]
    :param nt:      number of time steps
    :param stepper: string "rk1" or "rk4" Runge-Kutta schemes
    :param alph:    list of length 3, the alpha value multipliers
    :param intermediates: bool, True means save all intermediate time points along trajectories
    :return:
        z - tensor nex-by-d+4, features at time t1
        OR zFull - tensor nex-by-d+3-by-nt+1 , trajectories from time t0 to t1 (when intermediates=True)
    """
    device = x.device
    h = (tspan[1]-tspan[0]) / nt

    # initialize "hidden" vector to propagate with all the additional dimensions for all the ODEs
    z = nn.functional.pad(x, (0, 3, 0, 0), value=tspan[0]).to(device)

    tk = tspan[0]

    if intermediates: # save the intermediate values as well
        #zFull = torch.zeros(*z.shape, nt+1, device=device, dtype=x.dtype) # make tensor of size z.shape[0], z.shape[1], nt
        #zFull[:,:,0] = z
        #zFull = torch.cat((z.unsqueeze(-1), zFull[:,:,1:]), dim=-1)

        zFull = z.clone().reshape(z.shape[0], z.shape[1], 1)

        if stepper == 'rk4':
            for k in range(nt):
                #zFull[:,:,k+1] = stepRK4(odefun, zFull[:,:,k], net, alph, tk, tk+h)
                zFull = torch.cat((zFull, stepRK4(odefun, zFull[:,:,k], net, alph, tk, tk+h).unsqueeze(-1)), dim=-1)
                tk += h
        elif stepper == 'rk1':
            for k in range(nt):
                #zFull[:,:,k+1] = stepRK1(odefun, zFull[:,:,k], net, alph, tk, tk+h)
                zFull = torch.cat((zFull, stepRK4(odefun, zFull[:,:,k], net, alph, tk, tk+h).unsqueeze(-1)), dim=-1)
                tk += h

        return zFull

    else:
        if stepper == 'rk4':
            for k in range(nt):
                z = stepRK4(odefun, z, net, alph, tk, tk+h)
                tk += h
        elif stepper == 'rk1':
            for k in range(nt):
                z = stepRK1(odefun, z, net, alph, tk, tk+h)
                tk += h

        return z

    # return in case of error
    return -1


class Phi(nn.Module):
    def __init__(self, nTh, m, d, r=10, alph=[1.0] * 5, device=torch.device('cpu')):
        """
            neural network approximating Phi (see Eq. (10) in OT paper)

            Phi( x,t ) = w'*ResNet( [x;t]) + 0.5*[x' t] * A'A * [x;t] + b'*[x;t] + c
            A, b, c are parameterising quadratic mechanics, so N is focussing on higher orders

        :param nTh:  int, number of resNet layers , (number of theta layers)
        :param m:    int, hidden dimension
        :param d:    int, dimension of space input (expect inputs to be d+1 for space-time)
        :param r:    int, rank r for the A matrix
        :param alph: list, alpha values / weighted multipliers for the optimization problem
        """

        self.device = device
        super().__init__()

        self.m    = m
        self.nTh  = nTh
        self.d    = d
        self.alph = alph

        r = min(r,d+1)

        self.A  = nn.Parameter(torch.zeros(r, d+1, device=device) , requires_grad=True)
        self.A  = nn.init.xavier_uniform_(self.A)
        self.c  = nn.Linear( d+1  , 1  , bias=True)  # b'*[x;t] + c (bias)
        self.w  = nn.Linear( m    , 1  , bias=False).to(device)

        self.N = ResNet(d, m, nTh).to(device)

        # set initial values
        self.w.weight.data = torch.ones(self.w.weight.data.shape, device=device)   # w = 1
        self.c.weight.data = torch.zeros(self.c.weight.data.shape, device=device)  # b = 0
        self.c.bias.data   = torch.zeros(self.c.bias.data.shape, device=device)    # c = 0

    def switch_device(self, device):
        self.device = device
        self.A = nn.Parameter(self.A.to(device))
        self.c.to(device)
        self.w.to(device)
        self.N.to(device)

    def forward(self, x):
        """ calculating Phi(s, theta)...not used in OT-Flow """

        # force A to be symmetric
        symA = torch.matmul(torch.t(self.A), self.A) # A'A

        return self.w( self.N(x)) + 0.5 * torch.sum( torch.matmul(x , symA) * x , dim=1, keepdims=True) + self.c(x)
    
    def trHess(self, x, justGrad=False):
        """
        compute gradient of Phi wrt x and trace(Hessian of Phi); see Eq. (11) and Eq. (13), respectively
        recomputes the forward propogation portions of Phi

        :param x: input data, torch Tensor nex-by-d
        :param justGrad: boolean, if True only return gradient, if False return (grad, trHess)
        :return: gradient , trace(hessian)    OR    just gradient
        """

        # code in E = eye(d+1,d) as index slicing instead of matrix multiplication
        # assumes specific N.act as the antiderivative of tanh

        N    = self.N
        m    = N.layers[0].weight.shape[0]
        nex  = x.shape[0] # number of examples in the batch
        d    = x.shape[1]-1
        symA = torch.matmul(self.A.t(), self.A)

        u = [] # hold the u_0,u_1,...,u_M for the forward pass
        z = N.nTh*[None] # hold the z_0,z_1,...,z_M for the backward pass
        # preallocate z because we will store in the backward pass and we want the indices to match the paper

        # Forward of ResNet N and fill u
        opening = N.layers[0].forward(x) # K_0 * S + b_0
        u.append(N.act(opening)) # u0
        feat = u[0]

        for i in range(1,N.nTh):
            feat = feat + N.h * N.act(N.layers[i](feat))
            u.append(feat)

        # going to be used more than once
        tanhopen = torch.tanh(opening) # act'( K_0 * S + b_0 )

        # compute gradient and fill z
        for i in range(N.nTh-1,0,-1): # work backwards, placing z_i in appropriate spot
            if i == N.nTh-1:
                term = self.w.weight.t()
            else:
                term = z[i+1]

            # z_i = z_{i+1} + h K_i' diag(...) z_{i+1}
            z[i] = term + N.h * torch.mm( N.layers[i].weight.t() , torch.tanh( N.layers[i].forward(u[i-1]) ).t() * term)

        # z_0 = K_0' diag(...) z_1
        z[0] = torch.mm( N.layers[0].weight.t() , tanhopen.t() * z[1] )
        grad = z[0] + torch.mm(symA, x.t() ) + self.c.weight.t()

        if justGrad:
            return grad.t()
        
        # -----------------
        # trace of Hessian
        #-----------------

        # t_0, the trace of the opening layer
        Kopen = N.layers[0].weight[:,0:d]    # indexed version of Kopen = torch.mm( N.layers[0].weight, E  )
        temp  = derivTanh(opening.t()) * z[1]
        trH  = torch.sum(temp.reshape(m, -1, nex) * torch.pow(Kopen.unsqueeze(2), 2), dim=(0, 1)) # trH = t_0

        # grad_s u_0 ^ T
        temp = tanhopen.t()   # act'( K_0 * S + b_0 )
        Jac  = Kopen.unsqueeze(2) * temp.unsqueeze(1) # K_0' * act'( K_0 * S + b_0 )
        # Jac is shape m by d by nex

        # t_i, trace of the resNet layers
        # KJ is the K_i^T * grad_s u_{i-1}^T
        for i in range(1,N.nTh):
            KJ  = torch.mm(N.layers[i].weight , Jac.reshape(m,-1) )
            KJ  = KJ.reshape(m,-1,nex)
            if i == N.nTh-1:
                term = self.w.weight.t()
            else:
                term = z[i+1]

            temp = N.layers[i].forward(u[i-1]).t() # (K_i * u_{i-1} + b_i)
            t_i = torch.sum(  ( derivTanh(temp) * term ).reshape(m,-1,nex)  *  torch.pow(KJ,2) ,  dim=(0, 1) )
            trH  = trH + N.h * t_i  # add t_i to the accumulate trace
            Jac = Jac + N.h * torch.tanh(temp).reshape(m, -1, nex) * KJ # update Jacobian

        return grad.t(), trH + torch.trace(symA[0:d,0:d])
        # indexed version of: return grad.t() ,  trH + torch.trace( torch.mm( E.t() , torch.mm(  symA , E) ) )
    
    def fullHessian(self, x, steps=1):
        # Calculate Hessian of Phi for different time steps specified by steps
        n_steps = torch.linspace(0, 1, steps + 1).reshape(-1, 1)[:-1].to(self.device)
    
        # Integrate over time in steps
        #alph = [1.0, 100.0, 5.0] # not needed here?
        zFull = integrate(x, self, [0, 1], nt=steps, stepper="rk4", intermediates=True)[:, :self.d, :-1].to(self.device)
    
        # Prepare data for batch processing
        dim = x.shape[1]
        batch_size = x.shape[0]
        time_dim = n_steps.shape[0]
    
        # Create combined batch for input to Hessian calculation
        x_in = torch.cat([
            zFull.transpose(1, 2).reshape(batch_size * time_dim, self.d),
            n_steps.repeat(batch_size, 1)
        ], dim=1).to(self.device)
    
        # x_in requires_grad
        #x_in = x_in.clone().requires_grad_(True)

        # Define function for which to compute the Hessian
        def func(batch_x_in):
            
            return self.forward(batch_x_in.reshape(1, self.d+1)).sum()
    
        # Compute the Hessians function for all arguments but the last
        hess = vmap(torch.func.hessian(func))(x_in)
        hess = hess.reshape(batch_size, time_dim, self.d+1, self.d+1)
    
        # Extract spatial dimensions and time steps
        hess = hess[:, :, :self.d, :self.d]
    
        # Average over time steps
        hess = hess.mean(dim=1).reshape(-1, dim, dim)

        transformed = zFull[:, :dim]

        unif = jacobian_gaussian_to_sphere(transformed[:, :, -1].clone().reshape(-1, dim)) #.reshape(x.shape[0], dim, dim)

        full_hessians = torch.einsum('bij,bjk->bik', unif, hess)
        metric_t = torch.einsum('bji,bjk->bik', full_hessians, full_hessians)

        # add a small unit matrix to ensure numerical stability

        metric_t = metric_t + 1e-6 * torch.eye(dim, device=self.device).unsqueeze(0)

        return metric_t #, hess, unif
    
    def metric_tensor(self, x):
        # flatten x and reshape into original shape
        x_dims = torch.tensor(x.shape)
        x_new_dims = torch.concat([x_dims, x_dims[-1].reshape(-1)])

        #x = x.flatten(0, 1)
        x = x.reshape(-1, x.shape[-1])

        g = self.fullHessian(x, 15)

        g = g.reshape(*x_new_dims)
        return g
    

# define C cost
def C(z):
    """Expected negative log-likelihood; see Eq.(3) in the paper"""
    d = z.shape[1] - 3
    l = z[:, d]  # log-det

    return -(torch.sum(-0.5 * math.log(2 * math.pi) - torch.pow(z[:, 0:d], 2) / 2, 1, keepdim=True) + l.unsqueeze(1))


def OTFlowProblem(x, Phi, tspan, nt, stepper="rk4", alph=[1.0, 1.0, 1.0]):
    """
    Evaluate objective function of OT Flow problem; see Eq. (8) in the paper.

    :param x:       input data tensor nex-by-d
    :param Phi:     neural network
    :param tspan:   time range to integrate over, ex. [0.0 , 1.0]
    :param nt:      number of time steps
    :param stepper: string "rk1" or "rk4" Runge-Kutta schemes
    :param alph:    list of length 3, the alpha value multipliers
    :return:
        Jc - float, objective function value dot(alph,cs)
        cs - list length 5, the five computed costs
    """
    h = (tspan[1] - tspan[0]) / nt

    # initialize "hidden" vector to propagate with all the additional dimensions for all the ODEs
    z = nn.functional.pad(x, (0, 3, 0, 0), value=0)

    tk = tspan[0]

    if stepper == 'rk4':
        for k in range(nt):
            z = stepRK4(odefun, z, Phi, alph, tk, tk + h)
            tk += h
    elif stepper == 'rk1':
        for k in range(nt):
            z = stepRK1(odefun, z, Phi, alph, tk, tk + h)
            tk += h

    # ASSUME all examples are equally weighted
    costL = torch.mean(z[:, -2])
    costC = torch.mean(C(z))
    costR = torch.mean(z[:, -1])

    cs = [costL, costC, costR]

    # return dot(cs, alph)  , cs
    return sum(i[0] * i[1] for i in zip(cs, alph)), cs


def compute_loss(net, x, nt):
    Jc, cs = OTFlowProblem(x, net, [0, 1], nt=nt, stepper="rk4", alph=net.alph)
    return Jc, cs


def train_OTFlow(net, optimizer, X_train, X_val, epochs = 1500, nt=8, nt_val=8, drop_freq=100, lr_drop=2):
    net.train()

    loss_hist = {
        'train': [],
        'val': []
    }

    val_best = 9999999
    epoch_best = 0
    for itr in range(0, epochs):
        optimizer.zero_grad()
        loss, cost = compute_loss(net, X_train, nt)
        loss.backward()
        optimizer.step()
        if (itr % 100 == 99) or (itr == 0):
            net.eval()
            Jc, cs = compute_loss(net, X_train, nt)
            Jcval, csval = compute_loss(net, X_val, nt_val)
            print(f"Iteration {itr} | Train Loss: {Jc.item()} | Validation Loss: {Jcval.item()}")
            if Jcval < val_best:
                val_best = Jcval
                epoch_best = 0
            else:
                epoch_best += 1

            # improve on the early stopping
            if epoch_best >= 5:
                # stop training
                print(f"Early stopping at iteration {itr}")
                # break the loop
                break
            net.train()

            loss_hist['train'].append(Jc.item())
            loss_hist['val'].append(Jcval.item())

        if itr % drop_freq == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / lr_drop

    net.eval()

    return loss_hist

