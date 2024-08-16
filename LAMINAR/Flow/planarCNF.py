import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from torchdiffeq import odeint
from tqdm import tqdm
from scipy.stats import shapiro, combine_pvalues

from pingouin import multivariate_normality

from LAMINAR.utils.gaussian2uniform import gaussian_to_sphere, sphere_to_gaussian

'''
This code was partly taken from Ricky Chen's implementation of the Neural ODEs paper* and modified for our purposes.

GitHub: https://github.com/rtqichen/torchdiffeq/blob/master/examples/cnf.py

*Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2019). Neural Ordinary Differential Equations. arXiv. https://arxiv.org/abs/1806.07366.
'''


# define the hypernetwork, generating the time dependent parameters of the planar CNF
class HyperNetwork(nn.Module):
    def __init__(self,
                 in_out_dim: int = 2,
                 hidden_dim: int = 32,
                 width: int = 64,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        '''
        in_out_dim: int         - dimensionality of the input and output space
        hidden_dim: int         - dimensionality of the hidden layer of the MLP
        width: int              - width of the planar CNF
        device: torch.device    - device on which the model is trained, either GPU if available or CPU
        '''

        super(HyperNetwork, self).__init__()

        self.device = device

        self.blocksize = in_out_dim * width
        self.in_out_dim = in_out_dim
        self.width = width

        # fully conencted MLP with one hidden layer
        self.fc1 = nn.Linear(1, hidden_dim).to(self.device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).to(self.device)
        self.fc3 = nn.Linear(hidden_dim, 3 * self.blocksize + width).to(self.device)

    def forward(self,
                 t: torch.Tensor) -> list:
        '''
        t: torch.Tensor - time parameter for the forward pass
        '''

        # forward pass through the MLP
        parameters = t.reshape(1, 1)
        parameters = torch.tanh(self.fc1(parameters))
        parameters = torch.tanh(self.fc2(parameters))
        parameters = self.fc3(parameters)

        # split the output into the parameters of the planar CNF
        parameters = parameters.reshape(-1)
        W = parameters[:self.blocksize].reshape(self.width, self.in_out_dim, 1).to(self.device)

        U = parameters[self.blocksize:2 * self.blocksize].reshape(self.width, 1, self.in_out_dim).to(self.device)
        G = parameters[2 * self.blocksize:3 * self.blocksize].reshape(self.width, 1, self.in_out_dim).to(self.device)
        U = U * torch.sigmoid(G)

        B = parameters[3 * self.blocksize:].reshape(self.width, 1, 1).to(self.device)

        return [W, B, U]


# function for calculating the trace of the jacobian w.r.t. z
def trace_df_dz(f: nn.Module,
                z: torch.Tensor) -> torch.Tensor:
    '''
    f: nn.Module    - function for which the trace of the jacobian w.r.t. z is calculated
    z: torch.Tensor - input tensor for the function f
    '''
    sum_diag = 0
    for i in range(z.shape[1]):
       sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag


# actual planar flow model
class PlanarCNF(nn.Module):
    def __init__(self,
                 in_out_dim: int = 2,
                 hidden_dim: int = 32,
                 width: int = 64,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        '''
        in_out_dim: int         - dimensionality of the input and output space
        hidden_dim: int         - dimensionality of the hidden layer of the HyperNetwork
        width: int              - width of the planar CNF
        device: torch.device    - device on which the model is trained, either GPU if available or CPU
        '''    
    
        super(PlanarCNF, self).__init__()
        self.device = device
        self.hypernetwork = HyperNetwork(in_out_dim, hidden_dim, width, device)

    # forward pass of the planar CNF
    def forward(self,
                t: torch.Tensor,
                states: torch.Tensor) -> tuple:
        '''
        t: torch.Tensor         - time parameter for the forward pass
        states: torch.Tensor    - input tensor for the forward pass, containing the states z and the log probability of z
        '''

        z = states[0]
        logp_z = states[1]

        batch_size = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            W, B, U = self.hypernetwork(t)

            Z = torch.unsqueeze(z, 0).repeat(W.shape[0], 1, 1)

            h = torch.tanh(torch.matmul(Z, W) + B)
            dz_dt = torch.matmul(h, U).mean(0)

            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batch_size, 1)

        return (dz_dt, dlogp_z_dt)

    # loss function for the planar CNF
    def loss(self,
             z_t0: torch.Tensor,
             logp_diff_t0: torch.Tensor) -> torch.Tensor:
        '''
        z_t0: torch.Tensor          - initial state of the planar CNF
        logp_diff_t0: torch.Tensor  - log probability of the initial state
        '''        
    
        p_z0 = torch.distributions.MultivariateNormal(
            loc=torch.zeros(z_t0.shape[1]).to(self.device),
            covariance_matrix=torch.eye(z_t0.shape[1]).to(self.device))

        logp_x = p_z0.log_prob(z_t0) - logp_diff_t0.view(-1)
        return -logp_x.mean(0)

    # transform data to the latent space and vice versa
    # takes care of time steps and the transformation direction
    def transform(self,
                  data: torch.Tensor,
                  timesteps: int = 50,
                  reverse: bool = False) -> torch.Tensor:
        '''
        data: torch.Tensor  - input data to be transformed
        timesteps: int      - number of time steps for the transformation
        reverse: bool       - direction of the transformation, either data->latent space or latent space->data
        '''
        self.eval()

        if not reverse:  # transform data to gaussian -> sphere
            with torch.no_grad():
                z_t, _ = odeint(
                    self,
                    (data, torch.zeros(data.shape[0], 1).to(self.device)),
                    torch.linspace(1, 0, timesteps).type(torch.float32).to(self.device),
                    atol=1e-5, rtol=1e-5,
                    method='dopri5'
                )

                z_t = gaussian_to_sphere(z_t[-1].detach().cpu()).to(self.device)

        else:  # transform sphere->gaussian to data space
            with torch.no_grad():
                z_t, _ = odeint(
                    self,
                    (sphere_to_gaussian(data.detach().cpu()).to(self.device), torch.zeros(data.shape[0], 1).to(self.device)),
                    torch.linspace(0, 1, timesteps).type(torch.float32).to(self.device),
                    atol=1e-5, rtol=1e-5,
                    method='dopri5'
                )

                z_t = z_t[-1]
        return z_t


#class EarlyStopping:
#    def __init__(self, 
#                 patience: int = 15, 
#                 significance: float = 5):
#        
#        self.patience = patience
#        self.significance = significance
#
#    def __call__(self, loss_hist: list) -> bool:
#        if len(loss_hist) <= self.patience:
#            return False
#        
#        vals = torch.tensor(loss_hist[-self.patience-1:-1])
#        vals_mean = vals.mean()
#        vals_std = vals.std()
#
#        if loss_hist[-1] > vals_mean + self.significance * vals_std:
#            return True

class EarlyStopping:
    def __init__(self, patience: int = 5, p_lim: float = 0.05):
        self.patience = patience
        self.p_lim= p_lim

        self.p_hist = []

    def __call__(self, p_val) -> bool:
        self.p_hist.append(p_val)

        if len(self.p_hist) <= self.patience:
            return False
        
        if (torch.tensor(self.p_hist[-self.patience:]) > self.p_lim).all():
            return True

# train function of the planar CNF    
def train_PlanarCNF(
        model: PlanarCNF,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        epochs: int = 100,
        batch_size: int = 128,
        patience: int = 15,  # early stopping patience
        p_lim: float = 0.05, # early stopping significance
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose: bool = True) -> list:
    '''
    model: PlanarCNF                - model to be trained
    optimizer: optim.Optimizer      - optimizer for the training process
    train_loader: DataLoader        - data loader for the training data
    epochs: int                     - number of epochs for the training
    batch_size: int                 - batch size for the training
    patience: int                   - early stopping patience
    device: torch.device            - device on which the model is trained, either GPU if available or CPU
    verbose: bool                   - verbosity of the training process
    '''
    
    bad_p_val = False

    loss_history = []

    model.to(device)
    model.train()
    dataloader_train = DataLoader(train_loader, batch_size=batch_size, shuffle=True)

    pbar = tqdm(range(epochs), disable=not verbose)

    earlystop = EarlyStopping(patience=patience, p_lim=p_lim)
    
    for epoch in pbar:
        total_loss = 0
        for i, batch in enumerate(dataloader_train):
            batch = batch.to(device)
            logp_diff_t1 = torch.zeros(batch.shape[0], 1).to(device)

            optimizer.zero_grad()

            # solve ode
            z_t, logp_diff_t = odeint(
                model, 
                (batch, logp_diff_t1),
                torch.tensor([1, 0]).type(torch.float32).to(device),
                atol=1e-5, rtol=1e-5,
                method='dopri5',
            )

            z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]

            loss = model.loss(z_t0, logp_diff_t0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_history.append(total_loss / len(dataloader_train))

        if not bad_p_val:
            pushed = sphere_to_gaussian(model.transform(train_loader, reverse=False).detach().cpu())
            try:
                p_value = multivariate_normality(pushed.cpu().detach().numpy())[1]
            except np.linalg.LinAlgError:
                print('Unable to calculate p-value - Deactive early stopping with p-value monitoring')
                p_value = 0.0
                bad_p_val = True

        if verbose:
            pbar.set_description(f"Epoch {epoch+1} | Loss: {loss_history[-1]:.4f} | p-value: {p_value:.2E}")

        #if p_value > 0.05:
        #    if verbose:
        #        print(f"Early stopping at epoch {epoch+1}")
        #    break

        # early stopping
        if not bad_p_val:
            if earlystop(p_value):
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

    return loss_history
