import torch
import numpy as np

from typing import Union
from scipy.spatial import KDTree
from scipy.sparse.csgraph import dijkstra
from scipy.stats import shapiro, combine_pvalues    
from pingouin import multivariate_normality
from tqdm import tqdm
from LAMINAR.Flow.planarCNF import PlanarCNF, train_PlanarCNF
from LAMINAR.Flow.OTFlow import Phi, train_OTFlow, integrate
from LAMINAR.utils.gaussian2uniform import sphere_to_gaussian, jacobian_gaussian_to_sphere, gaussian_to_sphere
from LAMINAR.utils.geodesics import geodesic_length, geodesic_path

'''
Implementation of the LAM algorithm using a normalizing flow to transform the data
'''
class LAMINAR():
    def __init__(self,
                 data, 
                 alph = [1.0, 100.0, 5.0],
                 nt = 8, 
                 nt_val = 8,
                 nTh = 3,
                 m = 32,
                 lr = 0.1,
                 drop_freq = 100,
                 lr_drop = 2,
                 k_neigh_frac = 0.05,
                 epochs = 1500):
        
        self.device = data.device
        self.data = data

        self.alph = alph
        self.nt = nt
        self.nt_val = nt_val
        self.nTh = nTh
        self.m = m
        self.lr = lr
        self.drop_freq = drop_freq
        self.lr_drop = lr_drop
        self.k_neigh_frac = k_neigh_frac 
        self.epochs = epochs

        self.d = self.data.shape[1]
        self.n = self.data.shape[0]

        self.k_neigh = int(self.n*self.k_neigh_frac) + 1

        # split the data into training and validation

        self.data_train = self.data[:int(self.n*0.8)]
        self.data_val = self.data[int(self.n*0.8):]

        # initialize the normalizing flow

        self.net = Phi(self.nTh, self.m, self.d, alph=self.alph, device=self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        # train the model
        
        self.loss_hist = train_OTFlow(self.net, self.optimizer, self.data_train, self.data_val, self.epochs, self.nt, self.nt_val, self.drop_freq, self.lr_drop)

        # set up the graph

        self.set_up_graph()

        #...

    def switch_device(self, device):
        self.device = device
        self.net.to(device)

    def set_up_graph(self):
        self.X_pushed = integrate(self.data, self.net, [0, 1], nt=self.nt, stepper="rk4", alph=self.alph, intermediates=False).cpu().detach()[:, :self.d]
        self.X_pushed = gaussian_to_sphere(self.X_pushed)

        kdt = KDTree(self.X_pushed)
        dists, neighs = kdt.query(self.X_pushed, k=self.k_neigh)

        self.graph = torch.zeros(self.n, self.n).to(self.device)

        metric = self.net.fullHessian(self.data, steps=self.nt_val)

        m = (metric[neighs] + metric[neighs][:, :1]) / 2

        distances = torch.einsum('bni,bnij,bnj->bn', (self.data[neighs] - self.data[:, None]), m, (self.data[neighs] - self.data[:, None]))

        row_indices = torch.arange(neighs.shape[0]).repeat_interleave(neighs.shape[1])
        col_indices = neighs.flatten()
        dist_values = distances.flatten()

        self.graph[row_indices, col_indices] = dist_values 

        # graph symmetric, by transposition and insertion of values which are not yet in the graph
        graph_sub_transpose = self.graph - self.graph.t()

        # set positive values to zero
        graph_sub_transpose[graph_sub_transpose > 0] = 0
        self.graph = self.graph - graph_sub_transpose

        self.dist_matrix, self.predecessors = dijkstra(self.graph.detach().cpu().numpy(), return_predecessors=True)

    def query(self, start, k=None):
          # start is an array of shape (m, d)
        # calculate the k nearest points and their distance for each start point
        # data has shape (n, d)

        # find the closest point for each start point
        if start.shape == (self.d,):
            start = start.reshape(1, self.d)

        start_approx = torch.argmin(torch.norm(self.data.unsqueeze(0).expand(start.shape[0], -1, -1) - start.unsqueeze(1), dim=2), dim=1)

        # calculate the distances from each approximation to the actual start point
        start_delta = self.data[start_approx] - start
        start_mean = (self.data[start_approx] + start) / 2

        g = self.net.metric_tensor(start_mean)

        dist = torch.einsum('bi,bij,bj->b', start_delta, g, start_delta) # shape (m,)

        # dijkstra distances
        dists = torch.tensor(self.dist_matrix[start_approx]) # shape (m, n)

        # add dist to dists
        dists = dists + dist.unsqueeze(1)

        # sort distances from small to large
        dists, idx = torch.sort(dists, dim=1)

        ## sort idx of point with respect to distance
        #idx = torch.argsort(dists, dim=1)

        if k is not None:
            return idx[:, :k], dists[:, :k].detach()
        
        return idx, dists.detach()
    

    def distance_approx(self, start, end):
        # push start and end to the sphere
        extreme_points = torch.cat([start, end], dim=0).reshape(2, self.d)

        extreme_points = integrate(extreme_points, self.net, [0, 1], nt=self.nt_val, stepper="rk4", alph=self.alph, intermediates=False).cpu().detach()[:, :self.d]
        extreme_points = gaussian_to_sphere(extreme_points)

        start_pushed = extreme_points[0]
        end_pushed = extreme_points[1]

        # get closest points in x_pushed to start and end
        start_approx = torch.argmin(torch.norm(self.X_pushed - start_pushed, dim=1))
        end_approx = torch.argmin(torch.norm(self.X_pushed - end_pushed, dim=1))

        # get the distance from start to end

        path_idx = [end_approx]

        current = end_approx

        while current != start_approx:
            current = self.predecessors[start_approx, current]
            path_idx.append(current)

        path_idx = torch.tensor(path_idx).flip(0)

        path = self.data[path_idx]

        full_path = torch.concatenate([start.reshape(1, 2), path, end.reshape(1, 2)], dim=0)

        dist = geodesic_length(full_path.reshape(1, full_path.shape[0], self.d), start, end, self.net.metric_tensor)

        return dist, full_path

    def distance_smooth(self, start, end):
        dist, path = self.distance_approx(start, end)

        points, _ = geodesic_path(start, end, self.net.metric_tensor, lr=1e-2, initial_guess=path, max_iter=1000)

        dist = geodesic_length(points[1:-1].reshape(1, points[1:-1].shape[0], self.d), start, end, self.net.metric_tensor)

        return dist, points.detach()
    