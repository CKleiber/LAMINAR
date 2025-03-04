import torch
#import numpy as np

#from typing import Union
from scipy.spatial import KDTree
from scipy.sparse.csgraph import dijkstra
#from scipy.stats import shapiro, combine_pvalues    
#from pingouin import multivariate_normality
#from tqdm import tqdm
#from LAMINAR.Flow.planarCNF import PlanarCNF, train_PlanarCNF
from LAMINAR.Flow.OTFlow import Phi, train_OTFlow, integrate
from LAMINAR.utils.gaussian2uniform import gaussian_to_sphere
from LAMINAR.utils.geodesics import geodesic_length, geodesic_path, geodesic_straight_line

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
                 k_neigh = 10,
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
        self.k_neigh = k_neigh 
        self.epochs = epochs

        self.d = self.data.shape[1]
        self.n = self.data.shape[0]

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


    def switch_device(self, device):
        self.device = device
        self.net.to(device)
    

    def set_up_graph(self):
        self.X_pushed = integrate(self.data, self.net, [0, 1], nt=self.nt, stepper="rk4", alph=self.alph, intermediates=False).cpu().detach()[:, :self.d]
        self.X_pushed = gaussian_to_sphere(self.X_pushed)

        kdt = KDTree(self.X_pushed)
        dists, neighs = kdt.query(self.X_pushed, k=self.k_neigh)

        # get all start points
        starts = self.data[neighs[:, 0]]
        # repeat every point in starts for k_neigh times so that the same point appears k_neigh times right after each other
        starts = starts.repeat_interleave(self.k_neigh, dim=0)

        ends = self.data[neighs.flatten()]

        distances = geodesic_straight_line(starts, ends, self.net.metric_tensor, inbetween=1)

        row_indices = torch.arange(neighs.shape[0]).repeat_interleave(neighs.shape[1])
        col_indices = neighs.flatten()
        dist_values = distances.flatten()

        self.graph = torch.zeros(self.n, self.n).to(self.device)
        self.graph[row_indices, col_indices] = dist_values

        # graph symmetric, by transposition and insertion of values which are not yet in the graph
        graph_sub_transpose = self.graph - self.graph.t()

        # set positive values to zero
        graph_sub_transpose[graph_sub_transpose > 0] = 0
        self.graph = self.graph - graph_sub_transpose

        self.dist_matrix, self.predecessors = dijkstra(self.graph.detach().cpu().numpy(), return_predecessors=True)
        

    def expand_graph(self, additional_points):
        # additional_points is an array of shape (m, d) of points which temporarily need to be added to the graph
        # returns the expanded graph, the distance matrix and the predecessors

        #additional_points = additional_points.reshape(-1, self.d)
        expanded_data = torch.cat([self.data, additional_points], dim=0)
        additional_points_pushed = integrate(additional_points, self.net, [0, 1], nt=self.nt_val, stepper="rk4", alph=self.alph, intermediates=False).cpu().detach()[:, :self.d]
        additional_points_pushed = gaussian_to_sphere(additional_points_pushed)

        expanded_data_pushed = torch.concatenate([self.X_pushed, additional_points_pushed], dim=0)

        kdt = KDTree(expanded_data_pushed)
        _, neighs = kdt.query(additional_points_pushed, k=self.k_neigh)

        starts = additional_points.repeat_interleave(self.k_neigh, dim=0)
        ends = expanded_data[neighs.flatten()]

        distances = geodesic_straight_line(starts, ends, self.net.metric_tensor, inbetween=1)

        row_indices = torch.arange(neighs.shape[0]).repeat_interleave(neighs.shape[1])
        col_indices = neighs.flatten()

        expanded_graph = torch.zeros(expanded_data.shape[0], expanded_data.shape[0]).to(self.device)
        expanded_graph[:self.n, :self.n] = self.graph
        expanded_graph[row_indices + self.n, col_indices] = distances

        # graph
        expanded_graph_sub_transpose = expanded_graph - expanded_graph.t()
        expanded_graph_sub_transpose[expanded_graph_sub_transpose > 0] = 0
        expanded_graph = expanded_graph - expanded_graph_sub_transpose

        dist_matrix, predecessors = dijkstra(expanded_graph.detach().cpu().numpy(), return_predecessors=True)

        return expanded_graph, dist_matrix, predecessors 


    def check_expansion(self, points):
        # for each point, check wether it is in the data, if so note the indices
        # else add the point to the graph

        # check if the points are in the data
        kdt = KDTree(self.data)
        dists, idx = kdt.query(points, k=1)

        dists = torch.tensor(dists)

        # args where dists is zero
        in_data = torch.where(dists == 0)[0]
        at_indices = idx[in_data]
        at_indices = torch.tensor(at_indices).reshape(-1)

        not_in_data = torch.where(dists != 0)[0]

        # expand graph by not_in_data
        # if not_in_data is not empty
        if not_in_data.shape[0] != 0:
            # expand the graph
            _, dist_matrix, predecessors = self.expand_graph(points[not_in_data])

            # note the indices
            not_in_data_idx = torch.arange(self.n, self.n + not_in_data.shape[0]) #[not_in_data]

            # concat the windices of points in the data
            idx = torch.cat([at_indices, not_in_data_idx])

            # make the distance matrix a tensor and return
            dist_matrix = torch.tensor(dist_matrix)

            return idx, dist_matrix, predecessors
        
        else:
            # no extension needed, just return the indices and the distance matrix
            idx = at_indices
            dist_matrix = torch.tensor(self.dist_matrix)

            return idx, dist_matrix, self.predecessors

    
    def query(self, start, k=None):     # TODO expand to add point besides the data
        # start is an array of shape (m, d)
        # if shape is (d,) reshape to (1, d)
        if start.shape == (self.d,):
            start = start.reshape(1, self.d)

        # calculate the k nearest points and their distance for each start point
        # expand the graph with the new points
        idx_points, dist_matrix, _ = self.check_expansion(start)
        dists = torch.tensor(dist_matrix[idx_points]) # shape (m, n)
        dists, idx = torch.sort(dists, dim=1)
    
        if k is not None:
            return idx[:, :k], dists[:, :k].detach()
        
        else:
            return idx, dists.detach()
    
    def distance_approx(self, start, end, return_path=False):
        # expand graph by end and start points
        start = start.unsqueeze(0) if start.dim() == 1 else start
        end = end.unsqueeze(0) if end.dim() == 1 else end
        all_points = torch.cat([self.data, start, end], dim=0)
        #_, dist_matrix, predecessors = self.expand_graph(torch.cat([start, end], dim=0))

        idx, dist_matrix, predecessors = self.check_expansion(torch.cat([start, end], dim=0))

        start_idx = idx[-2]
        end_idx = idx[-1]    

        if return_path:
            path_idx = [end_idx]
            current = end_idx

            while current != start_idx:
                current = predecessors[start_idx, current]
                path_idx.append(current)

            path_idx = torch.tensor(path_idx).flip(0)
            path = all_points[path_idx]

            #dist = geodesic_length(path.reshape(1, path.shape[0], self.d), start, end, self.net.metric_tensor)
            dist = dist_matrix[start_idx, end_idx]

            return dist, path
        
        else:
            dist = dist_matrix[start_idx, end_idx]

            return dist


    def distance_smooth(self, start, end):
        _, path = self.distance_approx(start, end)
        points, _ = geodesic_path(start, end, self.net.metric_tensor, lr=1e-2, initial_guess=path, max_iter=1000)
        dist = geodesic_length(points[1:-1].reshape(1, points[1:-1].shape[0], self.d), start, end, self.net.metric_tensor)

        return dist[0].detach(), points.detach()
    