import torch
import numpy as np

from typing import Union
from scipy.spatial import KDTree
from scipy.sparse.csgraph import shortest_path, dijkstra
from NFLAM.Flow.planarCNF import PlanarCNF, train_PlanarCNF

# ALSO ADD SAVE FUNCTIONALITY TO THE MODEL
class NFLAM():
    def __init__(self,
                 data: Union[torch.Tensor, np.ndarray],
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        
        self.device = device
        self.d = data.shape[1]  # dimension of the data

        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

        self.flow = PlanarCNF(in_out_dim=data.shape[1], device=self.device)  # allow specification of flow parameters here
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=1e-3)
        self._train(data, optimizer=optimizer)
        self._generate_grid_distance_matrix(data)

    def _train(self,
              data: torch.Tensor,
              optimizer: torch.optim.Optimizer,
              epochs: int = 1,
              batch_size: int = 128,
              patience: int = 5,
              verbose: bool = True
              ):

        self.loss_history = train_PlanarCNF(self.flow, optimizer, data, epochs, batch_size, patience, self.device, verbose)

    def _generate_grid(self,
                       data: torch.Tensor,
                       resolution: int = 50  # pixels in grid per unit of data
                       ):
        dim = data.shape[1]
        # find minimum and maximum values for each dimension
        min_vals = data.min(dim=0).values
        max_vals = data.max(dim=0).values

        # find the range of each dimension
        ranges = max_vals - min_vals
        num_points = [int(resolution * r) for r in ranges]

        # create a multidimensional grid with the respective resolutions
        grid = torch.meshgrid([torch.linspace(min_vals[i], max_vals[i], num_points[i]) for i in range(dim)])

        # flatten the grid
        grid = torch.stack([g.flatten() for g in grid], dim=1)

        return grid

    def _generate_grid_distance_matrix(self,
                                        data: torch.Tensor,
                                        resolution: int = 50,  # pixels in grid per unit of data
                                        k_neighbours: int = 20):  # number of nearest neighbours to consider):
        self.k_neighbours = k_neighbours
        self.grid = self._generate_grid(data, resolution).to(self.device)

        # push grid through the model
        self.grid_pushed = self.flow.transform(self.grid).to(self.device)

        # fing the k nearest neighbours in the original data space
        kdtree = KDTree(self.grid_pushed.cpu().detach().numpy())
        _, self.indices = kdtree.query(self.grid_pushed.cpu().detach().numpy(), k=k_neighbours)

        # get covariance matrices for each point in the original data space
        self.cov_matrices = []

        for i in range(self.grid_pushed.shape[0]):
            neighbours = self.grid[self.indices[i]]
            cov_matrix = np.cov(neighbours.T)
            self.cov_matrices.append(cov_matrix)

        self.cov_matrices = torch.tensor(self.cov_matrices, dtype=torch.float32).to(self.device)

        self.distance_matrix = torch.zeros(self.grid_pushed.shape[0], self.grid_pushed.shape[0]).to(self.device)
        # set everythin to infinity
        self.distance_matrix.fill_(float('inf'))

        for i in range(self.grid_pushed.shape[0]):
            for neighbour_idx in self.indices[i]:
                if self.distance_matrix[i, neighbour_idx] != float('inf'):  # if the distance has already been calculated, skip
                    continue

                common_cov = 0.5 * (self.cov_matrices[i] + self.cov_matrices[neighbour_idx])

                x_i = self.grid_pushed[i]
                x_j = self.grid_pushed[neighbour_idx]

                cov_det = torch.det(common_cov) ** 1/self.d

                mahalanobis_distance = torch.sqrt(cov_det * (x_i - x_j).T @ torch.inverse(common_cov) @ (x_i - x_j))
                
                self.distance_matrix[i, neighbour_idx] = mahalanobis_distance
                self.distance_matrix[neighbour_idx, i] = mahalanobis_distance

    def _update_distance_matrix(self, 
                                x: torch.Tensor,
                                x_neighbours: np.ndarray,
                                y: torch.Tensor,
                                y_neighbours: np.ndarray):
        neighbours = [x_neighbours, y_neighbours]

        updated_grid_pushed = torch.cat([self.grid_pushed, x, y], dim=0)

        updated_distance_matrix = torch.zeros(self.grid_pushed.shape[0] + 2, self.grid_pushed.shape[0] + 2).to(self.device)
        updated_distance_matrix.fill_(float('inf'))

        # fill in the old distance matrix
        updated_distance_matrix[:self.grid_pushed.shape[0], :self.grid_pushed.shape[0]] = self.distance_matrix

        for i in range(updated_distance_matrix.shape[0]-2, updated_distance_matrix.shape[0]):
            for j in neighbours[i-self.distance_matrix.shape[0]][0]:
                print(i, j)
                if updated_distance_matrix[i, j] != float('inf'):  # if the distance has already been calculated, skip
                    continue
                common_cov = 0.5 * (self.cov_matrices[i] + self.cov_matrices[j])

                x_i = updated_grid_pushed[i]
                x_j = updated_grid_pushed[j]

                cov_det = torch.det(common_cov) ** 1/self.d

                mahalanobis_distance = torch.sqrt(cov_det * (x_i - x_j).T @ torch.inverse(common_cov) @ (x_i - x_j))
                
                updated_distance_matrix[i, j] = mahalanobis_distance
                updated_distance_matrix[j, i] = mahalanobis_distance

        return updated_distance_matrix
       
    def distance(self,
                 x: torch.Tensor,
                 y: torch.Tensor):
        # push x and y through the model
        x = self.flow.transform(x)
        y = self.flow.transform(y)

        # find the nearest neighbour in the grid for x and y
        kdtree = KDTree(self.grid_pushed.cpu().detach().numpy())
        _, x_idx = kdtree.query(x.cpu().detach().numpy(), k=self.k_neighbours)
        _, y_idx = kdtree.query(y.cpu().detach().numpy(), k=self.k_neighbours)

        # update the distance matrix with the new points
        updated_distance_matrix = self._update_distance_matrix(x, x_idx, y, y_idx)

        # find the distance of the shortest path between x and y
        dist = shortest_path(updated_distance_matrix.cpu().detach().numpy(), indices=[self.distance_matrix.shape[0]], method='D', directed=False)
        return dist[0, -1]
        
