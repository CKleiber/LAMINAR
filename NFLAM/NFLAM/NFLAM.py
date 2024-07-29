import torch
import numpy as np

from typing import Union
from scipy.spatial import KDTree
from scipy.sparse.csgraph import shortest_path, dijkstra
from NFLAM.Flow.planarCNF import PlanarCNF, train_PlanarCNF


class NFLAM():
    def __init__(self,
                 data: Union[np.ndarray, torch.Tensor],
                 epochs: int = 100,
                 k_neighbours: int = 20,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device   
        self.data = data
        self.k_neighbours = k_neighbours
        self.dimension = data.shape[1]

        if isinstance(data, np.ndarray):
            self.data = torch.tensor(data, dtype=torch.float32).to(device)

        self.flow = PlanarCNF(in_out_dim=self.dimension, device=self.device)
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=1e-3)
        self._train(data, optimizer, epochs=epochs)
        self.data_pushed = self.flow.transform(data)
        self._generate_distance_matrix()
        

    def _train(self,
              data: torch.Tensor,
              optimizer: torch.optim.Optimizer,
              epochs: int = 100,
              batch_size: int = 128,
              patience: int = 5,
              verbose: bool = True
              ):

        self.loss_history = train_PlanarCNF(self.flow, optimizer, data, epochs, batch_size, patience, self.device, verbose)    

    def _generate_distance_matrix(self):
        indices = self.query(range(self.data.shape[0]))
        
        cov_matrices = []
        for i in range(self.data.shape[0]):
            cov_matrices.append(np.cov(self.data[indices[i]].cpu().detach().numpy().T))

        self.cov_matrices = torch.tensor(cov_matrices, dtype=torch.float32).to(self.device)

        self.distance_matrix = torch.zeros((self.data.shape[0], self.data.shape[0])).to(self.device)
        self.distance_matrix.fill_(float('inf'))

        for i in range(self.data.shape[0]):
            for j in indices[i]:
                if self.distance_matrix[i, j] == float('inf'):
                    common_cov = (self.cov_matrices[i] + self.cov_matrices[j])/2 + 1e-6*torch.eye(self.dimension).to(self.device)

                    x_i = self.data[i].reshape(1, -1)
                    x_j = self.data[j].reshape(1, -1)

                    cov_det = torch.det(common_cov) ** 1/self.dimension

                    mahalanobis_distance = torch.sqrt(cov_det * (x_i - x_j) @ torch.inverse(common_cov) @ (x_i - x_j).T)

                    self.distance_matrix[i, j] = mahalanobis_distance
                    self.distance_matrix[j, i] = mahalanobis_distance

    def query(self,
              indices: np.ndarray,   # indices of the points to query
              k_neighbours: Union[int, None] = None) -> np.ndarray:
        
        if k_neighbours is None:
            k_neighbours = self.k_neighbours

        points = self.data[indices]

        points = self.flow.transform(points)
        
        kdtree = KDTree(self.data_pushed.cpu().detach().numpy())
        _, indices = kdtree.query(points.cpu().detach().numpy(), k=k_neighbours)

        return indices
    
    def distance(self, 
                 x_ind: np.ndarray,
                 y_ind: np.ndarray,
                 return_path: bool = False) -> float:
        dist, path = dijkstra(self.distance_matrix.cpu().detach().numpy(), indices=x_ind, return_predecessors=True)
        dist = dist[y_ind]

        if return_path:
            shortest_path = [y_ind]
            while shortest_path[-1] != x_ind:
                shortest_path.append(path[shortest_path[-1]])
            return dist, shortest_path[::-1]
        
        else:
            return dist