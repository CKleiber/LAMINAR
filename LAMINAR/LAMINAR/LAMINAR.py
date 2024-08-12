import torch
import numpy as np

from typing import Union
from scipy.spatial import KDTree
from scipy.sparse.csgraph import dijkstra
from scipy.stats import shapiro, normaltest
from LAMINAR.Flow.planarCNF import PlanarCNF, train_PlanarCNF
from LAMINAR.utils.gaussian2uniform import sphere_to_gaussian

'''
Implementation of the LAM algorithm using a normalizing flow to transform the data
'''
class LAMINAR():
    def __init__(self,
                 data: Union[np.ndarray, torch.Tensor],
                 epochs: int = 100,
                 k_neighbours: int = 20,
                 hyperparameters: dict = {},
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        '''
        data: Union[np.ndarray, torch.Tensor]  - input data to be transformed, either array or tensor
        epochs: int                            - number of (max) epochs for the training, early stopping is used
        k_neighbours: int                      - number of neighbours to consider; more neighbours lead to a more global metric but also to a higher computational cost, less neighbours lead to more local metric
        hyperparameters: dict                  - hyperparameters for the flow model
        device: torch.device                   - device on which the model is trained, either GPU if available or CPU
        '''

        self.device = device   
        self.k_neighbours = k_neighbours
        self.dimension = data.shape[1]

        # make data a tensor if it is an array
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

        self.data = data.to(self.device)

        # hyperparameters for the flow model
        self.hidden_dim = hyperparameters.get('hidden_dim', 32)
        self.width = hyperparameters.get('width', 64)
        self.timesteps = hyperparameters.get('timesteps', 50)
        self.learning_rate = hyperparameters.get('learning_rate', 1e-3)
        self.patience = hyperparameters.get('patience', 10)
        self.tolerance = hyperparameters.get('tolerance', 0.01)
        self.batch_size = hyperparameters.get('batch_size', 128)

        # initialize the flow
        self.flow = PlanarCNF(in_out_dim=self.dimension, hidden_dim=self.hidden_dim, width=self.width, device=self.device)
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.learning_rate)
        
        # train the flow on data
        self._train(self.data, optimizer, epochs=epochs, batch_size=self.batch_size, patience=self.patience, tolerance=self.tolerance)
        
        # push the data throught the flow
        self.data_pushed = self.flow.transform(self.data, timesteps=self.timesteps)
        
        # generate the distance matrix of the k closest neighbourhoods
        self._generate_distance_matrix()

    # get p value of the gaussian after the flow
    def p_value(self):
        '''
        Function to calculate the p-value of the pushed data distribution
        '''
        # calculate the p-value of the data distribution

        data = sphere_to_gaussian(self.data_pushed.cpu().detach()).to(self.device)

        _, shapiro_p = shapiro(data.cpu().detach().numpy())
        #_, normaltest_p = normaltest(data.cpu().detach().numpy())

        print(f'Shapiro-Wilk p-value:\t{shapiro_p}')
        #print(f'Normality test p-value:\t{normaltest_p}')

        return shapiro_p #, normaltest_p

    # train function
    def _train(self,
              data: torch.Tensor,
              optimizer: torch.optim.Optimizer,
              epochs: int = 100,
              batch_size: int = 128,
              patience: int = 5,
              tolerance: float = 0.01,
              verbose: bool = True
              ):
        '''
        data: torch.Tensor                  - input data to be transformed
        optimizer: torch.optim.Optimizer    - optimizer for the training process
        epochs: int                         - number of epochs for the training
        batch_size: int                     - batch size for the training
        patience: int                       - early stopping patience
        verbose: bool                       - verbosity of the training process
        '''
        self.loss_history = train_PlanarCNF(self.flow, optimizer, data, epochs, batch_size, patience, tolerance, self.device, verbose)    

    # function to generate the distance matrix for the neighbourhoods
    def _generate_distance_matrix(self):
        # get the neighbours of each point
        indices = self._internal_query(range(self.data.shape[0]))
        
        # calculate the covariance of each points neighbours
        cov_matrices = np.zeros((self.data.shape[0], self.dimension, self.dimension))
        for i in range(self.data.shape[0]):
            cov_matrices[i] = np.cov(self.data[indices[i]].cpu().detach().numpy().T)

        self.cov_matrices = torch.tensor(cov_matrices, dtype=torch.float32).to(self.device)

        # initialize the matrix with infinity
        self.distance_matrix = torch.zeros((self.data.shape[0], self.data.shape[0])).to(self.device)
        self.distance_matrix.fill_(float('inf'))

        for i in range(self.data.shape[0]):
            for j in indices[i]:
                if self.distance_matrix[i, j] == float('inf'):
                    # calculate the mahalanobis distance between the points
                    common_cov = (self.cov_matrices[i] + self.cov_matrices[j])/2 + 1e-6*torch.eye(self.dimension).to(self.device)

                    x_i = self.data[i].reshape(1, -1)
                    x_j = self.data[j].reshape(1, -1)

                    # determinant to keep the rescaling in mind
                    cov_det = torch.det(common_cov) ** 1/self.dimension

                    mahalanobis_distance = torch.sqrt(cov_det * (x_i - x_j) @ torch.inverse(common_cov) @ (x_i - x_j).T)

                    self.distance_matrix[i, j] = mahalanobis_distance
                    self.distance_matrix[j, i] = mahalanobis_distance

    # internal query for the initial k nearest neighbours
    def _internal_query(self,
                        indices: Union[int, np.ndarray],   # indices of the points to query
                        k_neighbours: Union[int, None] = None) -> np.ndarray:
        '''
        indices: np.ndarray            - indices of the points to query
        k_neighbours: Union[int, None] - number of neighbours to consider; None means the default value of the class is used
        '''

        if isinstance(indices, int):
            indices = [indices]
        
        if k_neighbours is None:
            k_neighbours = self.k_neighbours

        points = self.data[indices]

        points = self.flow.transform(points)
        
        # use kdtrees for the neighourhood search        
        kdtree = KDTree(self.data_pushed.cpu().detach().numpy())
        _, indices = kdtree.query(points.cpu().detach().numpy(), k=k_neighbours)

        return indices
    
    # query for the user to get the k nearest neighbours and their distances
    def query(self,
              indices: Union[int, np.ndarray],   # indices of the points to query
              k_neighbours: Union[int, None] = None) -> np.ndarray:
        '''
        indices: np.ndarray            - indices of the points to query
        k_neighbours: Union[int, None] - number of neighbours to consider; None means the default value of the class is used
        '''
        if isinstance(indices, int):
            indices = [indices]

        if k_neighbours is None:
            k_neighbours = self.k_neighbours

        dist = self.distance(indices, return_path=False)
        ind = []
        d = []

        for i in range(len(indices)):
            # sort ind and dist by the distance
            ind.append(np.argsort(dist[i])[:k_neighbours])
            d.append(np.sort(dist[i])[:k_neighbours])

        return ind, d   
    
    # function to calculate the distance between any points
    def distance(self, 
                 x_ind: np.ndarray,
                 y_ind: Union[np.ndarray, None] = None,
                 return_path: bool = False) -> float:
        '''
        x_ind: np.ndarray               - index of the reference point
        y_ind: Union[np.ndarray, None]  - index of the target point(s); None selects all points
        return_path: bool               - return all points on the shortest path
        '''
        dist, path = dijkstra(self.distance_matrix.cpu().detach().numpy(), indices=x_ind, return_predecessors=True)
        if y_ind is None:
            return dist
        
        else:
            dist = dist[y_ind]

            # path reconstruction only if a single target point is given
            if (return_path and isinstance(y_ind, int)) or (return_path and isinstance(y_ind, np.ndarray) and len(y_ind) == 1):
                shortest_path = [y_ind]
                while shortest_path[-1] != x_ind:
                    shortest_path.append(path[shortest_path[-1]])
                return dist, shortest_path[::-1]
            
            else:
                return dist