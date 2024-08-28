import torch
import numpy as np

from typing import Union
from scipy.spatial import KDTree
from scipy.sparse.csgraph import dijkstra
from scipy.stats import shapiro, combine_pvalues    
from pingouin import multivariate_normality
from tqdm import tqdm
from LAMINAR.Flow.planarCNF import PlanarCNF, train_PlanarCNF
from LAMINAR.utils.gaussian2uniform import sphere_to_gaussian, jacobian_gaussian_to_sphere


'''
Implementation of the LAM algorithm using a normalizing flow to transform the data
'''
class LAMINAR():
    def __init__(self,
                 data: Union[np.ndarray, torch.Tensor],
                 epochs: int = 100,
                 k_neighbours: int = 20,
                 grid_resolution: int = 10,
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
        self.grid_resolution = grid_resolution

        # make data a tensor if it is an array
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

        self.data = data.to(self.device)

        # hyperparameters for the flow model
        self.hidden_dim = hyperparameters.get('hidden_dim', 32)
        self.width = hyperparameters.get('width', 64)
        self.timesteps = hyperparameters.get('timesteps', 50)
        self.learning_rate = hyperparameters.get('learning_rate', 1e-3)
        self.patience = hyperparameters.get('patience', 50)
        self.sig = hyperparameters.get('sig', 3.0)
        self.batch_size = hyperparameters.get('batch_size', 128)

        # make grid in unit sphere with 100 points in every dimension
        self._make_grid()

        # initialize the flow
        self.flow = PlanarCNF(in_out_dim=self.dimension, hidden_dim=self.hidden_dim, width=self.width, device=self.device)
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.learning_rate)
        
        # train the flow on data
        self._train(self.data, optimizer, epochs=epochs, batch_size=self.batch_size, patience=self.patience, sig=self.sig)
        
        # concat the data and the grid
        self.reference = torch.cat((self.data, self.grid), dim=0)
        # push reference
        self.reference_pushed = self.flow.transform(self.reference, timesteps=self.timesteps)
        self.data_pushed = self.reference_pushed[:self.data.shape[0]]

        # generate distance matrix
        self._generate_distance_matrix()


    def _make_grid(self):
        # make a grid spaning over the entire data range

        # get min and max values of the data in each dimension
        min_values = torch.min(self.data, dim=0)[0]
        max_values = torch.max(self.data, dim=0)[0]

        # get the range of the data
        range_values = max_values - min_values

        # number of grid points in each dimension 
        grid_points = [int(self.grid_resolution * range_value.item()) for range_value in range_values]

        # make uniform grid with grid_points points in each dimension
        grid = torch.meshgrid(*[torch.linspace(min_values[i], max_values[i], grid_points[i]) for i in range(self.dimension)], indexing='ij')
        self.grid = torch.stack(grid, dim=-1).reshape(-1, self.dimension).to(self.device)

    # get p value of the gaussian after the flow
    def p_value(self):
        '''
        Function to calculate the p-value of the pushed data distribution
        '''
        # calculate the p-value of the data distribution

        data = sphere_to_gaussian(self.data_pushed.cpu().detach()).to(self.device)
        try:
            if self.dimension >= 2:
                p = multivariate_normality(data.cpu().detach().numpy())[1]
            else:
                p = shapiro(data.cpu().detach().numpy())[1]        
        except np.linalg.LinAlgError:
            print('Unable to calculate p-value')
            p = 0.0

        print(f'Henze-Zirkler p-value:\t{p}')
    
        return p 

    # train function
    def _train(self,
              data: torch.Tensor,
              optimizer: torch.optim.Optimizer,
              epochs: int = 100,
              batch_size: int = 128,
              patience: int = 50,
              sig: float = 3.0,
              verbose: bool = True
              ):
        '''
        data: torch.Tensor                  - input data to be transformed
        optimizer: torch.optim.Optimizer    - optimizer for the training process
        epochs: int                         - number of epochs for the training
        batch_size: int                     - batch size for the training
        patience: int                       - early stopping patience
        sig: float                          - significance level for the early stopping
        verbose: bool                       - verbosity of the training process
        '''
        self.loss_history = train_PlanarCNF(self.flow, optimizer, data, epochs, batch_size, patience, sig, self.device, verbose)    

    # function to generate the distance matrix for the neighbourhoods
    def _generate_distance_matrix(self):
        # jacobians
        pbar = tqdm(total=self.reference.shape[0], desc='Calculating Jacobians')
        self.jacobians = torch.zeros(self.reference.shape[0], self.dimension, self.dimension).to(self.device)
        for i in range(self.reference.shape[0]):
            self.jacobians[i] = self.jacobian(self.reference[i].reshape(1, -1))
            pbar.update(1)
        self.metric_t = torch.einsum('ijk, ikl -> ijl', self.jacobians, self.jacobians).to(self.device)

        # get neighbours of the reference points
        pbar = tqdm(total=self.reference.shape[0], desc='Calculating Neighbours')
        self.KDTree_reference_pushed = KDTree(self.reference_pushed.cpu().detach().numpy())
        self.reference_pushed_indices = []
        for i in range(self.reference_pushed.shape[0]):
            _, indices = self.KDTree_reference_pushed.query(self.reference_pushed[i].cpu().detach().numpy(), k=self.k_neighbours)
            self.reference_pushed_indices.append(indices)
            pbar.update(1)

        # generate matrix and fill with inf
        self.distance_matrix = torch.zeros((self.reference.shape[0], self.reference.shape[0])).to(self.device)
        self.distance_matrix.fill_(float('inf'))

        # fill non infty values
        pbar = tqdm(total=self.reference.shape[0], desc='Calculating Distances')
        for i in range(self.reference.shape[0]):
            for j in self.reference_pushed_indices[i]:
                if self.distance_matrix[i, j] == float('inf'):
                    # add small value for numerical stability
                    common_metric_t = (self.metric_t[i] + self.metric_t[j])/2 + 1e-6*torch.eye(self.dimension).to(self.device)

                    x_i = self.reference[i].reshape(1, -1)
                    x_j = self.reference[j].reshape(1, -1)

                    met_det = torch.det(common_metric_t) ** 1/self.dimension

                    mahalanobis_distance = torch.sqrt(met_det * (x_i - x_j) @ torch.inverse(common_metric_t) @ (x_i - x_j).T)

                    # symmetry reasons
                    self.distance_matrix[i, j] = mahalanobis_distance
                    self.distance_matrix[j, i] = mahalanobis_distance
            pbar.update(1)
        

    # function to calculate the distance between any points
    def distance(self, 
                 x: Union[int, np.ndarray],
                 y: Union[int, np.ndarray, None] = None,
                 return_path: bool = False) -> float:
        '''
        x_ind: Union[int, np.ndarray]       - index of the reference point
        y_ind: Union[int, np.ndarray, None] - index of the target point(s); None selects all points
        return_path: bool                   - return all points on the shortest path
        '''
        
        # make x and array of shape (1)
        if isinstance(x, int):
            x = np.array([x])
        assert len(x) == 1
        x = x.reshape(1)

        if y is None:
            y = np.arange(self.data.shape[0])
        elif isinstance(y, int):
            y = np.array([y])
        # make y and array of shape (len)
        y = y.reshape(y.shape[0])


        # get the shortest path between the points using dijkstra
        dist, path = dijkstra(self.distance_matrix.cpu().detach().numpy(), indices=x, return_predecessors=True)
        
        return_distances = np.array(dist[0, y])
        
        if return_path and len(y) == 1:
            shortest_path = []
            i = x[0]
            j = y[0]
            while j != i:
                shortest_path.append(j)
                j = path[0, j]

            shortest_path.append(i)
            shortest_path.reverse()

            shortest_path = np.array(shortest_path)
            shortest_path = self.reference[shortest_path]

            return return_distances, shortest_path
        
        else:
            return return_distances

    # function to calculate the jacobian of the flow and the cdf transformation
    def jacobian(self, X: torch.Tensor) -> torch.Tensor:
        '''
        X: torch.Tensor - input data
        '''
        X_transformed = self.flow.transform(X, timesteps=self.timesteps)

        J_flow = self.flow.get_jacobian(X)
        J_gaussian_to_sphere = jacobian_gaussian_to_sphere(X_transformed)

        J = J_flow @ J_gaussian_to_sphere

        return J.detach()
    

    def query(self,
              x: Union[int, np.ndarray],
              k_neighbours: Union[int, None] = None):
        
            if isinstance(x, int):
                x = np.array([x])
            x = x.reshape(1, -1)

            if k_neighbours is None:
                k_neighbours = self.k_neighbours

            neighbours = []
            distances = []

            for i in x[0]:
                dist = self.distance(int(i), return_path=False)
                _, indices = torch.topk(torch.tensor(dist), k=k_neighbours, largest=False)
                neighbours.append(indices.cpu().detach().numpy())
                distances.append(dist[indices])

            return neighbours, distances
