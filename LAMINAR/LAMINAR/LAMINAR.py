import torch
import numpy as np

from typing import Union
from scipy.spatial import KDTree
from scipy.sparse.csgraph import dijkstra
from scipy.stats import shapiro, combine_pvalues    
from pingouin import multivariate_normality
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
        self._make_grid(self.grid_resolution)

        # initialize the flow
        self.flow = PlanarCNF(in_out_dim=self.dimension, hidden_dim=self.hidden_dim, width=self.width, device=self.device)
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.learning_rate)
        
        # train the flow on data
        self._train(self.data, optimizer, epochs=epochs, batch_size=self.batch_size, patience=self.patience, sig=self.sig)
        
        # push the data throught the flow
        self.data_pushed = self.flow.transform(self.data, timesteps=self.timesteps)
        
        # pull back of grid
        self.grid = self.flow.transform(self.grid_pushed, timesteps=self.timesteps, reverse=True)
        
        # generate the distance matrix of the grid
        self._generate_distance_matrix()

    def _make_grid(self, grid_resolution: int):
        '''
        grid_resolution: int - resolution of the grid
        '''
        # create a grid from -1 to 1 with grid_resolution points in every dimension
        grid = np.linspace(-1, 1, grid_resolution)
        grid = np.array(np.meshgrid(*[grid for _ in range(self.dimension)])).reshape(self.dimension, -1).T
        grid = torch.tensor(grid, dtype=torch.float32)

        # filter out everything outside the unit sphere
        grid = grid[torch.norm(grid, dim=1) < 1]

        self.grid_pushed = grid.to(self.device).float()

        # get 3**dimension neighbours for each point in the grid
        self.KDTree_grid_pushed = KDTree(self.grid_pushed.cpu().detach().numpy())
        _, self.grid_pushed_indices = self.KDTree_grid_pushed.query(self.grid_pushed.cpu().detach().numpy(), k=3**self.dimension)

        
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
        self.jacobians = np.zeros((self.grid.shape[0], self.dimension, self.dimension))
        for i in range(self.grid.shape[0]):
            J = self.jacobian(self.grid[i].reshape(1, -1))
            self.jacobians[i] = J.cpu().detach().numpy()
        self.jacobians = torch.tensor(self.jacobians, dtype=torch.float32).to(self.device)

        # generate matrix and fill with inf
        self.distance_matrix = torch.zeros((self.grid.shape[0], self.grid.shape[0])).to(self.device)
        self.distance_matrix.fill_(float('inf'))

        # fill non infty values
        for i in range(self.grid.shape[0]):
            for j in self.grid_pushed_indices[i]:
                if self.distance_matrix[i, j] == float('inf'):
                    common_cov = (self.jacobians[i] + self.jacobians[j])/2 + 1e-6*torch.eye(self.dimension).to(self.device)

                    x_i = self.grid[i].reshape(1, -1)
                    x_j = self.grid[j].reshape(1, -1)

                    cov_det = torch.det(common_cov) ** 1/self.dimension

                    mahalanobis_distance = torch.sqrt(cov_det * (x_i - x_j) @ torch.inverse(common_cov) @ (x_i - x_j).T)

                    self.distance_matrix[i, j] = mahalanobis_distance
                    self.distance_matrix[j, i] = mahalanobis_distance
        

    # function to calculate the distance between any points
    def distance(self, 
                 x: np.ndarray,
                 y: Union[np.ndarray, None] = None,
                 return_path: bool = False) -> float:
        '''
        x_ind: np.ndarray               - index of the reference point
        y_ind: Union[np.ndarray, None]  - index of the target point(s); None selects all points
        return_path: bool               - return all points on the shortest path
        '''
        
        # prepare the points
        x = torch.tensor(x, dtype=torch.float32).to(self.device).reshape(1, -1)
        x_pushed = self.flow.transform(x, timesteps=self.timesteps)

        if y is not None:
            y = torch.tensor(y, dtype=torch.float32).to(self.device).reshape(1, -1)
            y_pushed = self.flow.transform(y, timesteps=self.timesteps)
        else:
            y = self.data
            y_pushed = self.data_pushed

        # generate new empty distance matrix with old entries

        new_distance_matrix = torch.zeros((self.grid.shape[0]+y.shape[0]+1, self.grid.shape[0]+y.shape[0]+1)).to(self.device)
        new_distance_matrix.fill_(float('inf'))

        new_distance_matrix[:self.grid.shape[0], :self.grid.shape[0]] = self.distance_matrix

        # for every y get the neighbous in the grid
        y_indices = []
        for i in range(y_pushed.shape[0]):
            _, y_index = self.KDTree_grid_pushed.query(y_pushed[i].cpu().detach().numpy(), k=self.k_neighbours)
            y_indices.append(y_index)

        # get jacobian of every y
        y_jacobians = np.zeros((y_pushed.shape[0], self.dimension, self.dimension))
        for i in range(y_pushed.shape[0]):
            J = self.jacobian(y[i].reshape(1, -1))
            y_jacobians[i] = J.cpu().detach().numpy()
        y_jacobians = torch.tensor(y_jacobians, dtype=torch.float32).to(self.device)

        # expand the entries of the new distance matrix by directed edges from the grid to the y points
        for i in range(y_pushed.shape[0]):
            for j in y_indices[i]:
                common_cov = (self.jacobians[j] + y_jacobians[i])/2 + 1e-6*torch.eye(self.dimension).to(self.device)

                x_i = self.grid[j].reshape(1, -1)
                x_j = y[i].reshape(1, -1)

                cov_det = torch.det(common_cov) ** 1/self.dimension

                mahalanobis_distance = torch.sqrt(cov_det * (x_i - x_j) @ torch.inverse(common_cov) @ (x_i - x_j).T)

                new_distance_matrix[j, self.grid.shape[0]+i] = mahalanobis_distance

        # concat the y_pushed and grid pushed and find the nearest neighbours of pushed x
        all = torch.cat((self.grid, y), dim=0)
        all_pushed = torch.cat((self.grid_pushed, y_pushed), dim=0) 
        KDTree_all_pushed = KDTree(all_pushed.cpu().detach().numpy())
        _, x_indices = KDTree_all_pushed.query(x_pushed[0].cpu().detach().numpy(), k=self.k_neighbours)

        # if all inidices are not in the grid, increase the numbers of neighbours until at least 1 neighbour is in the grid
        while len(set(x_indices).intersection(set(range(self.grid.shape[0]))) ) == 0:
            _, x_indices = KDTree_all_pushed.query(x_pushed[0].cpu().detach().numpy(), k=self.k_neighbours+1)
            self.k_neighbours += 1

        # combine the jacobians
        all_jacobians = torch.cat((self.jacobians, y_jacobians), dim=0)

        # get jacobian of x
        x_jacobian = self.jacobian(x)

        # expand the entries of the new distance matrix by directed edges from the x point to the grid
        for j in x_indices:
            common_cov = (all_jacobians[j] + x_jacobian)/2 + 1e-6*torch.eye(self.dimension).to(self.device)

            x_i = x.reshape(1, -1)
            x_j = all[j].reshape(1, -1)

            cov_det = torch.det(common_cov) ** 1/self.dimension

            mahalanobis_distance = torch.sqrt(cov_det * (x_i - x_j) @ torch.inverse(common_cov) @ (x_i - x_j).T)

            # dijkstra does not work with 0 distances, so set it to a small value
            if mahalanobis_distance == 0:       
                mahalanobis_distance = 1e-6

            new_distance_matrix[-1, j] = mahalanobis_distance

        self.new_distance_matrix = new_distance_matrix
        # calculate the shortest path
        distance_matrix_np = new_distance_matrix.cpu().detach().numpy()
        dist, path = dijkstra(distance_matrix_np, directed=True, indices=-1, return_predecessors=True)

        # round distance to 5 decimal places
        dist = np.round(dist, 5)

        if not return_path:
            # filter out the grid distances
            dist = dist[self.grid.shape[0]:-1]
            return dist
        
        if return_path and y is not None and y.shape[0] == 1:
            
            start_index = self.new_distance_matrix.shape[0]-1
            end_index = self.new_distance_matrix.shape[0]-2

            shortest_path = [end_index] 
            while end_index != start_index:
                end_index = path[end_index]
                shortest_path.append(end_index)

            # get the actual points of the indices
            shortest_path = [all[i].cpu().detach().numpy() for i in shortest_path[:-1]]

            #append x
            shortest_path = np.array(shortest_path)
            shortest_path = np.concatenate((shortest_path, x.cpu().detach().numpy()))

            return dist[-2], shortest_path
        
        else:
            dist = dist[self.grid.shape[0]:-1]
            return dist


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
              k_neighbours: int = 0):
        if k_neighbours == 0:
            k_neighbours = self.k_neighbours

        return_indices = []
        return_distances = []

        if isinstance(x, int):
            x = np.array([x])

        for i in range(x.shape[0]):
            dist = self.distance(self.data[x[i]].detach().numpy(), return_path=False)

            # sort data according to distance
            sorted_indices = np.argsort(dist)
            sorted_dist = dist[sorted_indices]

            return_indices.append(sorted_indices[:k_neighbours])
            return_distances.append(sorted_dist[:k_neighbours])

        return return_indices, return_distances
