import LAMINAR
import torch
import numpy as np

import LAMINAR.Flow


def test_LAMINAR():
    assert LAMINAR.add_one(1) == 2
    
    data = torch.rand(100, 2)
    LAM = LAMINAR.LAMINAR(data)

    # check basic structures
    assert LAM.data.shape == data.shape
    assert LAM.data_pushed.shape == data.shape
    assert LAM.k_neighbours == 20
    assert LAM.dimension == 2
    assert LAM.flow.device == torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert type(LAM.flow).__name__ == LAMINAR.Flow.planarCNF.PlanarCNF.__name__

    # check metric properties:
    
    # build full distance matrix
    dist_matrix = np.zeros((100, 100))
    for i in range(100):
        dist = LAM.distance(i)[0]
        dist_matrix[i, :] = dist

    # symmetry
    assert (dist_matrix == dist_matrix.T).all()

    # positive definiteness
    assert (np.diag(dist_matrix) == 0).all()
    assert (dist_matrix >= 0).all()

    # triangle inequality
    for i in range(100):
        for j in range(100):
            for k in range(100):
                assert dist_matrix[i, j] <= dist_matrix[i, k] + dist_matrix[k, j]

    