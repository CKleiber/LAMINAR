import LAMINAR
import torch

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
    assert LAM.distance_matrix.shape == (100, 100)
    assert LAM.cov_matrices.shape == (100, 2, 2)

    # assert index of min of loss hist is in the last 5 elements
    assert min(LAM.loss_history) == min(LAM.loss_history[-5:])

    # assert query
    assert LAM.query([0]).shape == (1, 20)
    assert LAM.query([0], k_neighbours=5).shape == (1, 5)

    # check metric properties:
    
    for i in range(100):
        for j in range(100):
            # positive semi-definite
            d = LAM.distance(i, j)
            assert d >= 0
            if d == 0:
                assert i == j
            if i == j:
                assert d == 0

            # symmetry
            assert d == LAM.distance(j, i)

            # triangle inequality
            for k in range(100):
                assert LAM.distance(i, j) <= LAM.distance(i, k) + LAM.distance(k, j)

    
    # check standardisation
    assert LAMINAR.standardize(data).shape == data.shape
    assert LAMINAR.standardize(data).mean() == 0
    assert LAMINAR.standardize(data).std() == 1
    