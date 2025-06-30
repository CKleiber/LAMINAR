import numpy as np
import numba as nb


# jit compiled dijkstra adjusted for sparse matrices and LAMINAR's use case
def dijkstra(graph, start_nodes, end_nodes):
    n = graph.shape[0]
    indptr = graph.indptr.astype(np.int32)
    indices = graph.indices.astype(np.int32)
    data = graph.data.astype(np.float32)
    return _dijkstra(n, indptr, indices, data, start_nodes, end_nodes)

@nb.njit
def _dijkstra(n, indptr, indices, data, start_nodes, end_nodes):
    inf = float('inf')

    distance_array = []
    path_array = []

    for start in start_nodes:
        distances = np.full(n, inf, dtype=np.float32)
        predecessors = np.full(n, -1, dtype=np.int32)
        visited = np.zeros(n, dtype=np.bool_)

        distances[start] = 0.0

        for _ in range(n):
            # Find the node with the smallest distance that hasn't been visited
            min_dist = inf
            u = -1
            for i in range(n):
                if not visited[i] and distances[i] < min_dist:
                    min_dist = distances[i]
                    u = i

            if u == -1 or min_dist == inf:
                break

            visited[u] = True

            # Update the distances to the neighbors
            for idx in range(indptr[u], indptr[u + 1]):
                v = indices[idx]
                length = data[idx]
                if not visited[v] and distances[u] + length < distances[v]:
                    distances[v] = distances[u] + length
                    predecessors[v] = u

        # Extract the shortest path distances to the end nodes
        end_distances = np.array([distances[end] for end in end_nodes])

        end_paths = []
        for end in end_nodes:
            path = []
            while end != -1:
                path.append(end)
                end = predecessors[end]
            path.reverse()
            end_paths.append(path)

        distance_array.append(end_distances)
        path_array.append(end_paths)

    return distance_array, path_array    
