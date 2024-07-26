import numpy as np

def dijkstra(graph: np.ndarray,
             start: int,
             end: int) -> np.ndarray:    
    n = graph.shape[0]
    print(n, 'n')
    unvisited = set(range(n))
    print(unvisited)
    distance = [float('inf')] * n
    distance[start] = 0
    while end in unvisited:
        print(len(unvisited))
        current = min(unvisited, key=lambda node: distance[node])
        unvisited.remove(current)
        for neighbour in range(n):
            if graph[current][neighbour] != float('inf') and neighbour in unvisited:
                print('CALCULATE NEW PATH')
                new_path = distance[current] + graph[current][neighbour]
                if new_path < distance[neighbour]:
                    print('UPDATE PATH')
                    distance[neighbour] = new_path
    print('END')
    return distance[end]