import numpy as np
from mip import Model, xsum, BINARY, maximize

def dfs(i, visited, adjacency_matrix, n, curr_cluster):
    visited[i] = True
    curr_cluster.append(i)

    for j in range(n):
        if adjacency_matrix[i, j] == 1 and not visited[j]:
            dfs(i, visited, adjacency_matrix, n, curr_cluster)


def find_clusters(adjacency_matrix):
    n = adjacency_matrix.shape[0]
    visited = [False] * n
    clusters = []

    for i in range(n):
        if not visited[i]:
            curr_cluster = []
            dfs(i, visited, adjacency_matrix, n, curr_cluster)
            clusters.append(curr_cluster)

    return clusters

def cluster_by_correlation(X, num_clusters):
    correlation_matrix = np.corrcoef(X)
    A = np.abs(correlation_matrix)
    n = A.shape[0] #num of rows
    mip_model = Model(sense=maximize)

    #variables x[i,j]=1 if rows rows i and j are in the same cluster
    x = [[mip_model.add_var(var_type=BINARY) for j in range(n)] for i in range(n)]

    clusters = [mip_model.add_var(var_type=BINARY) for i in range(n)]
    #maximize sum of correlations within cluster
    mip_model.objective = xsum(A[i, j] * x[i][j] for i in range(n) for j in range(i, n))

    mip_model.optimize()

    cluster_adjacency_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            cluster_adjacency_matrix[i, j] = x[i][j].x

    clusters = find_clusters(cluster_adjacency_matrix)

    return clusters

if __name__ == "__main__":
    X = np.array([[1.0, 0.2, 0.1, 0.5, 0.3],
                  [1.0, 0.2, 0.1, 0.5, 0.3],
                  [0.1, 0.4, 1.0, 0.6, 0.2],
                  [0.5, 0.3, 0.6, 1.0, 0.4],
                  [0.3, 0.7, 0.2, 0.4, 1.0]])

    clusters = cluster_by_correlation(X, num_clusters=3)

    print("Clusters:", clusters)