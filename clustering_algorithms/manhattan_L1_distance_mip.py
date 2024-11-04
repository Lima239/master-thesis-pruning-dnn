import numpy as np
from mip import Model, xsum, BINARY, CONTINUOUS, maximize
import torch

def permute_rows_based_on_clusters(M, clusters):
    ordered_indices = [row_index for cluster in clusters for row_index in cluster]
    return M[ordered_indices, :]

def permute_columns_based_on_clusters(M, clusters):
    ordered_indices = [column_index for cluster in clusters for column_index in cluster]
    return M[:, ordered_indices]

def manhattan_distance_matrix(X):
    n = X.shape[0]
    manhattan_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = torch.sum(torch.abs(X[i] - X[j]))
            manhattan_matrix[i, j] = dist
            manhattan_matrix[j, i] = dist
    return manhattan_matrix

def clustering_with_manhattan(X):
    distance_matrix = manhattan_distance_matrix(X)

    print("Manhattan Distance Matrix")
    print(distance_matrix)

    n = distance_matrix.shape[0]

    # Number of clusters
    k = int(np.sqrt(n))
    m = Model(sense=maximize)

    x = [[m.add_var(var_type=BINARY) for j in range(k)] for i in range(n)]
    z = [[[m.add_var(var_type=BINARY) for j in range(k)] for i2 in range(n)] for i in range(n)]

    # Each row must be in exactly one cluster
    for i in range(n):
        m += xsum(x[i][j] for j in range(k)) == 1

    # Each cluster has k rows
    for j in range(k):
        m += xsum(x[i][j] for i in range(n)) == k

    # Define z[i][i2][j] = x[i][j] * x[i2][j]
    for i in range(n):
        for i2 in range(i + 1, n):
            for j in range(k):
                m += z[i][i2][j] <= x[i][j]
                m += z[i][i2][j] <= x[i2][j]
                m += z[i][i2][j] >= x[i][j] + x[i2][j] - 1

    obj = xsum(-distance_matrix[i][i2] * z[i][i2][j]  # minimizing distance by maximizing negative
               for j in range(k) for i in range(n) for i2 in range(i + 1, n))

    m.objective = obj

    m.max_seconds = 100
    # m.gap = 0.05
    m.optimize()

    clusters = [[] for _ in range(k)]
    for i in range(n):
        for j in range(k):
            if x[i][j].x == 1:
                clusters[j].append(i)

    return clusters

if __name__ == "__main__":
    # X = np.array(...)  # Example data

    X = torch.load('../inputs/real_weights/pytorch_weights_36x36.pt')

    row_clusters = clustering_with_manhattan(X.T)
    X = permute_rows_based_on_clusters(X, row_clusters)
    column_clusters = clustering_with_manhattan(X)
    X = permute_columns_based_on_clusters(X, column_clusters)

    torch.save(X, '../inputs/real_weights/pytorch_weights_36x36_manhattan.pt')
