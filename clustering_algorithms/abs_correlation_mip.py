import numpy as np
from mip import Model, xsum, BINARY, maximize
import torch

def permute_rows_based_on_clusters(M, clusters):
    ordered_indices = [row_index for cluster in clusters for row_index in cluster]
    return M[ordered_indices, :]

def permute_columns_based_on_clusters(M, clusters):
    ordered_indices = [column_index for cluster in clusters for column_index in cluster]
    return M[:, ordered_indices]

def correlation_clustering(X):
    correlation_matrix = np.corrcoef(X)

    print("Correlation Matrix")
    print(correlation_matrix)

    n = correlation_matrix.shape[0]

    # number of clusters
    k = int(np.sqrt(n))
    m = Model(sense=maximize)

    x = [[m.add_var(var_type=BINARY) for j in range(k)] for i in range(n)]

    z = [[[m.add_var(var_type=BINARY) for j in range(k)] for i2 in range(n)] for i in range(n)]

    # each row must be exactly in one cluster
    for i in range(n):
        m += xsum(x[i][j] for j in range(k)) == 1

    # each cluster has k rows
    for j in range(k):
        m += xsum(x[i][j] for i in range(n)) == k

    # z[i][i2][j] = x[i][j] * x[i2][j]
    for i in range(n):
        for i2 in range(i + 1, n):
            for j in range(k):
                m += z[i][i2][j] <= x[i][j]
                m += z[i][i2][j] <= x[i2][j]
                m += z[i][i2][j] >= x[i][j] + x[i2][j] - 1

    obj = xsum(correlation_matrix[i][i2] * z[i][i2][j]
               for j in range(k) for i in range(n) for i2 in range(i + 1, n))

    m.objective = obj

    m.max_seconds = 4000
    #m.gap = 0.05
    m.optimize()

    clusters = [[] for _ in range(k)]
    for i in range(n):
        for j in range(k):
            if x[i][j].x == 1:
                clusters[j].append(i)

    return clusters

if __name__ == "__main__":
    # X = np.array([[1.0, 0.2, 0.1, 0.5, 0.3, 1.0, 0.2, 0.1, 0.5],
    #               [1.0, 0.2, 0.1, 0.5, 0.3, 1.0, 0.2, 0.1, 0.5],
    #               [0.1, 0.4, 1.0, 0.6, 0.2, 0.1, 0.4, 1.0, 0.6],
    #               [0.5, 0.3, 0.6, 1.0, 0.4, 0.5, 0.3, 0.6, 1.0],
    #               [0.3, 0.7, 0.2, 0.4, 1.0, 0.3, 0.7, 0.2, 0.4],
    #               [4.0, 0.2, 0.1, 0.5, 0.3, 1.0, 2.2, 0.1, 0.5],
    #               [1.0, 0.2, 3.1, 3.5, 2.3, 1.0, 0.2, 0.6, 4.5],
    #               [0.1, 3.4, 1.0, 1.6, 2.2, 3.1, 0.4, 1.6, 0.6],
    #               [4.5, 0.3, 0.6, 1.0, 2.4, 0.5, 0.3, 0.6, 4.0]])

    X = torch.load('../inputs/real_weights/pytorch_weights_36x36.pt')

    row_clusters = correlation_clustering(X.T)
    X = permute_rows_based_on_clusters(X, row_clusters)
    column_clusters = correlation_clustering(X)
    X = permute_columns_based_on_clusters(X, column_clusters)

    torch.save(X, '../inputs/real_weights/pytorch_weights_36x36_clustered.pt')