import numpy as np
from mip import Model, xsum, BINARY, maximize

# def dfs(i, visited, adjacency_matrix, n, curr_cluster):
#     visited[i] = True
#     curr_cluster.append(i)
#
#     for j in range(n):
#         if adjacency_matrix[i, j] == 1 and not visited[j]:
#             dfs(i, visited, adjacency_matrix, n, curr_cluster)


# def find_clusters(adjacency_matrix):
#     n = adjacency_matrix.shape[0]
#     visited = [False] * n
#     clusters = []
#
#     for i in range(n):
#         if not visited[i]:
#             curr_cluster = []
#             dfs(i, visited, adjacency_matrix, n, curr_cluster)
#             clusters.append(curr_cluster)
#
#     return clusters

def correlation_clustering(X):
    correlation_matrix = np.corrcoef(X)

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
    m.optimize()

    clusters = [[] for _ in range(k)]
    for i in range(n):
        for j in range(k):
            if x[i][j].x == 1:
                clusters[j].append(i)

    return clusters

if __name__ == "__main__":
    X = np.array([[1.0, 0.2, 0.1, 0.5, 0.3, 1.0, 0.2, 0.1, 0.5],
                  [1.0, 0.2, 0.1, 0.5, 0.3, 1.0, 0.2, 0.1, 0.5],
                  [0.1, 0.4, 1.0, 0.6, 0.2, 0.1, 0.4, 1.0, 0.6],
                  [0.5, 0.3, 0.6, 1.0, 0.4, 0.5, 0.3, 0.6, 1.0],
                  [0.3, 0.7, 0.2, 0.4, 1.0, 0.3, 0.7, 0.2, 0.4],
                  [4.0, 0.2, 0.1, 0.5, 0.3, 1.0, 2.2, 0.1, 0.5],
                  [1.0, 0.2, 3.1, 3.5, 2.3, 1.0, 0.2, 0.6, 4.5],
                  [0.1, 3.4, 1.0, 1.6, 2.2, 3.1, 0.4, 1.6, 0.6],
                  [4.5, 0.3, 0.6, 1.0, 2.4, 0.5, 0.3, 0.6, 4.0]])

    row_clusters = correlation_clustering(X.T)
    column_clusters = correlation_clustering(X)

    print(row_clusters)