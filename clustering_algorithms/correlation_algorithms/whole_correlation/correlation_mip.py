import numpy as np
from mip import Model, xsum, BINARY, CONTINUOUS, maximize
import torch
import sys
import os

def permute_rows_based_on_clusters(M, clusters):
    ordered_indices = [row_index for cluster in clusters for row_index in cluster]
    return M[ordered_indices, :]

def permute_columns_based_on_clusters(M, clusters):
    ordered_indices = [column_index for cluster in clusters for column_index in cluster]
    return M[:, ordered_indices]

def correlation_clustering(X, len_of_run):
    correlation_matrix = np.corrcoef(X)

    # print("Correlation Matrix")
    # n = correlation_matrix.shape[0]

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

    m.max_seconds = len_of_run
    #m.gap = 0.05
    m.optimize()

    clusters = [[] for _ in range(k)]
    for i in range(n):
        for j in range(k):
            if x[i][j].x == 1:
                clusters[j].append(i)

    return clusters

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 correlation_mip.py <input_file_path>")
    else:
        input_path = sys.argv[1]
        print("correlation_mip.py")
        print(input_path)

        X = torch.load(input_path)
        len_of_run = 4000 # in seconds
        #gap_of_run = 10

        row_clusters = correlation_clustering(X.T, len_of_run)
        X = permute_rows_based_on_clusters(X, row_clusters)
        column_clusters = correlation_clustering(X, len_of_run)
        X = permute_columns_based_on_clusters(X, column_clusters)

        # saving clustered matrix
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_name = os.path.splitext(os.path.basename(input_path))[0]
        save_name = f"{file_name}_{len_of_run}.pt"

        save_path = os.path.join(output_dir, save_name)
        torch.save(X, save_path)


    # X = np.array([[1.0, 0.2, 0.1, 0.5, 0.3, 1.0, 0.2, 0.1, 0.5],
    #               [1.0, 0.2, 0.1, 0.5, 0.3, 1.0, 0.2, 0.1, 0.5],
    #               [0.1, 0.4, 1.0, 0.6, 0.2, 0.1, 0.4, 1.0, 0.6],
    #               [0.5, 0.3, 0.6, 1.0, 0.4, 0.5, 0.3, 0.6, 1.0],
    #               [0.3, 0.7, 0.2, 0.4, 1.0, 0.3, 0.7, 0.2, 0.4],
    #               [4.0, 0.2, 0.1, 0.5, 0.3, 1.0, 2.2, 0.1, 0.5],
    #               [1.0, 0.2, 3.1, 3.5, 2.3, 1.0, 0.2, 0.6, 4.5],
    #               [0.1, 3.4, 1.0, 1.6, 2.2, 3.1, 0.4, 1.6, 0.6],
    #               [4.5, 0.3, 0.6, 1.0, 2.4, 0.5, 0.3, 0.6, 4.0]])