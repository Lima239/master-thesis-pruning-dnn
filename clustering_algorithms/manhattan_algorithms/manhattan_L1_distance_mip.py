import numpy as np
from mip import Model, xsum, BINARY, CONTINUOUS, MAXIMIZE
import torch
import sys
import os

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

def clustering_with_manhattan(X, len_of_run):
    distance_matrix = manhattan_distance_matrix(X)

#    print("Manhattan Distance Matrix")
#    print(distance_matrix)

    n = distance_matrix.shape[0]

    # Number of clusters
    k = int(np.sqrt(n))
    m = Model(sense=MAXIMIZE)

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
#                m += z[i][i2][j] >= x[i][j] + x[i2][j] - 1

    # minimizing distance by maximizing negative
    obj = xsum(-distance_matrix[i][i2] * z[i][i2][j]
               for j in range(k) for i in range(n) for i2 in range(i + 1, n))

    m.objective = obj

    m.max_seconds = len_of_run
    # m.gap = 0.05
    m.optimize()

    clusters = [[] for _ in range(k)]
    for i in range(n):
        for j in range(k):
            if x[i][j].x == 1:
                clusters[j].append(i)

    return clusters

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 manhattan_L1_distance_mip.py.py <input_file_path>")
    else:
        input_path = sys.argv[1]
        print("manhattan_L1_distance_mip.py")
        print(input_path)

        X = torch.load(input_path)
        len_of_run = 4000 # in seconds
        #gap_of_run = 10

        row_clusters = clustering_with_manhattan(X.T, len_of_run)
        X = permute_rows_based_on_clusters(X, row_clusters)
        column_clusters = clustering_with_manhattan(X, len_of_run)
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
