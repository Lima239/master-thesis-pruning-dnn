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

def permute_matrix_with_clusters(X, clusters, axis=0):
    n = X.shape[0] if axis == 0 else X.shape[1]
    ordered_indices = [idx for cluster in clusters for idx in cluster]
    permutation_matrix = np.eye(n)[ordered_indices]
    if axis == 0:  # Row permutation
        return permutation_matrix @ X, permutation_matrix
    else:  # Column permutation
        return X @ permutation_matrix.T, permutation_matrix.T

# clustering rows of X matrix
def correlation_clustering(X, k, len_of_run):
    correlation_matrix = np.corrcoef(X)

    n = X.shape[0]
    if n % k != 0:
        raise ValueError(f"Invalid number of clusters: {k}.")

    # number of clusters
    # k = int(np.sqrt(n))
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

def clustering_permutation_matrices(X, k, len_of_run):
    row_clusters = correlation_clustering(X.T, k, len_of_run)
    X, P_rows = permute_matrix_with_clusters(X, row_clusters, 0)
    column_clusters = correlation_clustering(X, k, len_of_run)
    X, P_columns = permute_columns_based_on_clusters(X, column_clusters, 1)

    return P_rows, P_columns

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

        k = int(np.sqrt(X.shape[0]))

        row_clusters = correlation_clustering(X.T, k, len_of_run)
        X = permute_rows_based_on_clusters(X, row_clusters)
        column_clusters = correlation_clustering(X, k, len_of_run)
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