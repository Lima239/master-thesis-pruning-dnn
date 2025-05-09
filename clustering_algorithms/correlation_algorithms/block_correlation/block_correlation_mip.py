import numpy as np
from mip import Model, xsum, BINARY, CONTINUOUS, MAXIMIZE, GRB
import torch
import sys
import os
from scipy.spatial.distance import cdist

def permute_rows_based_on_clusters(M, clusters):
    ordered_indices = [row_index for cluster in clusters for row_index in cluster]
    return M[ordered_indices, :]

def permute_columns_based_on_clusters(M, clusters):
    ordered_indices = [column_index for cluster in clusters for column_index in cluster]
    return M[:, ordered_indices]

def block_cosine_dist_clustering(X, k, len_of_run):
    submatrices = [X[:, i * k:(i + 1) * k] for i in range(k)]

    # cosine similarity matrix for each submatrix
    cosine_matrices = [abs(1 - cdist(submatrix, submatrix, metric='cosine')) for submatrix in submatrices]

    # compute the average cosine similarity matrix across all blocks
    average_cosine_matrix = sum(cosine_matrices) / len(cosine_matrices)

    n = X.shape[0]
    if n % k != 0:
        raise ValueError(f"Invalid number of clusters: {k}.")

    m = Model(sense=MAXIMIZE, solver_name=GRB)

    x = [[m.add_var(var_type=BINARY) for j in range(k)] for i in range(n)]

    z = [[[m.add_var(var_type=BINARY) for j in range(k)] for i2 in range(n)] for i in range(n)]

    # each row must be exactly in one cluster
    for i in range(n):
        m += xsum(x[i][j] for j in range(k)) == 1

    # each cluster has k rows
    for j in range(k):
        m += xsum(x[i][j] for i in range(n)) == n/k

    # z[i][i2][j] = x[i][j] * x[i2][j]
    for i in range(n):
        for i2 in range(i + 1, n):
            for j in range(k):
                m += z[i][i2][j] <= x[i][j]
                m += z[i][i2][j] <= x[i2][j]
                m += z[i][i2][j] >= x[i][j] + x[i2][j] - 1

    obj = xsum(average_cosine_matrix[i][i2] * z[i][i2][j]
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

    return torch.tensor(clusters).flatten()

def compute_block_cosine_permutations(X, k_rows, l_columns, len_of_run):
    row_clusters_perm = block_cosine_dist_clustering(X, k_rows, len_of_run)
    column_clusters_perm = block_cosine_dist_clustering(X.T, l_columns, len_of_run)

    return row_clusters_perm, column_clusters_perm

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 block_correlation_mip.py <input_file_path>")
    else:
        input_path = sys.argv[1]
        print("block_correlation_mip.py")
        print(input_path)

        X = torch.load(input_path)
        len_of_run = 100 # in seconds
        #gap_of_run = 10

        k = int(np.sqrt(X.shape[0]))

        P_rows, P_columns = compute_block_cosine_permutations(X, k,k, len_of_run)
        X = X[P_rows,:]
        X = X[:,P_columns]

        # saving clustered matrix
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_name = os.path.splitext(os.path.basename(input_path))[0]
        save_name = f"{file_name}_{len_of_run}.pt"

        save_path = os.path.join(output_dir, save_name)
        torch.save(X, save_path)