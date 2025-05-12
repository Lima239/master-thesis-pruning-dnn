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

def permute_matrix_with_clusters(X, clusters, axis=0):
    n = X.shape[0] if axis == 0 else X.shape[1]
    ordered_indices = [idx for cluster in clusters for idx in cluster]
    permutation_matrix = torch.eye(n, dtype=X.dtype, device=X.device)[ordered_indices]
    if axis == 0:  # Row permutation
        return permutation_matrix @ X, permutation_matrix
    else:  # Column permutation
        return X @ permutation_matrix.T, permutation_matrix.T

# clustering rows of X matrix
# k ---> number of clusters
def cosine_clustering(X, k, len_of_run):
    cosine_matrix = cdist(X, X, metric='cosine')
    cosine_matrix = abs(1 - cosine_matrix)

    n = X.shape[0]
    if n % k != 0:
        raise ValueError(f"Invalid number of clusters: {k}.")

    # number of clusters
    m = Model(sense=MAXIMIZE, solver_name=GRB)

    x = [[m.add_var(var_type=BINARY) for j in range(k)] for i in range(n)]

    z = [[[m.add_var(var_type=BINARY) for j in range(k)] for i2 in range(n)] for i in range(n)]

    # each row must be exactly in one cluster
    for i in range(n):
        m += xsum(x[i][j] for j in range(k)) == 1

    # each cluster has n/k rows
    for j in range(k):
        m += xsum(x[i][j] for i in range(n)) == n/k

    # z[i][i2][j] = x[i][j] * x[i2][j]
    for i in range(n):
        for i2 in range(i + 1, n):
            for j in range(k):
                m += z[i][i2][j] <= x[i][j]
                m += z[i][i2][j] <= x[i2][j]
                m += z[i][i2][j] >= x[i][j] + x[i2][j] - 1

    obj = xsum(cosine_matrix[i][i2] * z[i][i2][j]
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

    #print(clusters)
    return torch.tensor(clusters).flatten()

def compute_clustering_permutations(X, k_rows, l_columns, len_of_run):
    row_clusters_perm = cosine_clustering(X, k_rows, len_of_run)
    column_clusters_perm = cosine_clustering(X.T, l_columns, len_of_run)

    return row_clusters_perm, column_clusters_perm

def compute_row_permutation(X, k, len_of_run):
    row_clusters = cosine_clustering(X.T, k, len_of_run)
    X, P_rows = permute_matrix_with_clusters(X, row_clusters, 0)

    return X, P_rows
def compute_column_permutation(X, k, len_of_run):
    column_clusters = cosine_clustering(X, k, len_of_run)
    X, P_columns = permute_columns_based_on_clusters(X, column_clusters, 1)

    return X, P_columns

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python3 cosine_dist_mip.py <input_file_path>")
    # else:
    #     input_path = sys.argv[1]
        print("cosine_dist_mip.py")
        input_path = "/inputs/fc1.pt"
        print(input_path)

        X = torch.load(input_path)
        print(X.shape)
        len_of_run = 2000  # in seconds

        P_rows, P_columns = compute_clustering_permutations(X, 4, 4, len_of_run)
        X = X[P_rows,:]
        X = X[:,P_columns]
        # saving clustered matrix
        # script_dir = os.path.dirname(os.path.abspath(__file__))
        # output_dir = os.path.join(script_dir, "output")
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # file_name = os.path.splitext(os.path.basename(input_path))[0]
        # save_name = f"{file_name}_{len_of_run}.pt"
        #
        # save_path = os.path.join(output_dir, save_name)
        # torch.save(X, save_path)
        # X = torch.tensor([
        #     [10, 20, 30, 40, 50, 60, 70, 80, 90],  # Outlier
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1],  # Cluster 4 (identical to row 7)
        #     [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Cluster 1
        #     [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Cluster 1 (identical to row 1)
        #     [2, 4, 6, 8, 10, 12, 14, 16, 18],  # Cluster 2
        #     [2, 4, 6, 8, 10, 12, 14, 16, 18],  # Cluster 2 (identical to row 3)
        #     # [3, 6, 9, 12, 15, 18, 21, 24, 27],  # Cluster 3
        #     # [3, 6, 9, 12, 15, 18, 21, 24, 27],  # Cluster 3 (identical to row 5)
        #     # [1, 1, 1, 1, 1, 1, 1, 1, 1],  # Cluster 4 (constant row)
        # ])
        #
        #
        #
        # k = int(np.sqrt(X.shape[0]))
        # P_rows, P_columns = compute_clustering_permutations(X, 3, 3, 30)
        # print(P_rows)
        # print(P_columns)
        #
        # print(X)
        # #X = X[P_rows, :]
        # X = X.index_select(-2, P_rows)
        # print(X)
        # #X = X.T[P_columns,:].T
        # X = X.index_select(-1, P_columns)
        # print(X)

        # k = int(np.sqrt(X.shape[0]))
        # len_of_run = 100 # in seconds
        #
        #
        # #X = torch.load(input_path)
        # len_of_run = 100 # in seconds
        # #gap_of_run = 10
        #
        # k = int(np.sqrt(X.shape[0]))
        # S, P_rows = compute_row_permutation(X, k, len_of_run)
        # M = P_rows @ X
        # print(M)

        X = torch.load(input_path)
        print(X.size())
        len_of_run = 100 # in seconds
        k = int(np.sqrt(X.shape[0]))

        P_rows, P_columns = compute_clustering_permutations(X, k, k, len_of_run)
        X = X.index_select(-2, P_rows)
        X = X.index_select(-1, P_columns)

        # saving clustered matrix
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_name = os.path.splitext(os.path.basename(input_path))[0]
        save_name = f"{file_name}_{len_of_run}.pt"

        save_path = os.path.join(output_dir, save_name)
        torch.save(X, save_path)



        # row_clusters = cosine_clustering(X.T, k, len_of_run)
        # X = permute_rows_based_on_clusters(X, row_clusters)
        # print(row_clusters)
        # print(X.size())
        #
        # column_clusters = cosine_clustering(X, k, len_of_run)
        # X = permute_columns_based_on_clusters(X, column_clusters)
        # print(column_clusters)
        # print(X.size())