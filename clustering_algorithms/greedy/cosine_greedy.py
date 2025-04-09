import torch
import numpy as np
from scipy.spatial.distance import cdist

def greedy_cosine_clustering(X, k):

    X_np = X.detach().cpu().numpy()
    n = X_np.shape[0]
    if n % k != 0:
        raise ValueError(f"Invalid number of clusters: {k}. Cannot evenly divide {n} samples.")

    cluster_size = n // k
    permuted_indices = []
    assigned = set()

    sim_matrix = 1 - cdist(X_np, X_np, metric="cosine")  # cosine similarity
    np.fill_diagonal(sim_matrix, -np.inf)

    for cluster_idx in range(k):
        remaining = list(set(range(n)) - assigned)
        scores = sim_matrix[remaining][:, remaining].sum(axis=1)
        seed_idx = remaining[np.argmax(scores)]

        candidates = np.argsort(sim_matrix[seed_idx])[::-1]
        selected = []
        for idx in candidates:
            if idx not in assigned:
                selected.append(idx)
            if len(selected) == cluster_size - 1:
                break
        selected.append(seed_idx)
        assigned.update(selected)
        permuted_indices.extend(selected)

    return torch.tensor(permuted_indices)


def compute_clustering_permutations_cos_GREEDY(X, k1, k2):
    row_perm = greedy_cosine_clustering(X, k1)
    col_perm = greedy_cosine_clustering(X.T, k2)
    return row_perm, col_perm
